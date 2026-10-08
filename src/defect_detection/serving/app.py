"""FastAPI inference service.

Run:  uvicorn defect_detection.serving.app:create_app --factory --port 8000
Docs: http://localhost:8000/docs

Design notes
* Stateless; one ONNX session per process, scaled horizontally (replicas), not via
  in-process worker pools. ORT threads are capped per process to avoid oversubscription.
* CPU work (decode, preprocess, inference) runs in the threadpool so the event loop
  keeps serving health checks and metrics while inferences are in flight.
* Backpressure: a bounded semaphore limits in-flight inferences; overload returns 503
  with Retry-After instead of growing latency without bound.
* Every response carries X-Request-ID; every log line is JSON with the same id.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from defect_detection import __version__
from defect_detection.inference.predictor import DefectPredictor
from defect_detection.logging_utils import configure_logging, request_id_ctx
from defect_detection.preprocessing import ImageDecodeError, decode_image
from defect_detection.serving.schemas import (
    BatchItem,
    BatchPredictResponse,
    ErrorBody,
    ErrorResponse,
    ModelInfo,
    PredictionOut,
    PredictResponse,
    TimingOut,
)
from defect_detection.serving.settings import Settings, get_settings

log = logging.getLogger("defect_detection.api")


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None) -> None:
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


class Metrics:
    """Per-app registry (no global state -> safe to build many apps in tests)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("vdd_http_requests_total", "HTTP requests", ["path", "method", "status"], registry=r)
        self.latency = Histogram(
            "vdd_http_request_seconds", "End-to-end request latency", ["path"], registry=r,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
        )
        self.inference = Histogram(
            "vdd_inference_seconds", "Model inference latency per call", registry=r,
            buckets=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1),
        )
        self.predictions = Counter("vdd_predictions_total", "Predictions by class", ["predicted_class"], registry=r)
        self.review = Counter("vdd_review_recommended_total", "Predictions flagged for review", registry=r)
        # Score distribution is the cheapest production drift signal: if the share of
        # mid-range scores rises, the input distribution has probably shifted.
        self.scores = Histogram(
            "vdd_defect_probability", "Distribution of P(defective)", registry=r,
            buckets=tuple(round(i / 20, 2) for i in range(1, 21)),
        )
        self.rejected = Counter("vdd_rejected_inputs_total", "Invalid inputs", ["reason"], registry=r)
        self.inflight = Gauge("vdd_inflight_inferences", "Inferences currently running", registry=r)
        self.model_info = Gauge("vdd_model_info", "Loaded model", ["model_version", "threshold"], registry=r)


def create_app(settings: Settings | None = None, predictor: DefectPredictor | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.json_logs)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail fast: a pod without a valid model must crash visibly, not serve errors.
        pred = predictor or DefectPredictor(
            settings.model_dir,
            intra_op_threads=settings.ort_intra_op_threads or None,
            verify_checksum=settings.verify_model_checksum,
            threshold_override=settings.threshold_override,
        )
        pred.warmup()
        app.state.predictor = pred
        app.state.semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)
        metrics.model_info.labels(pred.version, f"{pred.threshold:.4f}").set(1)
        log.info("service ready", extra={"model_version": pred.version, "threshold": pred.threshold,
                                          "threshold_overridden": settings.threshold_override is not None})
        yield
        log.info("service shutting down")

    app = FastAPI(
        title="Visual Defect Detection API",
        version=__version__,
        description="Classifies product images as **normal** or **defective**.",
        lifespan=lifespan,
        responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
                   415: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
                   503: {"model": ErrorResponse}},
    )
    app.state.metrics = metrics
    app.state.settings = settings

    # ------------------------------------------------------------------ middleware
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_ctx.set(rid)
        start = time.perf_counter()
        try:
            length = int(request.headers.get("content-length") or 0)
            limit = settings.max_upload_bytes * settings.max_batch_size + 1_000_000
            if length > limit:  # reject oversized bodies before reading them
                response = _error_response(413, "payload_too_large", f"request body exceeds {limit} bytes", rid)
            else:
                response = await call_next(request)
        except Exception:  # noqa: BLE001 - last-resort guard, details go to the log only
            log.exception("unhandled error")
            response = _error_response(500, "internal_error", "internal server error", rid)
        finally:
            request_id_ctx.reset(token)
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        metrics.requests.labels(path, request.method, str(response.status_code)).inc()
        metrics.latency.labels(path).observe(elapsed)
        response.headers["X-Request-ID"] = rid
        if path not in ("/health/live", "/health/ready", "/metrics"):
            log.info("request", extra={"request_id": rid, "method": request.method, "path": path,
                                       "status": response.status_code, "latency_ms": round(elapsed * 1000, 2)})
        return response

    # ------------------------------------------------------------ error handlers
    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError):
        return _error_response(exc.status, exc.code, exc.message, _rid(request), exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return _error_response(422, "invalid_request", msg, _rid(request))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        return _error_response(exc.status_code, "http_error", str(exc.detail), _rid(request))

    # ------------------------------------------------------------------- helpers
    async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise APIError(401, "unauthorized", "missing or invalid X-API-Key")

    async def read_and_decode(file: UploadFile) -> tuple:
        ctype = (file.content_type or "application/octet-stream").split(";")[0].strip().lower()
        if ctype not in settings.allowed_content_types:
            metrics.rejected.labels("content_type").inc()
            raise APIError(415, "unsupported_media_type", f"content type '{ctype}' not allowed")
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            metrics.rejected.labels("too_large").inc()
            raise APIError(413, "payload_too_large", f"file exceeds {settings.max_upload_bytes} bytes")
        try:
            return await run_in_threadpool(
                decode_image, data, min_side=settings.min_image_side, max_pixels=settings.max_image_pixels
            )
        except ImageDecodeError as exc:
            metrics.rejected.labels("decode").inc()
            log.warning("input rejected", extra={"file_name": file.filename, "bytes": len(data),
                                                 "reason": str(exc), "cause": repr(exc.__cause__)})
            raise APIError(422, "invalid_image", str(exc)) from exc

    async def infer(request: Request, images: list) -> tuple[list, object]:
        sem: asyncio.Semaphore = request.app.state.semaphore
        try:
            await asyncio.wait_for(sem.acquire(), timeout=settings.queue_timeout_s)
        except asyncio.TimeoutError as exc:
            metrics.rejected.labels("overloaded").inc()
            raise APIError(503, "overloaded", "server busy, retry shortly", {"Retry-After": "1"}) from exc
        metrics.inflight.inc()
        try:
            preds, timing = await run_in_threadpool(request.app.state.predictor.predict, images)
        finally:
            metrics.inflight.dec()
            sem.release()
        metrics.inference.observe(timing.inference_ms / 1000)
        for p in preds:
            metrics.predictions.labels(p.predicted_class).inc()
            metrics.scores.observe(p.defect_probability)
            if p.review_recommended:
                metrics.review.inc()
        return preds, timing

    # ------------------------------------------------------------------ routes
    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": "visual-defect-detection", "version": __version__, "docs": "/docs"}

    @app.get("/health/live", tags=["ops"])
    async def live():
        return {"status": "ok"}

    @app.get("/health/ready", tags=["ops"])
    async def ready(request: Request):
        if getattr(request.app.state, "predictor", None) is None:
            raise APIError(503, "not_ready", "model not loaded")
        return {"status": "ready", "model_version": request.app.state.predictor.version}

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    async def prometheus_metrics():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/model", response_model=ModelInfo, tags=["model"])
    async def model_info(request: Request):
        p: DefectPredictor = request.app.state.predictor
        m = p.metadata
        return ModelInfo(
            model_version=p.version, backbone=m.get("backbone"), class_names=p.class_names,
            threshold=p.threshold, threshold_strategy=m.get("threshold_strategy"),
            review_margin=p.review_margin, input_size=p.spec.image_size,
            created_utc=m.get("created_utc"), val_average_precision=m.get("val_average_precision"),
        )

    @app.post("/v1/predict", response_model=PredictResponse, tags=["inference"],
              dependencies=[Depends(require_api_key)])
    async def predict(request: Request, file: UploadFile = File(..., description="Product image")):
        t0 = time.perf_counter()
        image = await read_and_decode(file)
        t1 = time.perf_counter()
        preds, timing = await infer(request, [image])
        total = (time.perf_counter() - t0) * 1000
        pred = preds[0]
        log.info("prediction", extra={"file_name": file.filename, **pred.to_dict()})
        return PredictResponse(
            request_id=_rid(request),
            model_version=request.app.state.predictor.version,
            filename=file.filename,
            prediction=PredictionOut(**pred.to_dict()),
            timing=TimingOut(decode_ms=round((t1 - t0) * 1000, 3), preprocess_ms=timing.preprocess_ms,
                             inference_ms=timing.inference_ms, total_ms=round(total, 3)),
        )

    @app.post("/v1/predict/batch", response_model=BatchPredictResponse, tags=["inference"],
              dependencies=[Depends(require_api_key)])
    async def predict_batch(request: Request, files: list[UploadFile] = File(..., description="Images")):
        """One forward pass for the whole batch. Invalid images are reported per item
        instead of failing the batch, so one bad frame does not block a production lot."""
        if len(files) > settings.max_batch_size:
            raise APIError(413, "batch_too_large", f"max {settings.max_batch_size} files per batch")
        t0 = time.perf_counter()
        items: list[BatchItem] = []
        images, positions = [], []
        for f in files:
            try:
                images.append(await read_and_decode(f))
                positions.append(len(items))
                items.append(BatchItem(filename=f.filename))
            except APIError as exc:
                items.append(BatchItem(filename=f.filename, error=ErrorBody(code=exc.code, message=exc.message)))
        t1 = time.perf_counter()
        preds, timing = await infer(request, images) if images else ([], None)
        for pos, p in zip(positions, preds, strict=True):
            items[pos].prediction = PredictionOut(**p.to_dict())
        total = (time.perf_counter() - t0) * 1000
        return BatchPredictResponse(
            request_id=_rid(request),
            model_version=request.app.state.predictor.version,
            results=items,
            timing=TimingOut(decode_ms=round((t1 - t0) * 1000, 3),
                             preprocess_ms=timing.preprocess_ms if timing else 0.0,
                             inference_ms=timing.inference_ms if timing else 0.0, total_ms=round(total, 3)),
        )

    return app


def _rid(request: Request) -> str:
    return request_id_ctx.get() or request.headers.get("x-request-id") or ""


def _error_response(status: int, code: str, message: str, rid: str | None, headers: dict | None = None):
    body = ErrorResponse(error=ErrorBody(code=code, message=message), request_id=rid)
    return JSONResponse(status_code=status, content=body.model_dump(), headers=headers)
