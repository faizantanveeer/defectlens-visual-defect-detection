"""Production predictor: ONNX Runtime + the shared preprocessing. No torch dependency.

A *model bundle* is a directory containing

    model.onnx       - graph: input "image" (N,3,S,S) float32 -> "defect_probability" (N,)
    metadata.json    - version, class names, preprocessing, decision threshold,
                       review margin, model sha256, evaluation summary

The decision threshold travels with the model, so retuning the operating point is a
metadata change that is versioned together with the weights it was tuned for.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from defect_detection.preprocessing import PreprocessSpec, preprocess_batch

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    predicted_class: str
    confidence: float
    defect_probability: float
    threshold: float
    review_recommended: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Timing:
    preprocess_ms: float
    inference_ms: float


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DefectPredictor:
    def __init__(
        self,
        model_dir: str | Path,
        intra_op_threads: int | None = None,
        verify_checksum: bool = True,
        threshold_override: float | None = None,
    ) -> None:
        import onnxruntime as ort

        model_dir = Path(model_dir)
        model_path, meta_path = model_dir / "model.onnx", model_dir / "metadata.json"
        if not model_path.is_file() or not meta_path.is_file():
            raise ModelLoadError(f"model bundle incomplete in {model_dir} (need model.onnx + metadata.json)")
        self.metadata: dict = json.loads(meta_path.read_text(encoding="utf-8"))
        if verify_checksum and self.metadata.get("model_sha256"):
            actual = sha256_file(model_path)
            if actual != self.metadata["model_sha256"]:
                raise ModelLoadError("model.onnx checksum does not match metadata.json")

        self.spec = PreprocessSpec.from_dict(self.metadata["preprocess"])
        self.class_names: list[str] = list(self.metadata["class_names"])
        self.threshold = float(
            threshold_override if threshold_override is not None else self.metadata["threshold"]
        )
        if not 0.0 < self.threshold < 1.0:
            raise ModelLoadError(f"invalid threshold {self.threshold}")
        self.review_margin = float(self.metadata.get("review_margin", 0.0))
        self.version: str = self.metadata.get("model_version", "unknown")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        threads = intra_op_threads or int(os.environ.get("ORT_INTRA_OP_THREADS", "0"))
        if threads:
            opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log.info(
            "model loaded",
            extra={"model_version": self.version, "threshold": self.threshold, "dir": str(model_dir)},
        )

    def warmup(self, runs: int = 2) -> None:
        dummy = np.zeros((1, 3, self.spec.image_size, self.spec.image_size), dtype=np.float32)
        for _ in range(runs):
            self.session.run([self.output_name], {self.input_name: dummy})

    def predict_proba_array(self, batch: np.ndarray) -> np.ndarray:
        out = self.session.run([self.output_name], {self.input_name: batch.astype(np.float32)})[0]
        return np.asarray(out, dtype=np.float64).reshape(-1)

    def _to_prediction(self, p: float) -> Prediction:
        defective = p >= self.threshold
        name = self.class_names[1] if defective else self.class_names[0]
        confidence = p if defective else 1.0 - p
        return Prediction(
            predicted_class=name,
            confidence=round(float(confidence), 6),
            defect_probability=round(float(p), 6),
            threshold=self.threshold,
            review_recommended=bool(abs(p - self.threshold) < self.review_margin),
        )

    def predict(self, images: list[Image.Image]) -> tuple[list[Prediction], Timing]:
        if not images:
            return [], Timing(0.0, 0.0)
        t0 = time.perf_counter()
        batch = preprocess_batch(images, self.spec)
        t1 = time.perf_counter()
        probs = self.predict_proba_array(batch)
        t2 = time.perf_counter()
        return [self._to_prediction(float(p)) for p in probs], Timing(
            round((t1 - t0) * 1000, 3), round((t2 - t1) * 1000, 3)
        )
