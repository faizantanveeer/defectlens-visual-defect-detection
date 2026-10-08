# Inference image: onnxruntime + FastAPI only (no torch) -> small, fast cold start.
# Build:  docker build -t defect-detection-api:latest .
# Run:    docker run -p 8000:8000 defect-detection-api:latest

# ---------------------------------------------------------------- builder
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements/serve.txt .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r serve.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN /opt/venv/bin/pip install --no-deps .

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    PORT=8000 \
    WEB_CONCURRENCY=1 \
    VDD_MODEL_DIR=/app/model \
    VDD_JSON_LOGS=true \
    VDD_ORT_INTRA_OP_THREADS=2

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
# The model bundle is baked in: one immutable image == one code + model version.
# To hot-swap models instead, mount a bundle over /app/model (see README).
COPY --chown=app:app artifacts/model /app/model
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health/ready', timeout=4)" || exit 1

# One worker per container; scale with replicas (keeps Prometheus metrics per-process
# consistent and avoids ORT thread oversubscription).
CMD ["sh", "-c", "exec uvicorn defect_detection.serving.app:create_app --factory --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --no-access-log --timeout-graceful-shutdown 20"]
