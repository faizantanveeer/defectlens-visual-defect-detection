"""Service configuration from environment variables (12-factor). Prefix: ``VDD_``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VDD_", env_file=".env", extra="ignore")

    model_dir: Path = Path("artifacts/model")
    verify_model_checksum: bool = True
    # Optional operating-point override without re-exporting the model (e.g. a stricter
    # line). Logged at startup so the active threshold is always auditable.
    threshold_override: float | None = Field(None, gt=0, lt=1)

    max_upload_bytes: int = Field(10 * 1024 * 1024, gt=0)
    max_batch_size: int = Field(16, ge=1, le=256)
    max_image_pixels: int = Field(40_000_000, gt=0)
    min_image_side: int = Field(32, ge=1)
    allowed_content_types: tuple[str, ...] = (
        "image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp",
        "application/octet-stream",  # some line-side clients do not set a type; bytes are still validated
    )

    # Backpressure: at most N inferences in flight per process; extra requests wait up to
    # queue_timeout_s and then get 503 + Retry-After instead of piling up unbounded.
    max_concurrent_inferences: int = Field(4, ge=1)
    queue_timeout_s: float = Field(2.0, gt=0)
    ort_intra_op_threads: int = Field(0, ge=0)  # 0 = onnxruntime default

    api_key: str | None = None  # if set, clients must send X-API-Key
    # Browser origins allowed to call the API (the bundled test UI runs on :5173).
    # Env: VDD_CORS_ORIGINS='["https://inspect.example.com"]'; empty list disables CORS.
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    log_level: str = "INFO"
    json_logs: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
