"""Response contracts (exposed in the OpenAPI docs at /docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictionOut(BaseModel):
    predicted_class: Literal["normal", "defective"]
    confidence: float = Field(..., ge=0, le=1, description="Model probability of the predicted class")
    defect_probability: float = Field(..., ge=0, le=1, description="P(defective)")
    threshold: float = Field(..., description="Decision threshold applied to defect_probability")
    review_recommended: bool = Field(
        ..., description="Score is close to the threshold; route to manual inspection"
    )


class TimingOut(BaseModel):
    decode_ms: float
    preprocess_ms: float
    inference_ms: float
    total_ms: float


class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    filename: str | None
    prediction: PredictionOut
    timing: TimingOut


class BatchItem(BaseModel):
    filename: str | None
    prediction: PredictionOut | None = None
    error: ErrorBody | None = None


class BatchPredictResponse(BaseModel):
    request_id: str
    model_version: str
    results: list[BatchItem]
    timing: TimingOut


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: str | None = None


class ModelInfo(BaseModel):
    model_version: str
    backbone: str | None
    class_names: list[str]
    threshold: float
    threshold_strategy: str | None
    review_margin: float
    input_size: int
    created_utc: str | None
    val_average_precision: float | None


BatchItem.model_rebuild()
