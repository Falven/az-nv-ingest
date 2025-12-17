from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, HttpUrl, field_validator

ParsedPrediction = Dict[str, List[List[float]]]


class InferInput(BaseModel):
    """Single inference input item."""

    type: Literal["image_url"]
    url: str | HttpUrl


class InferRequest(BaseModel):
    """HTTP inference request payload."""

    input: List[InferInput]

    @field_validator("input")
    @classmethod
    def ensure_not_empty(cls, value: List[InferInput]) -> List[InferInput]:
        """Ensure the request includes at least one input."""
        if not value:
            raise ValueError("input must contain at least one image")
        return value


class BoundingBox(BaseModel):
    """Normalized bounding box with confidence score."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float


class InferResponseItem(BaseModel):
    """Per-image inference result."""

    bounding_boxes: Dict[str, List[BoundingBox]]


class InferResponse(BaseModel):
    """Full HTTP inference response."""

    data: List[InferResponseItem]
