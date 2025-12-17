from __future__ import annotations

from typing import Dict, List, Sequence

from pydantic import BaseModel, field_validator

OUTPUT_LABELS: Sequence[str] = ("border", "cell", "row", "column", "header")


class ImageInput(BaseModel):
    """Single image input for inference requests."""

    type: str
    url: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Ensure the input type matches the supported scheme."""

        if value != "image_url":
            raise ValueError("type must be 'image_url'")
        return value


class InferRequest(BaseModel):
    """HTTP inference request payload."""

    input: List[ImageInput]

    @field_validator("input")
    @classmethod
    def ensure_non_empty(cls, value: List[ImageInput]) -> List[ImageInput]:
        """Validate that at least one image is provided."""

        if len(value) == 0:
            raise ValueError("input must include at least one image")
        return value


class BoundingBox(BaseModel):
    """Normalized bounding box prediction."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float


class Prediction(BaseModel):
    """Prediction container returned by inference."""

    bounding_boxes: Dict[str, List[BoundingBox]]
