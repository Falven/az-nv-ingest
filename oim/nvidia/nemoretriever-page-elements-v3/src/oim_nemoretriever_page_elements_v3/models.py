from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

OUTPUT_LABELS: list[str] = [
    "table",
    "chart",
    "title",
    "infographic",
    "paragraph",
    "header_footer",
]

LABEL_REMAP: dict[str, str] = {"text": "paragraph"}
INPUT_IMAGES_NAME = "INPUT_IMAGES"
THRESHOLDS_NAME = "THRESHOLDS"
OUTPUT_NAME = "OUTPUT"
RawPrediction = Dict[str, List[List[float]]]


class ImageInput(BaseModel):
    """
    Single image input descriptor for inference requests.
    """

    type: str
    url: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """
        Ensure the image input type matches the expected literal.
        """
        if value != "image_url":
            raise ValueError("type must be 'image_url'")
        return value


class InferRequest(BaseModel):
    """
    HTTP inference request payload containing one or more images.
    """

    input: List[ImageInput] = Field(default_factory=list)

    @field_validator("input")
    @classmethod
    def ensure_not_empty(cls, value: List[ImageInput]) -> List[ImageInput]:
        """
        Enforce at least one image in the request payload.
        """
        if len(value) == 0:
            raise ValueError("input must contain at least one image")
        return value


__all__ = [
    "ImageInput",
    "InferRequest",
    "INPUT_IMAGES_NAME",
    "OUTPUT_LABELS",
    "OUTPUT_NAME",
    "RawPrediction",
    "THRESHOLDS_NAME",
]
