from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

from pydantic import BaseModel, HttpUrl


class InferInput(BaseModel):
    type: Literal["image_url"]
    url: str | HttpUrl


class InferRequest(BaseModel):
    input: List[InferInput]
    merge_levels: List[str] | None = None


class Point(BaseModel):
    x: float
    y: float


class BoundingBox(BaseModel):
    points: List[Point]
    type: Literal["quadrilateral"] = "quadrilateral"


class TextPrediction(BaseModel):
    text: str
    confidence: float


class TextDetection(BaseModel):
    bounding_box: BoundingBox
    text_prediction: TextPrediction


class InferResponseItem(BaseModel):
    text_detections: List[TextDetection]


@dataclass
class ModelState:
    loaded: bool = False
    inference_count: int = 0
    last_inference_ns: int = 0
    success_ns_total: int = 0
