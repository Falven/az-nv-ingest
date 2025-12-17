from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

from .settings import MergeLevel


class InferInput(BaseModel):
    """
    Supported input type for OCR inference.

    Attributes:
        type: Input type indicator (always ``image_url``).
        url: Data URL, HTTP(S) URL, or filesystem path.
    """

    type: Literal["image_url"] = Field("image_url")
    url: str | HttpUrl


class InferRequest(BaseModel):
    """
    HTTP inference request schema.

    Attributes:
        input: Sequence of image_url inputs to process.
        merge_levels: Optional merge level per input.
    """

    input: list[InferInput]
    merge_levels: list[MergeLevel] | None = None


class Point(BaseModel):
    """
    Normalized point in image space.

    Attributes:
        x: Normalized x coordinate.
        y: Normalized y coordinate.
    """

    x: float
    y: float


class BoundingBox(BaseModel):
    """
    Quadrilateral bounding box.

    Attributes:
        points: Four normalized vertices describing the box.
        type: Box type identifier (quadrilateral).
    """

    points: list[Point]
    type: str = "quadrilateral"


class TextPrediction(BaseModel):
    """
    OCR text prediction with confidence.

    Attributes:
        text: Predicted text.
        confidence: Prediction confidence score.
    """

    text: str
    confidence: float


class TextDetection(BaseModel):
    """
    Single OCR detection pairing a bounding box with text and confidence.

    Attributes:
        bounding_box: Normalized quadrilateral for the detection.
        text_prediction: Text prediction and confidence.
    """

    bounding_box: BoundingBox
    text_prediction: TextPrediction


class InferResponseItem(BaseModel):
    """
    Collection of text detections for one page/image.

    Attributes:
        text_detections: All detections for the source image.
    """

    text_detections: list[TextDetection]


class InferResponse(BaseModel):
    """
    Response envelope for OCR inference requests.

    Attributes:
        data: Parsed detections per requested image.
    """

    data: list[InferResponseItem]


@dataclass
class ParsedPrediction:
    """
    Normalized predictions for a single image.
    """

    detections: list[TextDetection]
    boxes: list[list[list[float]]]
    texts: list[str]
    confidences: list[float]
