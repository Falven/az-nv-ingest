from __future__ import annotations

import base64
import binascii
import io
import json
import logging
from typing import Protocol, Sequence, cast

import numpy as np
import requests
from PIL import Image

from .errors import InvalidImageError, ModelLoadError
from .models import BoundingBox, ParsedPrediction, Point, TextDetection, TextPrediction
from .settings import MergeLevel, ServiceSettings

logger = logging.getLogger(__name__)

RawPrediction = dict[str, float | str]


class OCRModel(Protocol):
    """
    Callable OCR pipeline interface.
    """

    def __call__(
        self, image: np.ndarray, merge_level: MergeLevel, visualize: bool = False
    ) -> list[RawPrediction]:
        """
        Run inference for one image.
        """


class MockNemotronOCR:
    """
    Lightweight mock for NemotronOCR to enable dry-run operation.
    """

    def __call__(
        self, image: np.ndarray, merge_level: MergeLevel, visualize: bool = False
    ) -> list[RawPrediction]:
        """
        Produce a deterministic mock prediction for the provided image.
        """
        height, width = image.shape[:2]
        return [
            {
                "left": 0.1 * width,
                "right": 0.4 * width,
                "upper": 0.2 * height,
                "lower": 0.25 * height,
                "text": f"{merge_level}-mock",
                "confidence": 0.99,
            }
        ]


def create_model(settings: ServiceSettings) -> OCRModel:
    """
    Construct the OCR model or a mock variant based on configuration.
    """
    if settings.enable_mock_inference:
        logger.warning("Mock inference enabled; returning synthetic OCR results.")
        return MockNemotronOCR()
    try:
        from nemotron_ocr.inference.pipeline import NemotronOCR
    except ImportError as exc:  # pragma: no cover - surfaced at startup
        msg = (
            "nemotron_ocr is missing. Install the HF repo (git lfs install && git clone "
            "https://huggingface.co/nvidia/nemotron-ocr-v1 && pip install -v ./nemotron-ocr-v1/nemotron-ocr)."
        )
        raise ModelLoadError(msg) from exc
    return NemotronOCR(model_dir=settings.model_dir)


def decode_data_url(data_url: str) -> bytes:
    """
    Decode a data URL in the form data:image/<type>;base64,<payload>.
    """
    try:
        _, payload = data_url.split(",", maxsplit=1)
    except ValueError as exc:
        raise InvalidImageError(
            "Expected data URL in the form data:image/<type>;base64,<payload>."
        ) from exc
    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise InvalidImageError("Invalid base64 payload in data URL.") from exc


def load_base64_image(encoded: str) -> np.ndarray:
    """
    Load an image from a base64-encoded string.
    """
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise InvalidImageError("Invalid base64 image payload.") from exc

    try:
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise InvalidImageError("Failed to decode base64 image.") from exc


def load_image_reference(value: str, timeout: float) -> np.ndarray:
    """
    Load an RGB image from a data URL, HTTP(S) URL, filesystem path, or raw base64 string.
    """
    if value.startswith("data:"):
        return np.array(Image.open(io.BytesIO(decode_data_url(value))).convert("RGB"))

    if value.startswith("http://") or value.startswith("https://"):
        try:
            response = requests.get(value, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise InvalidImageError("Failed to fetch image from URL.") from exc
        try:
            return np.array(Image.open(io.BytesIO(response.content)).convert("RGB"))
        except Exception as exc:  # pragma: no cover - surfaced to caller
            raise InvalidImageError("Failed to decode fetched image content.") from exc

    try:
        return load_base64_image(value)
    except InvalidImageError:
        try:
            return np.array(Image.open(value).convert("RGB"))
        except (OSError, FileNotFoundError) as exc:
            raise InvalidImageError("Unable to decode image input.") from exc


def _normalize_coord(value: float, dimension: float) -> float:
    """
    Normalize a coordinate into the [0, 1] range relative to a dimension.
    """
    normalized = float(value)
    if dimension > 0 and abs(normalized) > 1.0:
        normalized /= dimension
    if dimension > 0:
        normalized = max(0.0, min(1.0, normalized))
    return normalized


def _parse_prediction(
    pred: RawPrediction, width: int, height: int
) -> tuple[list[list[float]], str, float] | None:
    """
    Convert a raw prediction dict into normalized points, text, and confidence.
    """
    if isinstance(pred.get("left"), str):
        return None

    try:
        left = float(pred.get("left", 0.0))
        right = float(pred.get("right", 0.0))
        upper = float(pred.get("upper", 0.0))
        lower = float(pred.get("lower", 0.0))
    except (TypeError, ValueError):
        return None

    x_min = _normalize_coord(min(left, right), float(width))
    x_max = _normalize_coord(max(left, right), float(width))
    y_min = _normalize_coord(min(lower, upper), float(height))
    y_max = _normalize_coord(max(lower, upper), float(height))

    points = [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ]

    text = str(pred.get("text", ""))
    try:
        confidence = float(pred.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return points, text, confidence


def parse_predictions(
    predictions: Sequence[RawPrediction], width: int, height: int
) -> ParsedPrediction:
    """
    Normalize raw model predictions into structured response objects.
    """
    detections: list[TextDetection] = []
    boxes: list[list[list[float]]] = []
    texts: list[str] = []
    confidences: list[float] = []
    for parsed in (_parse_prediction(pred, width, height) for pred in predictions):
        if parsed is None:
            continue
        points, text, confidence = parsed
        boxes.append(points)
        texts.append(text)
        confidences.append(confidence)
        bbox_points = [Point(x=pt[0], y=pt[1]) for pt in points]
        detections.append(
            TextDetection(
                bounding_box=BoundingBox(points=bbox_points),
                text_prediction=TextPrediction(text=text, confidence=confidence),
            )
        )
    return ParsedPrediction(
        detections=detections, boxes=boxes, texts=texts, confidences=confidences
    )


def run_ocr(
    model: OCRModel, images: Sequence[np.ndarray], merge_levels: Sequence[MergeLevel]
) -> list[ParsedPrediction]:
    """
    Execute OCR inference for a batch of images.
    """
    if len(images) != len(merge_levels):
        raise ValueError("merge_levels length must match number of images.")
    results: list[ParsedPrediction] = []
    for image, merge_level in zip(images, merge_levels):
        preds = model(image, merge_level=merge_level, visualize=False)
        height, width = image.shape[:2]
        results.append(parse_predictions(preds, width, height))
    return results


def parsed_from_triton_output(raw: np.ndarray) -> list[ParsedPrediction]:
    """
    Convert Triton BYTES output back into structured ParsedPrediction objects.
    """
    if raw is None:
        return []
    output = np.array(raw)
    if output.ndim == 1 and output.shape == (3,):
        output = output.reshape(1, 3)
    if output.ndim == 2:
        if output.shape[1] == 3:
            pass
        elif output.shape[0] == 3:
            output = output.transpose((1, 0))
        else:
            raise ValueError(f"Unexpected output shape {output.shape} from Triton.")
    else:
        raise ValueError(f"Unexpected output shape {output.shape} from Triton.")

    parsed: list[ParsedPrediction] = []
    for row in output:
        boxes = json_loads(row[0])
        texts = json_loads(row[1])
        confidences = json_loads(row[2])
        detections = [
            TextDetection(
                bounding_box=BoundingBox(
                    points=[Point(x=pt[0], y=pt[1]) for pt in box]
                ),
                text_prediction=TextPrediction(text=text, confidence=float(confidence)),
            )
            for box, text, confidence in zip(boxes, texts, confidences)
        ]
        parsed.append(
            ParsedPrediction(
                detections=detections,
                boxes=boxes,
                texts=texts,
                confidences=[float(score) for score in confidences],
            )
        )
    return parsed


def json_loads(value: object) -> list:
    """
    Decode a JSON payload that may be bytes or string-like.
    """
    raw = value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
    return cast(list, json.loads(raw))
