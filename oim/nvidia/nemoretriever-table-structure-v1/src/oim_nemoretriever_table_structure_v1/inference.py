from __future__ import annotations

import asyncio
import base64
import binascii
import io
from typing import Iterable, List

from PIL import Image

from .errors import InvalidImageError, InferenceError
from .models import BoundingBox, ImageInput, Prediction, OUTPUT_LABELS


def _decode_data_url(data_url: str) -> bytes:
    """
    Decode a base64 data URL into raw bytes.

    Args:
        data_url: Input data URL containing a base64-encoded image.

    Returns:
        Decoded raw bytes.

    Raises:
        InvalidImageError: When the URL is malformed or not an image data URL.
    """
    if "base64," not in data_url:
        raise InvalidImageError("only data:image/* base64 URLs are supported")
    prefix, payload = data_url.split("base64,", maxsplit=1)
    if not prefix.startswith("data:image"):
        raise InvalidImageError("only data:image/* base64 URLs are supported")
    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise InvalidImageError("invalid base64 payload") from exc


def _ensure_png_bytes(raw: bytes) -> bytes:
    """
    Normalize raw image bytes to PNG.

    Args:
        raw: Raw image content.

    Returns:
        PNG-encoded image bytes.

    Raises:
        InvalidImageError: When the bytes are not a valid image.
    """
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise InvalidImageError(f"invalid image payload: {exc}") from exc
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def load_image_bytes(url: str) -> bytes:
    """
    Load an image from a supported URL into PNG bytes.

    Args:
        url: Data URL pointing to the image payload.

    Returns:
        PNG-encoded image bytes.

    Raises:
        InvalidImageError: When the URL or image payload is invalid.
    """
    if not url.startswith("data:"):
        raise InvalidImageError("only data URLs are supported")
    raw = _decode_data_url(url)
    return _ensure_png_bytes(raw)


async def encode_request_images(items: List[ImageInput]) -> List[str]:
    """
    Decode and normalize request images to base64 PNG strings.

    Args:
        items: Request input descriptors.

    Returns:
        Base64-encoded PNG strings sized to the batch.

    Raises:
        InvalidImageError: When an image cannot be decoded.
        InferenceError: When an unexpected error occurs during encoding.
    """
    tasks = [asyncio.to_thread(load_image_bytes, item.url) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    encoded: List[str] = []
    for result in results:
        if isinstance(result, bytes):
            encoded.append(base64.b64encode(result).decode("utf-8"))
            continue
        if isinstance(result, InvalidImageError):
            raise result
        if isinstance(result, Exception):  # pragma: no cover - surfaced to caller
            raise InferenceError(str(result)) from result
        raise InferenceError("Unexpected image load failure.")
    return encoded


def format_http_predictions(
    raw_predictions: Iterable[dict[str, list[list[float]]]],
) -> List[Prediction]:
    """
    Convert Triton predictions into HTTP response models.

    Args:
        raw_predictions: Parsed predictions keyed by label.

    Returns:
        List of Prediction objects for HTTP responses.

    Raises:
        InferenceError: When prediction payloads are malformed.
    """
    formatted: List[Prediction] = []
    for prediction in raw_predictions:
        bounding_boxes: dict[str, list[BoundingBox]] = {
            label: [] for label in OUTPUT_LABELS
        }
        for label, boxes in prediction.items():
            if label not in bounding_boxes:
                continue
            for box in boxes:
                if len(box) < 5:
                    raise InferenceError("prediction box is incomplete")
                bounding_boxes[label].append(
                    BoundingBox(
                        x_min=float(box[0]),
                        y_min=float(box[1]),
                        x_max=float(box[2]),
                        y_max=float(box[3]),
                        confidence=float(box[4]),
                    )
                )
        formatted.append(Prediction(bounding_boxes=bounding_boxes))
    return formatted
