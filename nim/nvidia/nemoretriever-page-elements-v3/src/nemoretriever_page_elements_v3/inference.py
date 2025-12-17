from __future__ import annotations

import asyncio
import base64
import binascii
import io
from typing import Iterable

import requests
from PIL import Image

from .errors import InvalidImageError, InferenceError
from .models import ImageInput, OUTPUT_LABELS


def _encode_png(image: Image.Image) -> bytes:
    """
    Encode a PIL image into PNG bytes.

    Args:
        image: PIL image to encode.

    Returns:
        PNG-encoded image bytes.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _decode_data_url(data_url: str) -> bytes:
    """
    Decode the payload portion of a data URL.

    Args:
        data_url: Data URL containing a base64 payload.

    Returns:
        Raw decoded bytes.

    Raises:
        InvalidImageError: When the data URL or payload is malformed.
    """
    try:
        _, payload = data_url.split(",", maxsplit=1)
    except ValueError as exc:
        raise InvalidImageError(
            "Expected data URL in the form data:image/<type>;base64,<payload>."
        ) from exc
    try:
        return base64.b64decode(payload)
    except (ValueError, binascii.Error) as exc:
        raise InvalidImageError("Invalid base64 payload in data URL.") from exc


def _ensure_png_bytes(raw: bytes) -> bytes:
    """
    Convert raw image bytes into normalized PNG bytes after validation.

    Args:
        raw: Raw image content.

    Returns:
        PNG-encoded bytes.

    Raises:
        InvalidImageError: When the bytes do not represent a valid image.
    """
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # pragma: no cover - validation path
        raise InvalidImageError(f"Failed to decode image payload: {exc}") from exc
    return _encode_png(image)


def load_image_bytes(url: str, timeout_seconds: float) -> bytes:
    """
    Load an image from a data URL, HTTP(S) endpoint, or local path.

    Args:
        url: Source URL or path.
        timeout_seconds: HTTP timeout for remote fetches.

    Returns:
        PNG-encoded image bytes.

    Raises:
        InvalidImageError: When the image cannot be loaded.
        requests.RequestException: When remote fetch fails.
    """
    try:
        if url.startswith("data:"):
            return _ensure_png_bytes(_decode_data_url(url))
        if url.startswith("http://") or url.startswith("https://"):
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            return _ensure_png_bytes(response.content)
        with open(url, "rb") as file:
            return _ensure_png_bytes(file.read())
    except (OSError, ValueError, InvalidImageError):
        raise
    except requests.RequestException:
        raise
    except Exception as exc:
        raise InvalidImageError(f"Failed to load image from {url}: {exc}") from exc


async def encode_request_images(
    items: list[ImageInput], timeout_seconds: float
) -> list[str]:
    """
    Load request images and return base64 payloads for Triton.

    Args:
        items: Request input descriptors.
        timeout_seconds: Timeout for remote fetches.

    Returns:
        Base64-encoded PNG strings sized to the batch.

    Raises:
        InvalidImageError: When an image cannot be loaded or decoded.
        InferenceError: When loading fails unexpectedly.
    """
    tasks = [
        asyncio.to_thread(load_image_bytes, item.url, timeout_seconds) for item in items
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    encoded: list[str] = []
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
) -> list[dict[str, dict[str, list[dict[str, float]]]]]:
    """
    Format parsed predictions into the HTTP response schema.

    Args:
        raw_predictions: Parsed predictions keyed by label.

    Returns:
        List of response items compatible with the HTTP API.
    """
    formatted: list[dict[str, dict[str, list[dict[str, float]]]]] = []
    for prediction in raw_predictions:
        labels = {label: prediction.get(label, []) for label in OUTPUT_LABELS}
        formatted.append(
            {
                "bounding_boxes": {
                    label: [
                        {
                            "x_min": bbox[0],
                            "y_min": bbox[1],
                            "x_max": bbox[2],
                            "y_max": bbox[3],
                            "confidence": bbox[4],
                        }
                        for bbox in boxes
                    ]
                    for label, boxes in labels.items()
                }
            }
        )
    return formatted
