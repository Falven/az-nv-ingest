from __future__ import annotations

from typing import Iterable

from oim_common.errors import InferenceError, InvalidImageError
from oim_common.images import encode_request_images as encode_common_images

from .models import InferInput

__all__ = [
    "encode_request_images",
    "format_http_predictions",
    "InvalidImageError",
    "InferenceError",
    "InferInput",
]


async def encode_request_images(
    items: list[InferInput], timeout_seconds: float
) -> list[str]:
    """
    Load request images and return base64 payloads for Triton.
    """
    return await encode_common_images(items, timeout_seconds)

DEFAULT_LABELS: list[str] = [
    "chart_title",
    "x_title",
    "y_title",
    "xlabel",
    "ylabel",
    "other",
    "legend_label",
    "legend_title",
    "mark_label",
    "value_label",
]


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
        labels = {label: prediction.get(label, []) for label in DEFAULT_LABELS}
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
