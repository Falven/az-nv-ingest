from __future__ import annotations

from typing import Iterable, List

from oim_common.errors import InferenceError, InvalidImageError
from oim_common.images import encode_request_images as encode_common_images

from .models import BoundingBox, ImageInput, Prediction, OUTPUT_LABELS

__all__ = [
    "encode_request_images",
    "format_http_predictions",
    "InvalidImageError",
    "InferenceError",
    "BoundingBox",
    "Prediction",
]


async def encode_request_images(
    items: List[ImageInput],
    timeout_seconds: float,
    *,
    allow_remote: bool = False,
    allow_file: bool = False,
) -> List[str]:
    """
    Decode and normalize request images to base64 PNG strings.
    """
    return await encode_common_images(
        items,
        timeout_seconds=timeout_seconds,
        allow_remote=allow_remote,
        allow_file=allow_file,
        require_data_url=True,
    )


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
