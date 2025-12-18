from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class _Block:
    content: str
    bbox: Sequence[float]


@dataclass
class _Result:
    page_size: tuple[int, int]
    blocks: List[_Block]


class PaddleOCRVL:
    """
    Lightweight stub of PaddleOCRVL used for tests.
    """

    def __init__(self, *args, **kwargs):
        _ = args, kwargs

    def predict(self, images: List[np.ndarray], **kwargs) -> List[_Result]:
        _ = kwargs
        results: List[_Result] = []
        for idx, image in enumerate(images):
            height, width = (
                (image.shape[0], image.shape[1]) if image is not None else (8, 8)
            )
            bbox = [
                0.1 * width,
                0.1 * height,
                0.9 * width,
                0.1 * height,
                0.9 * width,
                0.9 * height,
                0.1 * width,
                0.9 * height,
            ]
            results.append(
                _Result(
                    page_size=(width, height),
                    blocks=[_Block(content=f"text-{idx}", bbox=bbox)],
                )
            )
        return results
