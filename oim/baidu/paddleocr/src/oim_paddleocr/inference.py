from __future__ import annotations

import asyncio
import base64
import io
import time
from typing import List, Sequence

import numpy as np
import requests
from fastapi import HTTPException
from PIL import Image

from .models import (
    BoundingBox,
    InferResponseItem,
    ModelState,
    Point,
    TextDetection,
    TextPrediction,
)
from .settings import ServiceSettings

try:
    from paddleocr import PaddleOCRVL
except ImportError as exc:  # pragma: no cover - surfaced at startup
    msg = (
        "paddleocr is missing. Install PaddlePaddle GPU + paddleocr[doc-parser] "
        "(see README.md) before running the server."
    )
    raise RuntimeError(msg) from exc


_settings: ServiceSettings | None = None
_paddleocr_instance: PaddleOCRVL | None = None
_model_state = ModelState()
_state_lock = asyncio.Lock()


def init_inference(settings: ServiceSettings) -> None:
    global _settings, _paddleocr_instance
    _settings = settings
    _paddleocr_instance = _create_paddleocr_instance(settings)
    _model_state.loaded = True


def _create_paddleocr_instance(settings: ServiceSettings) -> PaddleOCRVL:
    return PaddleOCRVL(
        use_layout_detection=settings.use_layout_detection,
        use_chart_recognition=settings.use_chart_recognition,
        format_block_content=settings.format_block_content,
        vl_rec_backend=settings.vl_rec_backend,
        vl_rec_server_url=settings.vl_rec_server_url,
    )


async def load_model(force: bool = False) -> None:
    global _paddleocr_instance
    async with _state_lock:
        if _model_state.loaded and _paddleocr_instance is not None and not force:
            return
        _model_state.loaded = False
        instance = await asyncio.to_thread(
            _create_paddleocr_instance, _require_settings()
        )
        _paddleocr_instance = instance
        _model_state.loaded = True


async def unload_model() -> None:
    global _paddleocr_instance
    async with _state_lock:
        _paddleocr_instance = None
        _model_state.loaded = False


def model_ready() -> bool:
    return _model_state.loaded and _paddleocr_instance is not None


async def record_inference_success(duration_seconds: float) -> None:
    async with _state_lock:
        _model_state.inference_count += 1
        _model_state.last_inference_ns = time.time_ns()
        _model_state.success_ns_total += int(duration_seconds * 1e9)


def _require_settings() -> ServiceSettings:
    if _settings is None:
        raise RuntimeError("Service settings not initialized")
    return _settings


def _get_paddleocr() -> PaddleOCRVL:
    if _paddleocr_instance is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return _paddleocr_instance


def decode_data_url(data_url: str) -> bytes:
    try:
        _, payload = data_url.split(",", maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            "Expected data URL in the form data:image/<type>;base64,<payload>."
        ) from exc
    return base64.b64decode(payload)


def to_bgr(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))[:, :, ::-1]


def load_image(url: str, timeout: float) -> np.ndarray:
    if url.startswith("data:"):
        return to_bgr(Image.open(io.BytesIO(decode_data_url(url))))
    if url.startswith("http://") or url.startswith("https://"):
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return to_bgr(Image.open(io.BytesIO(response.content)))
    return to_bgr(Image.open(url))


def normalize_points(
    bbox: Sequence[float],
    width: int,
    height: int,
) -> List[Point]:
    if len(bbox) != 8:
        raise ValueError("bbox must have 8 float values")
    xs = bbox[0::2]
    ys = bbox[1::2]
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    norm = [Point(x=x / float(width), y=y / float(height)) for x, y in zip(xs, ys)]
    return norm


def format_result(result) -> InferResponseItem:
    detections: List[TextDetection] = []
    try:
        width, height = result.page_size
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(
            status_code=500, detail=f"Invalid result object: {exc}"
        ) from exc

    for block in getattr(result, "blocks", []):
        try:
            text = str(getattr(block, "content", "") or "").strip()
            bbox = getattr(block, "bbox", None)
        except Exception:  # pragma: no cover - defensive guard
            continue

        if text == "" or bbox is None:
            continue

        try:
            points = normalize_points(bbox, width, height)
        except ValueError:
            continue

        detections.append(
            TextDetection(
                bounding_box=BoundingBox(points=points),
                text_prediction=TextPrediction(text=text, confidence=1.0),
            )
        )

    return InferResponseItem(text_detections=detections)


def infer(
    images: List[np.ndarray], settings: ServiceSettings
) -> List[InferResponseItem]:
    try:
        results = _get_paddleocr().predict(
            images,
            use_layout_detection=settings.use_layout_detection,
            use_chart_recognition=settings.use_chart_recognition,
            format_block_content=settings.format_block_content,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return [format_result(item) for item in results]


def denormalize_chw(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected image tensor with shape (3, H, W).")
    chw = image.astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    rgb = (chw * std + mean) * 255.0
    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    return np.transpose(rgb, (1, 2, 0))[:, :, ::-1]  # convert to BGR


def collect_triplets(item: InferResponseItem):
    boxes: List[List[List[float]]] = []
    texts: List[str] = []
    confidences: List[float] = []
    for detection in item.text_detections:
        points = detection.bounding_box.points
        boxes.append([[float(pt.x), float(pt.y)] for pt in points])
        texts.append(detection.text_prediction.text)
        confidences.append(float(detection.text_prediction.confidence))
    return boxes, texts, confidences


def state_snapshot() -> ModelState:
    return _model_state
