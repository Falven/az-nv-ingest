from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import triton_python_backend_utils as pb_utils
import torch
from PIL import Image

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
DEFAULT_CONF_THRESHOLD = float(os.environ.get("DEFAULT_CONF_THRESHOLD", "0.01"))
DEFAULT_IOU_THRESHOLD = float(os.environ.get("DEFAULT_IOU_THRESHOLD", "0.25"))
DEFAULT_SCORE_THRESHOLD = float(os.environ.get("DEFAULT_SCORE_THRESHOLD", "0.1"))


@dataclass
class ThresholdPair:
    """Pair of per-request confidence and IoU thresholds."""

    conf: float
    iou: float


def _decode_base64_image(value: bytes | str) -> np.ndarray:
    """
    Decode a base64 payload into an RGB numpy array.

    Args:
        value: Base64 payload, optionally prefixed with ``base64,`` or a data URL.

    Returns:
        RGB numpy array.

    Raises:
        ValueError: When the payload cannot be decoded into an image.
    """
    text = (
        value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
    )
    if "base64," in text:
        text = text.split("base64,", 1)[1]
    try:
        data = base64.b64decode(text)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 image payload") from exc
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # pragma: no cover - surfaced via Triton error
        raise ValueError("Failed to decode image payload") from exc
    return np.asarray(image)


def _prepare_threshold_pairs(
    threshold_array: np.ndarray | None,
    batch_size: int,
    default_conf: float,
    default_iou: float,
) -> list[ThresholdPair]:
    """
    Normalize threshold inputs into a per-image list of pairs.

    Args:
        threshold_array: Raw FP32 array from the request.
        batch_size: Number of images in the batch.
        default_conf: Default confidence threshold.
        default_iou: Default IoU threshold.

    Returns:
        List of ThresholdPair instances sized to the batch.

    Raises:
        ValueError: When the array shape or values are invalid.
    """
    if threshold_array is None or threshold_array.size == 0:
        return [
            ThresholdPair(conf=default_conf, iou=default_iou) for _ in range(batch_size)
        ]

    thresholds = threshold_array
    if thresholds.ndim == 1:
        if thresholds.size % 2 != 0:
            raise ValueError(
                "THRESHOLDS must provide pairs of [conf_threshold, iou_threshold]"
            )
        thresholds = thresholds.reshape((-1, 2))
    if thresholds.shape[1] < 2:
        raise ValueError("THRESHOLDS must have a trailing dimension of 2")

    rows = thresholds.shape[0]
    if rows not in (1, batch_size):
        raise ValueError(f"THRESHOLDS batch dimension must be 1 or {batch_size}")

    pairs: list[ThresholdPair] = []
    for idx in range(batch_size):
        conf, iou = thresholds[0 if rows == 1 else idx][:2]
        if not np.isfinite(conf) or not np.isfinite(iou):
            raise ValueError("THRESHOLDS values must be finite")
        if conf < 0 or iou < 0:
            raise ValueError("THRESHOLDS values must be non-negative")
        pairs.append(ThresholdPair(conf=float(conf), iou=float(iou)))
    return pairs


def _mock_prediction(
    labels: list[str], score_threshold: float
) -> dict[str, list[list[float]]]:
    """Produce a deterministic mock prediction for smoke testing."""
    payload: dict[str, list[list[float]]] = {label: [] for label in labels}
    confidence = max(0.95, min(score_threshold, 0.99))
    payload[labels[0]] = [[0.1, 0.1, 0.9, 0.2, confidence]]
    return payload


class TritonPythonModel:
    """Python backend entrypoint for the graphic elements model."""

    def initialize(self, args: dict[str, str]) -> None:
        """Load the underlying model and preprocess configuration."""
        torch.set_grad_enabled(False)
        self.logger = logging.getLogger("yolox_ensemble")
        self.model_name = args["model_config"]["name"]
        config = json.loads(args["model_config"])
        self.max_batch_size = int(config.get("max_batch_size", 0) or 0)
        self.device = os.environ.get("DEVICE")
        self.enable_mock = os.environ.get("ENABLE_MOCK_INFERENCE", "0") == "1"
        self.score_threshold = float(
            os.environ.get("DEFAULT_SCORE_THRESHOLD", DEFAULT_SCORE_THRESHOLD)
        )
        self.labels = list(DEFAULT_LABELS)
        self._base_conf_threshold = DEFAULT_CONF_THRESHOLD
        self._base_iou_threshold = DEFAULT_IOU_THRESHOLD
        self._postprocess = None
        self.model: torch.nn.Module | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Instantiate the model or configure mock inference."""
        if self.enable_mock:
            self.logger.warning("Mock inference enabled; returning synthetic results.")
            return
        try:
            from nemotron_graphic_elements_v1.model import define_model
            from nemotron_graphic_elements_v1.utils import (
                postprocess_preds_graphic_element,
            )
        except ImportError as exc:  # pragma: no cover - surfaced via Triton error
            raise RuntimeError(
                "nemotron_graphic_elements_v1 is missing. Ensure the Hugging Face repo is installed."
            ) from exc

        model = define_model("graphic_element_v1")
        if self.device:
            model = model.to(self.device)
        model.eval()
        self.model = model
        self.labels = list(getattr(model, "labels", DEFAULT_LABELS))
        self._base_conf_threshold = float(
            getattr(model, "conf_thresh", DEFAULT_CONF_THRESHOLD)
        )
        self._base_iou_threshold = float(
            getattr(model, "iou_thresh", DEFAULT_IOU_THRESHOLD)
        )
        self._postprocess = postprocess_preds_graphic_element
        self.logger.info(
            "Model loaded (version=%s, conf_thresh=%.4f, iou_thresh=%.4f)",
            getattr(model, "version", "unknown"),
            self._base_conf_threshold,
            self._base_iou_threshold,
        )

    def execute(
        self, requests: Iterable[pb_utils.InferenceRequest]
    ) -> list[pb_utils.InferenceResponse]:
        """Handle a batch of Triton inference requests."""
        responses: list[pb_utils.InferenceResponse] = []
        for request in requests:
            try:
                responses.append(self._handle_request(request))
            except Exception as exc:  # pragma: no cover - surfaced to Triton
                self.logger.exception("Inference failed.")
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))
                )
        return responses

    def _handle_request(
        self, request: pb_utils.InferenceRequest
    ) -> pb_utils.InferenceResponse:
        """Run inference for a single Triton request."""
        images_input = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGES")
        if images_input is None:
            raise ValueError("INPUT_IMAGES is required.")
        image_values = images_input.as_numpy()
        if image_values is None:
            raise ValueError("INPUT_IMAGES payload is empty.")
        flat_values = image_values.reshape(-1)
        images: list[np.ndarray] = []
        for value in flat_values.tolist():
            images.append(_decode_base64_image(value))

        thresholds_input = pb_utils.get_input_tensor_by_name(request, "THRESHOLDS")
        thresholds_array = (
            thresholds_input.as_numpy() if thresholds_input is not None else None
        )

        if self.max_batch_size and len(images) > self.max_batch_size:
            raise ValueError(
                f"Batch size {len(images)} exceeds limit {self.max_batch_size}"
            )
        if len(images) == 0:
            raise ValueError("No images provided.")

        threshold_pairs = _prepare_threshold_pairs(
            thresholds_array,
            len(images),
            self._base_conf_threshold,
            self._base_iou_threshold,
        )
        predictions = self._run_inference(images, threshold_pairs)
        payload = np.array(
            [json.dumps(pred).encode("utf-8") for pred in predictions], dtype=np.object_
        )
        output_tensor = pb_utils.Tensor("OUTPUT", payload)
        return pb_utils.InferenceResponse(output_tensors=[output_tensor])

    def _run_inference(
        self, images: list[np.ndarray], threshold_pairs: list[ThresholdPair]
    ) -> list[dict[str, list[list[float]]]]:
        """Execute model inference for a batch of images."""
        if self.enable_mock:
            return [_mock_prediction(self.labels, self.score_threshold) for _ in images]
        if self.model is None or self._postprocess is None:
            raise RuntimeError("Model is not loaded.")

        tensors = [self.model.preprocess(image) for image in images]
        results: list[dict[str, list[list[float]]]] = []
        unique_pairs = {(pair.conf, pair.iou) for pair in threshold_pairs}

        try:
            if len(unique_pairs) == 1:
                pair = threshold_pairs[0]
                self.model.conf_thresh = pair.conf
                self.model.iou_thresh = pair.iou
                batch = torch.stack(
                    [
                        tensor.to(self.device) if self.device else tensor
                        for tensor in tensors
                    ]
                )
                sizes: list[tuple[int, int]] = [
                    (image.shape[0], image.shape[1]) for image in images
                ]
                with torch.inference_mode():
                    preds = self.model(batch, sizes)
                results.extend([self._postprocess_prediction(pred) for pred in preds])
                return results

            for tensor, image, pair in zip(tensors, images, threshold_pairs):
                self.model.conf_thresh = pair.conf
                self.model.iou_thresh = pair.iou
                prepared = tensor.to(self.device) if self.device else tensor
                with torch.inference_mode():
                    pred = self.model(
                        torch.stack([prepared]), [(image.shape[0], image.shape[1])]
                    )[0]
                results.append(self._postprocess_prediction(pred))
        finally:
            try:
                self.model.conf_thresh = self._base_conf_threshold
                self.model.iou_thresh = self._base_iou_threshold
            except Exception:  # pragma: no cover - best-effort reset
                self.logger.debug(
                    "Failed to reset model thresholds after inference", exc_info=True
                )
        return results

    def _postprocess_prediction(
        self, prediction: dict[str, np.ndarray]
    ) -> dict[str, list[list[float]]]:
        """Convert raw model predictions into the normalized response schema."""
        if self._postprocess is None:
            raise RuntimeError(
                "Post-processing is unavailable because the model is not loaded."
            )
        boxes, labels, scores = self._postprocess(
            prediction, threshold=self.score_threshold, class_labels=self.labels
        )
        bbox_dict: dict[str, list[list[float]]] = {label: [] for label in self.labels}
        for bbox, label_idx, score in zip(boxes, labels, scores):
            label_name = self.labels[int(label_idx)]
            bbox_dict[label_name].append(
                [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                    float(score),
                ]
            )
        return bbox_dict

    def finalize(self) -> None:
        """Release resources when the model is unloaded."""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                self.logger.debug(
                    "Failed to clear CUDA cache during finalize", exc_info=True
                )
