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

DEFAULT_LABELS: list[str] = ["border", "cell", "row", "column", "header"]
DEFAULT_SCORE_THRESHOLD = float(os.environ.get("DEFAULT_SCORE_THRESHOLD", "0.1"))


@dataclass
class ThresholdValue:
    """Score threshold for a single image."""

    score: float


def _decode_data_url(value: bytes | str) -> np.ndarray:
    """
    Decode a data URL or base64 payload into an RGB numpy array.

    Args:
        value: String or bytes representing the encoded image.

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
    if not text:
        raise ValueError("Image payload is empty.")
    try:
        data = base64.b64decode(text)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 image payload.") from exc
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # pragma: no cover - surfaced via Triton error
        raise ValueError("Failed to decode image payload.") from exc
    return np.asarray(image)


def _prepare_thresholds(
    thresholds: np.ndarray | None, batch_size: int, default_score: float
) -> list[ThresholdValue]:
    """
    Normalize threshold inputs into a per-image list.

    Args:
        thresholds: Raw FP32 array from the request.
        batch_size: Number of images in the batch.
        default_score: Default score threshold.

    Returns:
        List of ThresholdValue instances sized to the batch.

    Raises:
        ValueError: When the array shape or values are invalid.
    """
    if thresholds is None or thresholds.size == 0:
        return [ThresholdValue(score=default_score) for _ in range(batch_size)]

    parsed = thresholds
    if parsed.ndim == 1:
        parsed = parsed.reshape((-1, 1))
    if parsed.shape[1] < 1:
        raise ValueError("THRESHOLDS must have at least one column.")

    rows = parsed.shape[0]
    if rows not in (1, batch_size):
        raise ValueError(f"THRESHOLDS batch dimension must be 1 or {batch_size}.")

    values: list[ThresholdValue] = []
    for idx in range(batch_size):
        score = float(parsed[0 if rows == 1 else idx][0])
        if not np.isfinite(score):
            raise ValueError("THRESHOLDS values must be finite.")
        if score < 0:
            raise ValueError("THRESHOLDS values must be non-negative.")
        values.append(ThresholdValue(score=score))
    return values


def _mock_prediction(
    labels: list[str], score_threshold: float
) -> dict[str, list[list[float]]]:
    """Produce a deterministic mock prediction for smoke testing."""
    payload: dict[str, list[list[float]]] = {label: [] for label in labels}
    confidence = max(0.95, min(score_threshold, 0.99))
    payload["cell"] = [[0.1, 0.1, 0.4, 0.4, confidence]]
    return payload


class TritonPythonModel:
    """Python backend entrypoint for the table-structure model."""

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
        self._base_threshold = self.score_threshold
        self._postprocess = None
        self.model: torch.nn.Module | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Instantiate the model or configure mock inference."""
        if self.enable_mock:
            self.logger.warning("Mock inference enabled; returning synthetic results.")
            return
        try:
            from nemotron_table_structure_v1.model import define_model
            from nemotron_table_structure_v1.utils import (
                postprocess_preds_table_structure,
            )
        except ImportError as exc:  # pragma: no cover - surfaced via Triton error
            raise RuntimeError(
                "nemotron_table_structure_v1 is missing. Ensure the Hugging Face repo is installed."
            ) from exc

        model = define_model("table_structure_v1", verbose=False)
        if self.device:
            model = model.to(self.device)
        model.eval()
        self.model = model
        self.labels = list(getattr(model, "labels", DEFAULT_LABELS))
        self._base_threshold = float(getattr(model, "threshold", self.score_threshold))
        self._postprocess = postprocess_preds_table_structure
        self.logger.info(
            "Model loaded (version=%s, threshold=%.4f)",
            getattr(model, "version", "unknown"),
            self._base_threshold,
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
        images: list[np.ndarray] = [
            _decode_data_url(value) for value in flat_values.tolist()
        ]

        thresholds_input = pb_utils.get_input_tensor_by_name(request, "THRESHOLDS")
        thresholds_array = (
            thresholds_input.as_numpy() if thresholds_input is not None else None
        )

        if self.max_batch_size and len(images) > self.max_batch_size:
            raise ValueError(
                f"Batch size {len(images)} exceeds limit {self.max_batch_size}."
            )
        if len(images) == 0:
            raise ValueError("No images provided.")

        threshold_values = _prepare_thresholds(
            thresholds_array,
            len(images),
            self._base_threshold,
        )
        predictions = self._run_inference(images, threshold_values)
        payload = np.array(
            [json.dumps(pred).encode("utf-8") for pred in predictions], dtype=np.object_
        )
        output_tensor = pb_utils.Tensor("OUTPUT", payload)
        return pb_utils.InferenceResponse(output_tensors=[output_tensor])

    def _run_inference(
        self, images: list[np.ndarray], thresholds: list[ThresholdValue]
    ) -> list[dict[str, list[list[float]]]]:
        """Execute model inference for a batch of images."""
        if self.enable_mock:
            return [_mock_prediction(self.labels, self.score_threshold) for _ in images]
        if self.model is None or self._postprocess is None:
            raise RuntimeError("Model is not loaded.")

        tensors = [self.model.preprocess(image) for image in images]
        results: list[dict[str, list[list[float]]]] = []
        unique_thresholds = {value.score for value in thresholds}

        if len(unique_thresholds) == 1:
            score = thresholds[0].score
            batch = torch.stack(
                [
                    tensor.to(self.device) if self.device else tensor
                    for tensor in tensors
                ]
            )
            original_sizes = np.array([image.shape[:2] for image in images])
            with torch.inference_mode():
                preds = self.model(batch, original_sizes)
            results.extend(
                [self._postprocess_prediction(pred, score) for pred in preds]
            )
            return results

        for tensor, image, threshold in zip(tensors, images, thresholds):
            prepared = tensor.to(self.device) if self.device else tensor
            with torch.inference_mode():
                pred = self.model(
                    torch.stack([prepared]),
                    np.array([image.shape[:2]]),
                )[0]
            results.append(self._postprocess_prediction(pred, threshold.score))
        return results

    def _postprocess_prediction(
        self, prediction: dict[str, np.ndarray], threshold: float
    ) -> dict[str, list[list[float]]]:
        """Convert raw model predictions into the normalized response schema."""
        if self._postprocess is None:
            raise RuntimeError(
                "Post-processing is unavailable because the model is not loaded."
            )
        boxes, labels, scores = self._postprocess(
            prediction, threshold=threshold, class_labels=self.labels, reorder=True
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
