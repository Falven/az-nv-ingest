from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import triton_python_backend_utils as pb_utils  # type: ignore[reportMissingImports]
from PIL import Image

DEFAULT_LABELS: list[str] = [
    "table",
    "chart",
    "title",
    "infographic",
    "paragraph",
    "header_footer",
]
LABEL_REMAP: dict[str, str] = {"text": "paragraph"}
DEFAULT_CONF_THRESHOLD = float(os.environ.get("DEFAULT_CONF_THRESHOLD", "0.01"))
DEFAULT_IOU_THRESHOLD = float(os.environ.get("DEFAULT_IOU_THRESHOLD", "0.5"))
DEFAULT_SCORE_THRESHOLD = float(os.environ.get("DEFAULT_SCORE_THRESHOLD", "0.1"))


@dataclass
class ThresholdPair:
    """Pair of per-request confidence and IoU thresholds."""

    conf: float
    iou: float


def _decode_image(value: bytes | str) -> np.ndarray:
    """
    Decode a base64 payload (optionally a data URL) into an RGB numpy array.

    Args:
        value: Base64-encoded image bytes or string.

    Returns:
        RGB numpy array.

    Raises:
        ValueError: If the payload cannot be decoded into an image.
    """
    text = (
        value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
    )
    if "base64," in text:
        text = text.split("base64,", 1)[1]
    try:
        data = base64.b64decode(text)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image payload") from exc
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # pragma: no cover - surfaced through Triton
        raise ValueError("Failed to decode image payload") from exc
    return np.asarray(image)


def _prepare_threshold_pairs(
    threshold_array: np.ndarray | None,
    batch_size: int,
    base_conf: float,
    base_iou: float,
) -> list[ThresholdPair]:
    """
    Normalize threshold inputs into a per-image list of pairs.

    Args:
        threshold_array: Raw FP32 array from the request.
        batch_size: Number of images in the batch.
        base_conf: Default confidence threshold.
        base_iou: Default IoU threshold.

    Returns:
        List of ThresholdPair instances matching the batch size.

    Raises:
        ValueError: When the array shape or values are invalid.
    """
    if threshold_array is None or threshold_array.size == 0:
        return [ThresholdPair(base_conf, base_iou) for _ in range(batch_size)]

    thresholds = np.asarray(threshold_array, dtype=np.float32)
    if thresholds.ndim == 1:
        if thresholds.shape[0] not in (1, batch_size):
            raise ValueError("THRESHOLDS length must be 1 or match batch size")
        pairs: list[ThresholdPair] = []
        for idx in range(batch_size):
            conf = thresholds[0 if thresholds.shape[0] == 1 else idx]
            if not np.isfinite(conf) or conf < 0:
                raise ValueError("THRESHOLDS values must be finite and non-negative")
            pairs.append(ThresholdPair(conf=float(conf), iou=base_iou))
        return pairs

    if thresholds.ndim >= 2:
        if thresholds.shape[1] < 2:
            raise ValueError("THRESHOLDS must include [conf, iou] values")
        rows = thresholds.shape[0]
        if rows not in (1, batch_size):
            raise ValueError(f"THRESHOLDS batch dimension must be 1 or {batch_size}")
        pairs = []
        for idx in range(batch_size):
            conf, iou = thresholds[0 if rows == 1 else idx][:2]
            if not np.isfinite(conf) or not np.isfinite(iou) or conf < 0 or iou < 0:
                raise ValueError("THRESHOLDS values must be finite and non-negative")
            pairs.append(ThresholdPair(conf=float(conf), iou=float(iou)))
        return pairs

    raise ValueError("THRESHOLDS must be a 1D or 2D FP32 tensor")


def _mock_prediction(
    labels: Sequence[str], score_threshold: float
) -> dict[str, list[list[float]]]:
    """
    Produce a deterministic mock prediction payload.

    Args:
        labels: Known output labels.
        score_threshold: Score to attach to the synthetic box.

    Returns:
        Prediction dictionary keyed by label.
    """
    payload: dict[str, list[list[float]]] = {label: [] for label in labels}
    payload["paragraph"] = [[0.1, 0.1, 0.6, 0.6, max(score_threshold, 0.9)]]
    return payload


class TritonPythonModel:
    """Python backend entrypoint for the page elements model."""

    def initialize(self, args: dict[str, str]) -> None:
        """Load the detection model or enable mock mode."""
        torch.set_grad_enabled(False)
        self.logger = logging.getLogger("pipeline")
        self.model_name = args["model_config"]["name"]
        config = json.loads(args["model_config"])
        self.max_batch_size = int(config.get("max_batch_size", 0) or 0)
        self.enable_mock = os.environ.get("ENABLE_MOCK_INFERENCE", "0") == "1"
        self.score_threshold = float(
            os.environ.get("DEFAULT_SCORE_THRESHOLD", DEFAULT_SCORE_THRESHOLD)
        )
        self.device = os.environ.get("DEVICE")
        self._base_conf_threshold = DEFAULT_CONF_THRESHOLD
        self._base_iou_threshold = DEFAULT_IOU_THRESHOLD
        self.labels: list[str] = list(DEFAULT_LABELS)
        self.model: torch.nn.Module | None = None
        self._postprocess = None
        self._load_model()

    def _load_model(self) -> None:
        """Instantiate the model unless mock inference is enabled."""
        if self.enable_mock:
            self.logger.warning("Mock inference enabled; returning synthetic results.")
            return
        try:
            from nemotron_page_elements_v3.model import define_model
            from nemotron_page_elements_v3.utils import postprocess_preds_page_element
        except ImportError as exc:  # pragma: no cover - surfaced via Triton error
            raise RuntimeError(
                "nemotron_page_elements_v3 is missing. Ensure the Hugging Face repo is installed."
            ) from exc

        model = define_model("page_element_v3", verbose=False)
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
        self._postprocess = postprocess_preds_page_element
        self.logger.info(
            "Model loaded (version=%s, conf_thresh=%.4f, iou_thresh=%.4f)",
            getattr(model, "version", "unknown"),
            self._base_conf_threshold,
            self._base_iou_threshold,
        )

    def execute(
        self, requests: Iterable[pb_utils.InferenceRequest]
    ) -> list[pb_utils.InferenceResponse]:
        """Handle one or more Triton inference requests."""
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
        """Run inference for a single request."""
        images_input = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGES")
        if images_input is None:
            raise ValueError("INPUT_IMAGES is required.")
        image_values = images_input.as_numpy()
        if image_values is None:
            raise ValueError("INPUT_IMAGES payload is empty.")
        flat_values = image_values.reshape(-1)
        images = [_decode_image(value) for value in flat_values.tolist()]

        threshold_tensor = pb_utils.get_input_tensor_by_name(request, "THRESHOLDS")
        thresholds_array = (
            threshold_tensor.as_numpy() if threshold_tensor is not None else None
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
        self, images: Sequence[np.ndarray], threshold_pairs: Sequence[ThresholdPair]
    ) -> list[dict[str, list[list[float]]]]:
        """Execute model inference for the provided images and thresholds."""
        if self.enable_mock:
            return [
                _mock_prediction(DEFAULT_LABELS, self.score_threshold) for _ in images
            ]
        if self.model is None or self._postprocess is None:
            raise RuntimeError("Model is not loaded.")

        tensors = [self.model.preprocess(image) for image in images]
        original_sizes = np.array([image.shape[:2] for image in images])
        unique_pairs = {(pair.conf, pair.iou) for pair in threshold_pairs}
        predictions: list[dict[str, list[list[float]]]] = []

        try:
            if len(unique_pairs) == 1:
                pair = threshold_pairs[0]
                self._apply_thresholds(pair)
                batch = torch.stack(tensors)
                with torch.inference_mode():
                    preds = self.model(batch, original_sizes)
                predictions.extend(self._convert_predictions(preds))
            else:
                for idx, pair in enumerate(threshold_pairs):
                    self._apply_thresholds(pair)
                    with torch.inference_mode():
                        preds = self.model(
                            tensors[idx].unsqueeze(0),
                            np.array([original_sizes[idx]]),
                        )
                    predictions.extend(self._convert_predictions(preds))
        finally:
            self._restore_thresholds()

        return predictions

    def _apply_thresholds(self, pair: ThresholdPair) -> None:
        """Apply per-request thresholds to the underlying model."""
        if self.model is None:
            return
        if hasattr(self.model, "conf_thresh"):
            self.model.conf_thresh = pair.conf
        if hasattr(self.model, "iou_thresh"):
            self.model.iou_thresh = pair.iou

    def _restore_thresholds(self) -> None:
        """Restore base thresholds after inference."""
        if self.model is None:
            return
        if hasattr(self.model, "conf_thresh"):
            self.model.conf_thresh = self._base_conf_threshold
        if hasattr(self.model, "iou_thresh"):
            self.model.iou_thresh = self._base_iou_threshold

    def _convert_predictions(
        self, preds: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, list[list[float]]]]:
        """Convert model outputs into normalized bounding box structures."""
        if self.model is None or self._postprocess is None:
            raise RuntimeError("Model is not loaded.")

        results: list[dict[str, list[list[float]]]] = []
        for pred in preds:
            boxes, labels, scores = self._postprocess(
                pred, getattr(self.model, "thresholds_per_class", None), self.labels
            )
            bounding_boxes: dict[str, list[list[float]]] = {
                label: [] for label in DEFAULT_LABELS
            }
            for box, label_idx, score in zip(boxes, labels, scores):
                label_name = self.labels[int(label_idx)]
                output_label = LABEL_REMAP.get(label_name, label_name)
                if output_label not in bounding_boxes:
                    continue
                xmin, ymin, xmax, ymax = (float(coord) for coord in box.tolist())
                bounding_boxes[output_label].append(
                    [xmin, ymin, xmax, ymax, float(score)]
                )
            results.append(bounding_boxes)
        return results
