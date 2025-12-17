from __future__ import annotations

import json
import logging
from typing import List, Sequence, cast

import numpy as np
import triton_python_backend_utils as pb_utils  # type: ignore[reportMissingImports]

from oim_nemoretriever_ocr_v1.errors import InvalidImageError, ModelLoadError
from oim_nemoretriever_ocr_v1.inference import (
    OCRModel,
    create_model,
    load_image_reference,
    run_ocr,
)
from oim_nemoretriever_ocr_v1.models import ParsedPrediction
from oim_nemoretriever_ocr_v1.settings import MergeLevel, ServiceSettings

logger = logging.getLogger("scene_text_wrapper_backend")


def _as_string(value: object) -> str:
    """
    Decode bytes or bytearray values to UTF-8 strings.
    """
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return str(value)


def _error_response(message: str) -> pb_utils.InferenceResponse:
    """
    Build a Triton error response with the provided message.
    """
    error = pb_utils.TritonError(message)
    return pb_utils.InferenceResponse(output_tensors=[], error=error)


def _build_output(predictions: Sequence[ParsedPrediction]) -> pb_utils.Tensor:
    """
    Serialize ParsedPrediction objects into the expected BYTES tensor.
    """
    output = np.empty((len(predictions), 3), dtype=np.object_)
    for idx, parsed in enumerate(predictions):
        output[idx, 0] = json.dumps(parsed.boxes).encode("utf-8")
        output[idx, 1] = json.dumps(parsed.texts).encode("utf-8")
        output[idx, 2] = json.dumps(parsed.confidences).encode("utf-8")
    return pb_utils.Tensor("OUTPUT", output)


def _decode_merge_levels(
    tensor: pb_utils.Tensor | None, batch_size: int, default_level: MergeLevel
) -> list[MergeLevel]:
    """
    Decode merge levels from the incoming tensor or use defaults.
    """
    if tensor is None:
        return [default_level for _ in range(batch_size)]
    raw_values = tensor.as_numpy().reshape(-1)
    levels: list[MergeLevel] = []
    for raw in raw_values:
        level = cast(MergeLevel, _as_string(raw) or default_level)
        if level not in ("word", "sentence", "paragraph"):
            raise InvalidImageError(f"Invalid merge level: {level}")
        levels.append(level)
    if len(levels) != batch_size:
        raise InvalidImageError(
            "MERGE_LEVELS batch size does not match INPUT_IMAGE_URLS."
        )
    return levels


def _decode_images(values: np.ndarray, timeout: float) -> list[np.ndarray]:
    """
    Convert encoded input strings into RGB numpy arrays.
    """
    images: list[np.ndarray] = []
    for raw in values.reshape(-1):
        images.append(load_image_reference(_as_string(raw), timeout))
    return images


class TritonPythonModel:
    """
    Triton Python backend for nemoretriever-ocr-v1.
    """

    def initialize(self, args: dict | None) -> None:
        """
        Initialize the OCR model using ServiceSettings.
        """
        _ = args
        self.settings: ServiceSettings = ServiceSettings()
        try:
            self.model: OCRModel = create_model(self.settings)
        except ModelLoadError as exc:  # pragma: no cover - surfaced at load time
            logger.exception("Failed to load OCR model: %s", exc)
            raise

    def execute(
        self, requests: List[pb_utils.InferenceRequest]
    ) -> List[pb_utils.InferenceResponse]:
        """
        Run inference for one or more incoming Triton requests.
        """
        responses: list[pb_utils.InferenceResponse] = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(
                request, "INPUT_IMAGE_URLS"
            )
            if input_tensor is None:
                responses.append(_error_response("INPUT_IMAGE_URLS tensor is missing."))
                continue

            merge_tensor = pb_utils.get_input_tensor_by_name(request, "MERGE_LEVELS")

            try:
                images = _decode_images(
                    input_tensor.as_numpy(), self.settings.request_timeout_seconds
                )
                merge_levels = _decode_merge_levels(
                    merge_tensor, len(images), self.settings.merge_level
                )
                predictions = run_ocr(self.model, images, merge_levels)
            except (InvalidImageError, ValueError) as exc:
                responses.append(_error_response(str(exc)))
                continue
            except Exception as exc:  # pragma: no cover - surfaced to Triton logs
                logger.exception("Inference failed: %s", exc)
                responses.append(_error_response("Model inference failed."))
                continue

            output_tensor = _build_output(predictions)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses
