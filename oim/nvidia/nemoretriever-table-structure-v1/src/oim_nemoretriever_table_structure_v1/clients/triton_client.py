from __future__ import annotations

import asyncio
import json
import logging
from typing import Iterable, List

import numpy as np
from google.protobuf.json_format import MessageToDict
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tritonclient.grpc import (
    InferInput,
    InferRequestedOutput,
    InferenceServerClient,
    InferenceServerException,
)

from ..errors import TritonInferenceError
from ..settings import ServiceSettings

logger = logging.getLogger(__name__)

INPUT_IMAGES_NAME = "INPUT_IMAGES"
THRESHOLDS_NAME = "THRESHOLDS"
OUTPUT_NAME = "OUTPUT"


class TritonClient:
    """Thin wrapper around the Triton gRPC client with retry semantics."""

    def __init__(self, settings: ServiceSettings) -> None:
        """Create a client bound to the configured Triton endpoint."""
        self._settings = settings
        self._client = InferenceServerClient(
            url=settings.triton_grpc_endpoint,
            verbose=settings.log_verbose > 0,
            network_timeout=settings.request_timeout_seconds,
        )

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        try:
            self._client.close()
        except Exception:
            logger.debug("Failed to close Triton client cleanly", exc_info=True)

    def is_server_ready(self) -> bool:
        """Return Triton server readiness."""
        try:
            return bool(self._client.is_server_ready())
        except InferenceServerException:
            logger.debug("Triton server readiness probe failed", exc_info=True)
            return False

    def is_model_ready(self) -> bool:
        """Return model readiness."""
        try:
            return bool(self._client.is_model_ready(self._settings.triton_model_name))
        except InferenceServerException:
            logger.debug("Triton model readiness probe failed", exc_info=True)
            return False

    def is_live(self) -> bool:
        """Alias for Triton server liveness."""
        return self.is_server_ready()

    def is_ready(self) -> bool:
        """Alias for Triton model readiness."""
        return self.is_model_ready()

    def repository_index(self) -> list[dict[str, object]]:
        """Return the Triton model repository index."""
        try:
            response = self._client.get_model_repository_index()
        except InferenceServerException:
            logger.debug("Triton repository index probe failed", exc_info=True)
            return []
        parsed = MessageToDict(response, preserving_proto_field_name=True)
        models = parsed.get("models")
        return models if isinstance(models, list) else []

    async def wait_for_model_ready(self) -> None:
        """Block until Triton reports the model is ready."""
        await asyncio.to_thread(self._wait_for_model_ready_sync)

    @retry(
        retry=retry_if_exception_type(TritonInferenceError),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def _wait_for_model_ready_sync(self) -> None:
        """Sync helper to poll model readiness with retries."""
        if not self.is_server_ready() or not self.is_model_ready():
            raise TritonInferenceError("Triton server is not ready yet.")

    def model_metadata(self) -> dict[str, object]:
        """Fetch model metadata as a JSON-serializable dict."""
        response = self._client.get_model_metadata(self._settings.triton_model_name)
        return MessageToDict(response, preserving_proto_field_name=True)

    def model_config(self) -> dict[str, object]:
        """Fetch model config as a JSON-serializable dict."""
        response = self._client.get_model_config(self._settings.triton_model_name)
        return MessageToDict(response, preserving_proto_field_name=True)

    async def infer(
        self, images: Iterable[str], thresholds: np.ndarray | None = None
    ) -> List[dict[str, list[list[float]]]]:
        """
        Run Triton inference for the provided base64 images.

        Args:
            images: Base64-encoded image payloads.
            thresholds: Optional FP32 array of shape [batch, 2].

        Returns:
            Parsed predictions per image.

        Raises:
            TritonInferenceError: When the request fails or returns malformed output.
        """
        return await asyncio.to_thread(self._infer_sync, list(images), thresholds)

    @retry(
        retry=retry_if_exception_type(TritonInferenceError),
        wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _infer_sync(
        self, images: List[str], thresholds: np.ndarray | None = None
    ) -> List[dict[str, list[list[float]]]]:
        """Synchronous inference entrypoint used by the async wrapper."""
        if not images:
            return []
        inputs = self._build_inputs(images, thresholds)
        outputs = [InferRequestedOutput(OUTPUT_NAME, binary_data=True)]
        try:
            response = self._client.infer(
                model_name=self._settings.triton_model_name,
                inputs=inputs,
                outputs=outputs,
                client_timeout=self._settings.request_timeout_seconds,
            )
        except InferenceServerException as exc:
            raise TritonInferenceError(str(exc)) from exc
        return self._parse_output(response)

    def _build_inputs(
        self, images: List[str], thresholds: np.ndarray | None
    ) -> List[InferInput]:
        """
        Construct Triton input tensors for the request.

        Args:
            images: Base64 image payloads.
            thresholds: Optional FP32 thresholds array.

        Returns:
            List of configured Triton input tensors.
        """
        image_input = InferInput(
            INPUT_IMAGES_NAME, shape=[len(images)], datatype="BYTES"
        )
        image_input.set_data_from_numpy(
            np.array([value.encode("utf-8") for value in images], dtype=np.object_)
        )

        if thresholds is None:
            return [image_input]
        threshold_input = InferInput(
            THRESHOLDS_NAME, shape=list(thresholds.shape), datatype="FP32"
        )
        threshold_input.set_data_from_numpy(np.asarray(thresholds, dtype=np.float32))
        return [image_input, threshold_input]

    def _parse_output(self, response) -> List[dict[str, list[list[float]]]]:
        """
        Decode the Triton response into parsed predictions.

        Args:
            response: Triton InferenceResult object.

        Returns:
            Parsed predictions per image.

        Raises:
            TritonInferenceError: When the output tensor is missing or malformed.
        """
        output = response.as_numpy(OUTPUT_NAME)
        if output is None:
            raise TritonInferenceError("Triton response is missing OUTPUT tensor.")
        flattened = output.reshape(-1).tolist()
        parsed: List[dict[str, list[list[float]]]] = []
        for entry in flattened:
            text = (
                entry.decode("utf-8")
                if isinstance(entry, (bytes, bytearray))
                else str(entry)
            )
            try:
                parsed.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise TritonInferenceError(
                    "Failed to parse Triton output payload."
                ) from exc
        return parsed
