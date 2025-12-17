from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import numpy as np
from fastapi import HTTPException, status
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tritonclient import http as triton_http
from tritonclient.utils import InferenceServerException

from ..settings import ServiceSettings

logger = logging.getLogger(__name__)


@dataclass
class TritonCaptionRequest:
    """
    Input payload sent to the Triton caption model.
    """

    system_prompt: str
    user_prompt: str
    image_bytes: bytes
    max_new_tokens: int
    temperature: float
    top_p: float


class TritonCaptionClient:
    """
    Thin wrapper around the Triton HTTP client for caption inference.
    """

    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings
        self._client = triton_http.InferenceServerClient(
            url=settings.triton_http_endpoint,
            verbose=bool(settings.log_verbose),
            connection_timeout=settings.triton_timeout,
            network_timeout=settings.triton_timeout,
        )
        self._model_name = settings.triton_model_name

    def close(self) -> None:
        """
        Close the underlying Triton client session.
        """
        try:
            self._client.close()
        except Exception:
            logger.debug("Ignoring error during Triton client close", exc_info=True)

    def is_ready(self) -> bool:
        """
        Report readiness based on Triton server and model status.
        """
        try:
            return bool(
                self._client.is_server_ready()
                and self._client.is_model_ready(model_name=self._model_name)
            )
        except Exception:
            return False

    async def wait_for_ready(self) -> None:
        """
        Poll Triton readiness until the model is available or the timeout elapses.
        """
        deadline = time.monotonic() + self._settings.triton_timeout
        while time.monotonic() < deadline:
            if self.is_ready():
                return
            await asyncio.sleep(0.5)
        raise RuntimeError("Triton model did not become ready in time")

    def caption(self, request: TritonCaptionRequest) -> str:
        """
        Execute a caption request via Triton and return the decoded caption text.
        """
        try:
            response = self._infer(request)
        except InferenceServerException as exc:
            logger.exception("Triton inference failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Triton inference failed",
            ) from exc
        output = response.as_numpy("OUTPUT_TEXT")
        if output is None or output.size == 0:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Triton response missing OUTPUT_TEXT",
            )
        caption_bytes = output[0]
        if isinstance(caption_bytes, (bytes, bytearray, memoryview)):
            return bytes(caption_bytes).decode("utf-8", errors="replace").strip()
        return str(caption_bytes).strip()

    @retry(
        retry=retry_if_exception_type(InferenceServerException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
        reraise=True,
    )
    def _infer(self, request: TritonCaptionRequest) -> triton_http.InferResult:  # type: ignore[override]
        """
        Issue an inference request to Triton.
        """
        inputs = self._build_inputs(request)
        outputs = [triton_http.InferRequestedOutput("OUTPUT_TEXT")]
        return self._client.infer(
            model_name=self._model_name,
            inputs=inputs,
            outputs=outputs,
            timeout=self._settings.triton_timeout,
        )

    def _build_inputs(
        self, request: TritonCaptionRequest
    ) -> list[triton_http.InferInput]:
        """
        Build Triton HTTP inputs from the validated request payload.
        """
        image_input = triton_http.InferInput("IMAGE_BYTES", [1], "BYTES")
        image_input.set_data_from_numpy(
            np.asarray([request.image_bytes], dtype=np.object_)
        )

        system_input = triton_http.InferInput("SYSTEM_PROMPT", [1], "BYTES")
        system_input.set_data_from_numpy(
            np.asarray(
                [request.system_prompt.encode("utf-8")],
                dtype=np.object_,
            )
        )

        user_input = triton_http.InferInput("USER_PROMPT", [1], "BYTES")
        user_input.set_data_from_numpy(
            np.asarray(
                [request.user_prompt.encode("utf-8")],
                dtype=np.object_,
            )
        )

        max_tokens_input = triton_http.InferInput("MAX_NEW_TOKENS", [1], "INT32")
        max_tokens_input.set_data_from_numpy(
            np.asarray([request.max_new_tokens], dtype=np.int32)
        )

        temperature_input = triton_http.InferInput("TEMPERATURE", [1], "FP32")
        temperature_input.set_data_from_numpy(
            np.asarray([request.temperature], dtype=np.float32)
        )

        top_p_input = triton_http.InferInput("TOP_P", [1], "FP32")
        top_p_input.set_data_from_numpy(np.asarray([request.top_p], dtype=np.float32))

        return [
            image_input,
            system_input,
            user_input,
            max_tokens_input,
            temperature_input,
            top_p_input,
        ]
