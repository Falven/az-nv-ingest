from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tritonclient import grpc as triton_grpc
from tritonclient import http as triton_http
from tritonclient.utils import InferenceServerException

logger = logging.getLogger(__name__)


def parse_max_batch_size(config: Dict[str, Any]) -> Optional[int]:
    """
    Extract max_batch_size from a Triton model config payload.
    """
    max_batch = config.get("max_batch_size")
    return int(max_batch) if isinstance(max_batch, int) and max_batch > 0 else None


class TritonHttpClient:
    """
    Lightweight Triton HTTP client with readiness helpers and retries.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        timeout: float,
        verbose: bool = False,
    ):
        self._client = triton_http.InferenceServerClient(
            url=endpoint,
            verbose=verbose,
            connection_timeout=timeout,
            network_timeout=timeout,
        )
        self.model_name = model_name
        self.timeout = timeout

    def is_ready(self) -> bool:
        try:
            return bool(
                self._client.is_server_ready()
                and self._client.is_model_ready(model_name=self.model_name)
            )
        except Exception:
            return False

    def is_live(self) -> bool:
        try:
            return bool(self._client.is_server_live())
        except Exception:
            return False

    def model_metadata(self) -> Dict[str, Any]:
        return self._client.get_model_metadata(model_name=self.model_name)

    def model_config(self) -> Dict[str, Any]:
        return self._client.get_model_config(model_name=self.model_name)

    def repository_index(self) -> Iterable[Dict[str, Any]]:
        return self._client.get_model_repository_index()


class TritonGrpcClient:
    """
    Lightweight Triton gRPC client with readiness helpers.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        timeout: float,
        verbose: bool = False,
    ):
        self._client = triton_grpc.InferenceServerClient(
            url=endpoint,
            verbose=verbose,
            network_timeout=timeout,
        )
        self.model_name = model_name
        self.timeout = timeout

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            logger.debug("Failed to close Triton gRPC client cleanly", exc_info=True)

    def is_ready(self) -> bool:
        try:
            return bool(self._client.is_model_ready(self.model_name))
        except Exception:
            return False

    def is_live(self) -> bool:
        try:
            return bool(self._client.is_server_ready())
        except Exception:
            return False

    def model_metadata(self) -> Dict[str, Any]:
        response = self._client.get_model_metadata(self.model_name, as_json=True)
        return response if isinstance(response, dict) else {}

    def model_config(self) -> Dict[str, Any]:
        response = self._client.get_model_config(self.model_name, as_json=True)
        return response if isinstance(response, dict) else {}

    def repository_index(self) -> Iterable[Dict[str, Any]]:
        response = self._client.get_model_repository_index()
        return response if isinstance(response, list) else []

    @retry(
        retry=retry_if_exception_type(InferenceServerException),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def wait_for_ready(self) -> None:
        if not self.is_ready():
            raise InferenceServerException("Triton server is not ready yet.")
