from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

from fastapi import HTTPException, status
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tritonclient import http as triton_http
from tritonclient.utils import InferenceServerException

try:
    from tritonclient import grpc as triton_grpc
except ImportError:  # pragma: no cover - optional dependency
    triton_grpc = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def parse_max_batch_size(config: Dict[str, Any]) -> Optional[int]:
    """
    Extract max_batch_size from a Triton model config payload.
    """
    max_batch = config.get("max_batch_size")
    return int(max_batch) if isinstance(max_batch, int) and max_batch > 0 else None


def resolve_max_batch_size(
    config_limit: Optional[int], settings_limit: Optional[int]
) -> Optional[int]:
    """
    Determine the effective max_batch_size from Triton config and settings.
    """
    if config_limit is not None and settings_limit is not None:
        return min(config_limit, settings_limit)
    if config_limit is not None:
        return config_limit
    return settings_limit


def validate_requested_model(
    requested_model: str | None, expected_model_id: str
) -> None:
    """
    Raise an HTTP 400 when a request targets an unexpected model identifier.
    """
    if requested_model is None:
        return
    if requested_model != expected_model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model '{requested_model}', expected '{expected_model_id}'",
        )


def validate_batch_size(item_count: int, max_batch_size: Optional[int]) -> None:
    """
    Enforce an upper bound on batch sizes when configured.
    """
    if max_batch_size is None:
        return
    if item_count > max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {item_count} exceeds limit {max_batch_size}",
        )


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

    @property
    def client(self) -> triton_http.InferenceServerClient:
        """
        Expose the underlying Triton HTTP client for inference calls.
        """
        return self._client

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
        if triton_grpc is None:
            raise RuntimeError("tritonclient[grpc] is required for gRPC clients")
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
