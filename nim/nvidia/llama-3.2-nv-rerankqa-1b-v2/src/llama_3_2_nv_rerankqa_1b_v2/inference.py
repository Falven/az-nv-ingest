from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

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

from .models import (
    PASSAGES_INPUT_NAME,
    QUERY_INPUT_NAME,
    SCORES_OUTPUT_NAME,
    TRUNCATE_INPUT_NAME,
    RankingRequest,
    TruncationMode,
)
from .settings import ServiceSettings

logger = logging.getLogger(__name__)

TRUNCATION_CODES: Dict[TruncationMode, int] = {
    TruncationMode.NONE: 0,
    TruncationMode.END: 1,
    TruncationMode.START: 2,
}


def _parse_max_batch_size(config: Dict[str, Any]) -> Optional[int]:
    """
    Read the Triton max_batch_size from the model config payload.
    """
    max_batch = config.get("max_batch_size")
    return int(max_batch) if isinstance(max_batch, int) and max_batch > 0 else None


def _resolve_batch_limit(
    config_limit: Optional[int], settings_limit: Optional[int]
) -> Optional[int]:
    """
    Combine Triton config and settings batch size limits.
    """
    if config_limit is not None and settings_limit is not None:
        return min(config_limit, settings_limit)
    if config_limit is not None:
        return config_limit
    return settings_limit


class TritonRerankClient:
    """
    Thin wrapper around the Triton HTTP client for rerank inference.
    """

    def __init__(self, settings: ServiceSettings):
        self._settings = settings
        self._client = triton_http.InferenceServerClient(
            url=settings.triton_http_endpoint,
            verbose=bool(settings.log_verbose),
            connection_timeout=settings.triton_timeout,
            network_timeout=settings.triton_timeout,
        )
        self._model_name = settings.triton_model_name
        metadata = self._client.get_model_metadata(model_name=self._model_name)
        config = self._client.get_model_config(model_name=self._model_name)
        self._max_batch_size = _resolve_batch_limit(
            _parse_max_batch_size(config), settings.max_batch_size
        )
        logger.info(
            "Loaded Triton model %s (versions=%s, max_batch=%s)",
            metadata.get("name"),
            metadata.get("versions"),
            self._max_batch_size,
        )

    @property
    def max_batch_size(self) -> Optional[int]:
        """
        Return the resolved batch size limit.
        """
        return self._max_batch_size

    def is_ready(self) -> bool:
        """
        Report readiness from the Triton server and model.
        """
        try:
            return bool(
                self._client.is_server_ready()
                and self._client.is_model_ready(model_name=self._model_name)
            )
        except Exception:
            return False

    def is_live(self) -> bool:
        """
        Report liveness from the Triton server.
        """
        try:
            return bool(self._client.is_server_live())
        except Exception:
            return False

    def model_metadata(self) -> Dict[str, Any]:
        """
        Fetch model metadata for Triton v2 endpoints.
        """
        return self._client.get_model_metadata(model_name=self._model_name)

    def model_config(self) -> Dict[str, Any]:
        """
        Fetch model config for Triton v2 endpoints.
        """
        return self._client.get_model_config(model_name=self._model_name)

    def repository_index(self) -> List[Dict[str, Any]]:
        """
        Return the model repository index.
        """
        return self._client.get_model_repository_index()

    def rerank(self, request_body: RankingRequest) -> Dict[str, Any]:
        """
        Execute reranking via Triton and format the response payload.
        """
        self._validate_model_name(request_body.model)
        self._validate_batch_size(len(request_body.passages))
        passages = [passage.text for passage in request_body.passages]
        try:
            scores = self._infer(
                query=request_body.query.text,
                passages=passages,
                truncate=request_body.truncate,
            )
        except InferenceServerException as exc:
            logger.exception("Triton inference failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Triton inference failed",
            ) from exc
        if scores.size == 0:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Triton response missing scores",
            )
        rankings = self._format_rankings(scores)
        return {"rankings": rankings}

    def _validate_model_name(self, model_name: Optional[str]) -> None:
        if model_name is None:
            return
        if model_name != self._settings.model_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported model '{model_name}', expected '{self._settings.model_id}'",
            )

    def _validate_batch_size(self, item_count: int) -> None:
        if self._max_batch_size is None:
            return
        if item_count > self._max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size {item_count} exceeds limit {self._max_batch_size}",
            )

    @retry(
        retry=retry_if_exception_type(InferenceServerException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
        reraise=True,
    )
    def _infer(
        self, query: str, passages: Sequence[str], truncate: TruncationMode
    ) -> np.ndarray:
        inputs = self._build_inputs(query, passages, truncate)
        outputs = [triton_http.InferRequestedOutput(SCORES_OUTPUT_NAME)]
        response = self._client.infer(
            model_name=self._model_name,
            inputs=inputs,
            outputs=outputs,
            timeout=self._settings.triton_timeout,
        )
        scores = response.as_numpy(SCORES_OUTPUT_NAME)
        if scores is None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Triton response missing scores",
            )
        return scores.astype("float32").reshape(-1)

    def _build_inputs(
        self, query: str, passages: Sequence[str], truncate: TruncationMode
    ) -> List[triton_http.InferInput]:
        query_input = triton_http.InferInput(QUERY_INPUT_NAME, [1], "BYTES")
        query_input.set_data_from_numpy(np.asarray([query], dtype=np.object_))

        passages_input = triton_http.InferInput(
            PASSAGES_INPUT_NAME, [len(passages)], "BYTES"
        )
        passages_input.set_data_from_numpy(np.asarray(passages, dtype=np.object_))

        truncate_code = np.asarray([TRUNCATION_CODES.get(truncate, 1)], dtype=np.int32)
        truncate_input = triton_http.InferInput(TRUNCATE_INPUT_NAME, [1], "INT32")
        truncate_input.set_data_from_numpy(truncate_code)

        return [query_input, passages_input, truncate_input]

    def _format_rankings(self, scores: np.ndarray) -> List[Dict[str, float]]:
        rankings: List[Dict[str, float]] = []
        for index, score in enumerate(scores.tolist()):
            rankings.append({"index": index, "logit": float(score)})
        return sorted(rankings, key=lambda item: item["logit"], reverse=True)
