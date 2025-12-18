from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from fastapi import HTTPException, status
from oim_common.logging import get_logger
from oim_common.triton import (
    TritonHttpClient,
    parse_max_batch_size,
    resolve_max_batch_size,
    validate_batch_size,
    validate_requested_model,
)
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

logger = get_logger(__name__)

TRUNCATION_CODES: Dict[TruncationMode, int] = {
    "NONE": 0,
    "END": 1,
    "START": 2,
}


class TritonRerankClient:
    """
    Thin wrapper around the Triton HTTP client for rerank inference.
    """

    def __init__(self, settings: ServiceSettings):
        self._settings = settings
        self._triton_client = TritonHttpClient(
            endpoint=settings.triton_http_endpoint,
            model_name=settings.triton_model_name,
            timeout=settings.triton_timeout,
            verbose=bool(settings.log_verbose),
        )
        self._client = self._triton_client.client
        self._model_name = self._triton_client.model_name
        metadata = self._triton_client.model_metadata()
        config = self._triton_client.model_config()
        self._max_batch_size = resolve_max_batch_size(
            parse_max_batch_size(config), settings.max_batch_size
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
        return self._triton_client.is_ready()

    def is_live(self) -> bool:
        """
        Report liveness from the Triton server.
        """
        return self._triton_client.is_live()

    def model_metadata(self) -> Dict[str, Any]:
        """
        Fetch model metadata for Triton v2 endpoints.
        """
        return self._triton_client.model_metadata()

    def model_config(self) -> Dict[str, Any]:
        """
        Fetch model config for Triton v2 endpoints.
        """
        return self._triton_client.model_config()

    def repository_index(self) -> List[Dict[str, Any]]:
        """
        Return the model repository index.
        """
        return list(self._triton_client.repository_index())

    def rerank(self, request_body: RankingRequest) -> Dict[str, Any]:
        """
        Execute reranking via Triton and format the response payload.
        """
        validate_requested_model(request_body.model, self._settings.model_id)
        validate_batch_size(len(request_body.passages), self._max_batch_size)
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
