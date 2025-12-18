from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import HTTPException, status
from oim_common.logging import get_logger
from oim_common.triton import (
    TritonHttpClient,
    parse_max_batch_size,
    resolve_max_batch_size,
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
    DIMENSIONS_INPUT_NAME,
    OUTPUT_NAME,
    TEXT_INPUT_NAME,
    TOKENS_OUTPUT_NAME,
    TRUNCATE_INPUT_NAME,
    EmbeddingsRequest,
    EncodingFormat,
    TruncationMode,
)
from .settings import ServiceSettings

logger = get_logger(__name__)

TRUNCATION_CODES: Dict[TruncationMode, int] = {"NONE": 0, "END": 1, "START": 2}


def _prefix_texts(texts: List[str], input_type: str) -> List[str]:
    """
    Apply the model-specific prefix to each input text.
    """
    normalized_type = input_type.lower()
    if normalized_type == "query":
        prefix = "query:"
    elif normalized_type in {"document", "doc", "passage"}:
        prefix = "passage:"
    else:
        prefix = f"{normalized_type}:"
    return [f"{prefix} {text}" for text in texts]


def _to_base64(vector: np.ndarray) -> str:
    """
    Encode a vector as base64-encoded float32 bytes.
    """
    float_bytes = vector.astype("float32").tobytes()
    return base64.b64encode(float_bytes).decode("ascii")


def _parse_embedding_dim(metadata: Dict[str, Any]) -> Optional[int]:
    """
    Infer the embedding dimension from Triton model metadata.
    """
    outputs = metadata.get("outputs") or []
    for output in outputs:
        if output.get("name") != OUTPUT_NAME:
            continue
        dims = output.get("shape") or output.get("dims") or []
        if dims and dims[-1] > 0:
            return int(dims[-1])
    return None


class TritonEmbeddingClient:
    """
    Thin wrapper around the Triton HTTP client for embedding inference.
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
        self._embedding_dim = _parse_embedding_dim(metadata)
        config = self._triton_client.model_config()
        self._max_batch_size = resolve_max_batch_size(
            parse_max_batch_size(config), settings.max_batch_size
        )

    def is_ready(self) -> bool:
        """
        Report readiness based on Triton server and model status.
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

    def embed(self, request_body: EmbeddingsRequest) -> Dict[str, Any]:
        """
        Execute an embeddings request via Triton and format the OpenAI-style response.
        """
        self._validate_model_name(request_body.model)
        self._validate_batch_size(len(request_body.input))
        self._validate_dimensions(request_body.dimensions)
        prefixed_inputs = _prefix_texts(request_body.input, request_body.input_type)
        try:
            embeddings, token_counts = self._infer(
                prefixed_inputs, request_body.dimensions, request_body.truncate
            )
        except InferenceServerException as exc:
            logger.exception("Triton inference failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Triton inference failed",
            ) from exc
        data = self._format_embeddings(embeddings, request_body.encoding_format)
        total_tokens = int(np.sum(token_counts)) if token_counts.size else 0
        return {
            "object": "list",
            "model": self._settings.model_id,
            "data": data,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }

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

    def _validate_dimensions(self, dims: Optional[int]) -> None:
        if dims is None or self._embedding_dim is None:
            return
        if dims <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="dimensions must be positive",
            )
        if dims > self._embedding_dim:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"dimensions {dims} exceeds embedding size {self._embedding_dim}",
            )

    @retry(
        retry=retry_if_exception_type(InferenceServerException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
        reraise=True,
    )
    def _infer(
        self, texts: List[str], dims: Optional[int], truncate: TruncationMode
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Issue an inference request to Triton.
        """
        inputs = self._build_inputs(texts, dims, truncate)
        outputs = [
            triton_http.InferRequestedOutput(OUTPUT_NAME),
            triton_http.InferRequestedOutput(TOKENS_OUTPUT_NAME),
        ]
        response = self._client.infer(
            model_name=self._model_name,
            inputs=inputs,
            outputs=outputs,
            timeout=self._settings.triton_timeout,
        )
        embeddings = response.as_numpy(OUTPUT_NAME)
        token_counts = response.as_numpy(TOKENS_OUTPUT_NAME)
        if embeddings is None or token_counts is None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Triton response missing outputs",
            )
        return embeddings.astype("float32"), token_counts.astype("int32")

    def _build_inputs(
        self, texts: List[str], dims: Optional[int], truncate: TruncationMode
    ) -> List[triton_http.InferInput]:
        """
        Build Triton HTTP inputs from the validated request payload.
        """
        text_input = triton_http.InferInput(TEXT_INPUT_NAME, [len(texts)], "BYTES")
        text_input.set_data_from_numpy(np.asarray(texts, dtype=np.object_))

        dims_value = np.asarray([dims if dims is not None else 0], dtype=np.int32)
        dims_input = triton_http.InferInput(DIMENSIONS_INPUT_NAME, [1], "INT32")
        dims_input.set_data_from_numpy(dims_value)

        truncate_value = np.asarray([TRUNCATION_CODES.get(truncate, 1)], dtype=np.int32)
        truncate_input = triton_http.InferInput(TRUNCATE_INPUT_NAME, [1], "INT32")
        truncate_input.set_data_from_numpy(truncate_value)

        return [text_input, dims_input, truncate_input]

    def _format_embeddings(
        self,
        embeddings: np.ndarray,
        encoding_format: EncodingFormat,
    ) -> List[Dict[str, Any]]:
        """
        Convert embeddings into OpenAI-style response payloads.
        """
        data: List[Dict[str, Any]] = []
        for index, vector in enumerate(embeddings):
            embedding_payload: Any
            if encoding_format == "base64":
                embedding_payload = _to_base64(vector)
            else:
                embedding_payload = vector.astype("float32").tolist()
            data.append(
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": embedding_payload,
                    "model": self._settings.model_id,
                }
            )
        return data
