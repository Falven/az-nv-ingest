from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TRUNCATION_CODES: Dict[int, str] = {0: "NONE", 1: "END", 2: "START"}


def _select_device() -> str:
    """
    Choose the execution device based on CUDA availability.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def _select_dtype(device: str) -> torch.dtype:
    """
    Choose a preferred dtype for the selected device.
    """
    if device == "cuda":
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _get_param(params: Dict[str, Dict[str, str]], key: str, default: str) -> str:
    """
    Retrieve a string parameter from the Triton config parameters.
    """
    value = params.get(key, {}).get("string_value")
    return value if value is not None else default


def _format_pair(query_text: str, passage_text: str) -> str:
    """
    Apply the rerank prompt template to a query/passage pair.
    """
    return f"question:{query_text} \\n \\n passage:{passage_text}"


def _as_text(value: object) -> str:
    """
    Convert a raw tensor element into a normalized string.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _mock_scores(query: str, passages: Sequence[str]) -> np.ndarray:
    """
    Produce deterministic mock scores for environments without model weights.
    """
    scores: List[float] = []
    for index, passage in enumerate(passages):
        digest = hashlib.sha256(f"{query}::{passage}::{index}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:4], byteorder="big", signed=False) / float(2**32)
        scores.append((value * 2.0) - 1.0)
    return np.asarray(scores, dtype=np.float32)


class TritonPythonModel:
    """
    Triton Python backend implementing rerank inference.
    """

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Load model assets and configuration at backend startup.
        """
        config = json.loads(args["model_config"])
        params = config.get("parameters", {})
        self._max_batch_size = int(config.get("max_batch_size", 0) or 0)
        self._model_id = _get_param(
            params, "model_id", "nvidia/llama-3.2-nv-rerankqa-1b-v2"
        )
        max_seq_param = _get_param(params, "max_sequence_length", "8192")
        self._max_sequence_length = int(os.getenv("MAX_SEQUENCE_LENGTH", max_seq_param))
        enable_mock_param = _get_param(params, "enable_mock_inference", "0")
        self._enable_mock = os.getenv("ENABLE_MOCK_INFERENCE", enable_mock_param) == "1"

        self._device = _select_device()
        self._dtype = _select_dtype(self._device)
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForSequenceClassification | None = None

        if self._enable_mock:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, trust_remote_code=True, padding_side="left"
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.truncation_side = "right"

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_id,
            trust_remote_code=True,
            torch_dtype=self._dtype,
        ).to(self._device)
        if self._model.config.pad_token_id is None:
            self._model.config.pad_token_id = self._tokenizer.pad_token_id
        self._model.eval()

        configured_max_length = (
            self._max_sequence_length
            or getattr(self._model.config, "max_position_embeddings", None)
            or 8192
        )
        self._max_sequence_length = int(configured_max_length)

    def execute(
        self, requests: Iterable[pb_utils.InferenceRequest]
    ) -> List[pb_utils.InferenceResponse]:
        """
        Handle one or more inference requests from Triton.
        """
        responses: List[pb_utils.InferenceResponse] = []
        for request in requests:
            try:
                query, passages, truncate_code = self._parse_request(request)
                scores = self._compute_scores(query, passages, truncate_code)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor("SCORES", scores)]
                    )
                )
            except Exception as exc:  # pragma: no cover - surfaced to Triton
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))
                )
        return responses

    def _parse_request(
        self, request: pb_utils.InferenceRequest
    ) -> Tuple[str, List[str], int]:
        """
        Decode and validate incoming request tensors.
        """
        query_tensor = pb_utils.get_input_tensor_by_name(request, "QUERY")
        passages_tensor = pb_utils.get_input_tensor_by_name(request, "PASSAGES")
        if query_tensor is None or passages_tensor is None:
            raise ValueError("QUERY and PASSAGES inputs are required")

        query_values = query_tensor.as_numpy().reshape(-1)
        if query_values.size != 1:
            raise ValueError("QUERY input must contain exactly one item")
        query_text = _as_text(query_values[0]).strip()
        if not query_text:
            raise ValueError("QUERY input must not be empty")

        passage_values = passages_tensor.as_numpy().reshape(-1)
        passages: List[str] = []
        for raw_value in passage_values:
            normalized = _as_text(raw_value).strip()
            if normalized:
                passages.append(normalized)
        if not passages:
            raise ValueError("PASSAGES input must include at least one entry")
        if self._max_batch_size > 0 and len(passages) > self._max_batch_size:
            raise ValueError(
                f"Batch size {len(passages)} exceeds limit {self._max_batch_size}"
            )

        truncate_tensor = pb_utils.get_input_tensor_by_name(request, "TRUNCATE")
        truncate_code = self._parse_truncate(truncate_tensor)
        return query_text, passages, truncate_code

    @staticmethod
    def _parse_truncate(tensor: Optional[pb_utils.Tensor]) -> int:
        """
        Extract the truncate mode code from the optional TRUNCATE tensor.
        """
        if tensor is None:
            return 1
        values = tensor.as_numpy().reshape(-1)
        if values.size == 0:
            return 1
        try:
            raw_value = int(values[0])
        except Exception:
            return 1
        return raw_value if raw_value in TRUNCATION_CODES else 1

    def _compute_scores(
        self, query: str, passages: Sequence[str], truncate_code: int
    ) -> np.ndarray:
        """
        Run the rerank model or mock fallback and return scores.
        """
        if self._enable_mock:
            return _mock_scores(query, passages)
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Model assets are not available")

        pairs = [_format_pair(query, passage) for passage in passages]
        encoded = self._encode_pairs(pairs, truncate_code)
        with torch.inference_mode():
            logits = self._model(**encoded).logits
        return logits.view(-1).float().cpu().numpy()

    def _encode_pairs(
        self, pairs: Sequence[str], truncate_code: int
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize input pairs with the requested truncation behavior.
        """
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized")

        truncation_mode = TRUNCATION_CODES.get(truncate_code, "END")
        truncation_side = "left" if truncation_mode == "START" else "right"
        should_truncate = truncation_mode != "NONE"
        previous_side = tokenizer.truncation_side
        tokenizer.truncation_side = truncation_side
        try:
            encoded = tokenizer(
                list(pairs),
                padding=True,
                truncation=should_truncate,
                max_length=self._max_sequence_length,
                return_tensors="pt",
            )
        finally:
            tokenizer.truncation_side = previous_side
        return {key: value.to(self._device) for key, value in encoded.items()}
