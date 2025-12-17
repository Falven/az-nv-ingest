from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils
from transformers import AutoModel, AutoTokenizer


def _select_device() -> str:
    """
    Choose the execution device based on CUDA availability.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def _select_dtype(device: str) -> torch.dtype:
    """
    Choose the preferred dtype for the selected device.
    """
    if device == "cuda":
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _average_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute mean pooled embeddings with attention masking and L2 normalization.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    masked = last_hidden_state.masked_fill(~mask.bool(), 0.0)
    summed = masked.sum(dim=1)
    counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
    return F.normalize(summed / counts, p=2, dim=-1)


def _maybe_downproject(
    embeddings: torch.Tensor, requested_dim: Optional[int], full_dim: int
) -> torch.Tensor:
    """
    Optionally slice the embedding dimension and renormalize.
    """
    if requested_dim is None:
        return embeddings
    if requested_dim > full_dim:
        raise ValueError(
            f"Requested dimensions {requested_dim} exceeds embedding size {full_dim}"
        )
    sliced = embeddings[:, :requested_dim]
    return F.normalize(sliced, p=2, dim=-1)


def _mock_embeddings(
    texts: List[str],
    embedding_dim: int,
    requested_dim: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce deterministic mock embeddings for environments without model weights.
    """
    target_dim = requested_dim or embedding_dim
    if target_dim > embedding_dim:
        raise ValueError(
            f"Requested dimensions {target_dim} exceeds embedding size {embedding_dim}"
        )
    vectors: List[np.ndarray] = []
    token_counts: List[int] = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(embedding_dim).astype("float32")
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        vectors.append(vector[:target_dim])
        token_counts.append(max(len(text.split()), 1))
    stacked = np.stack(vectors, axis=0).astype("float32")
    counts = np.asarray(token_counts, dtype=np.int32)
    return stacked, counts


def _get_param(params: Dict[str, Dict[str, str]], key: str, default: str) -> str:
    """
    Retrieve a string parameter from the Triton model config parameters.
    """
    value = params.get(key, {}).get("string_value")
    return value if value is not None else default


class TritonPythonModel:
    """
    Triton Python backend implementing embedding inference.
    """

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Load model assets and configuration when the backend initializes.
        """
        config = json.loads(args["model_config"])
        params = config.get("parameters", {})
        self._max_batch_size = int(config.get("max_batch_size", 0) or 0)
        self._model_id = _get_param(
            params, "model_id", "nvidia/llama-3.2-nv-embedqa-1b-v2"
        )
        self._max_sequence_length = int(
            os.getenv(
                "MAX_SEQUENCE_LENGTH",
                _get_param(params, "max_sequence_length", "8192"),
            )
        )
        enable_mock_param = _get_param(params, "enable_mock_inference", "0")
        self._enable_mock = os.getenv("ENABLE_MOCK_INFERENCE", enable_mock_param) == "1"
        mock_dim_param = _get_param(
            params,
            "mock_embedding_dim",
            str(config.get("output", [{}])[0].get("dims", [2048])[-1]),
        )
        self._mock_embedding_dim = int(os.getenv("MOCK_EMBEDDING_DIM", mock_dim_param))

        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModel | None = None
        self._device = _select_device()
        self._dtype = _select_dtype(self._device)

        if self._enable_mock:
            self._embedding_dim = self._mock_embedding_dim
            return

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.truncation_side = "right"
        self._model = AutoModel.from_pretrained(
            self._model_id,
            trust_remote_code=True,
            torch_dtype=self._dtype,
        ).to(self._device)
        if self._model.config.pad_token_id is None:
            self._model.config.pad_token_id = self._tokenizer.pad_token_id

        configured_max_length = (
            self._max_sequence_length
            or getattr(self._model.config, "max_position_embeddings", None)
            or 8192
        )
        embedding_dim = (
            getattr(self._model.config, "hidden_size", None)
            or getattr(self._model.config, "d_model", None)
            or 2048
        )
        self._max_sequence_length = int(configured_max_length)
        self._embedding_dim = int(embedding_dim)

    def execute(
        self, requests: Iterable[pb_utils.InferenceRequest]
    ) -> List[pb_utils.InferenceResponse]:
        """
        Handle one or more inference requests from Triton.
        """
        responses: List[pb_utils.InferenceResponse] = []
        for request in requests:
            try:
                texts, dims, truncate_mode = self._parse_request(request)
                embeddings, token_counts = self._compute_embeddings(
                    texts, dims, truncate_mode
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("EMBEDDINGS", embeddings),
                            pb_utils.Tensor("TOKENS", token_counts),
                        ]
                    )
                )
            except Exception as exc:  # pragma: no cover - surfaced to Triton
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))
                )
        return responses

    def _parse_request(
        self, request: pb_utils.InferenceRequest
    ) -> Tuple[List[str], Optional[int], int]:
        """
        Decode inputs and validate request limits.
        """
        text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
        if text_tensor is None:
            raise ValueError("TEXT input is required")
        text_array = text_tensor.as_numpy()
        flat_texts = text_array.reshape(-1).tolist()
        texts: List[str] = []
        for value in flat_texts:
            decoded = (
                value.decode("utf-8")
                if isinstance(value, (bytes, bytearray))
                else str(value)
            ).strip()
            if decoded:
                texts.append(decoded)
        if not texts:
            raise ValueError("No non-empty TEXT values provided")
        if self._max_batch_size and len(texts) > self._max_batch_size:
            raise ValueError(
                f"Batch size {len(texts)} exceeds limit {self._max_batch_size}"
            )

        dims_tensor = pb_utils.get_input_tensor_by_name(request, "DIMENSIONS")
        dims_value: Optional[int] = None
        if dims_tensor is not None:
            dims_array = dims_tensor.as_numpy().reshape(-1)
            if dims_array.size > 0:
                dims_candidate = int(dims_array[0])
                if dims_candidate > 0:
                    dims_value = dims_candidate

        truncate_tensor = pb_utils.get_input_tensor_by_name(request, "TRUNCATE")
        truncate_mode = 1  # END
        if truncate_tensor is not None:
            truncate_array = truncate_tensor.as_numpy().reshape(-1)
            if truncate_array.size > 0:
                truncate_value = int(truncate_array[0])
                if truncate_value in (0, 1, 2):
                    truncate_mode = truncate_value

        return texts, dims_value, truncate_mode

    def _compute_embeddings(
        self, texts: List[str], dims: Optional[int], truncate_mode: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings for the provided texts.
        """
        if self._enable_mock:
            return _mock_embeddings(texts, self._embedding_dim, dims)

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Model assets are not loaded")

        truncation_side = "left" if truncate_mode == 2 else "right"
        previous_truncation_side = self._tokenizer.truncation_side
        self._tokenizer.truncation_side = truncation_side
        try:
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=truncate_mode != 0,
                max_length=self._max_sequence_length,
                return_tensors="pt",
            )
        finally:
            self._tokenizer.truncation_side = previous_truncation_side

        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = self._model(**encoded)
        embeddings = _average_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = embeddings.to(torch.float32)
        embeddings = _maybe_downproject(embeddings, dims, self._embedding_dim)
        token_counts = encoded["attention_mask"].sum(dim=1).to(torch.int32)
        return (
            embeddings.cpu().numpy().astype("float32"),
            token_counts.cpu().numpy().astype("int32"),
        )
