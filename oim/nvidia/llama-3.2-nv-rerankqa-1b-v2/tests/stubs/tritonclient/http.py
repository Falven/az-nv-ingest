from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from .utils import InferenceServerException


class InferInput:
    """Minimal stub capturing input data."""

    def __init__(self, name: str, shape: List[int], datatype: str):
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self._data = None

    def set_data_from_numpy(self, array: np.ndarray, binary_data: bool = True) -> None:
        self._data = array


class InferRequestedOutput:
    """Placeholder output selector."""

    def __init__(self, name: str):
        self.name = name


class _FakeResponse:
    """Response wrapper mimicking tritonclient response API."""

    def __init__(self, outputs: dict[str, np.ndarray]):
        self._outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray | None:
        return self._outputs.get(name)


class InferenceServerClient:
    """Lightweight stand-in for tritonclient.http.InferenceServerClient."""

    def __init__(
        self,
        url: str,
        verbose: bool | int = False,
        connection_timeout: float | None = None,
        network_timeout: float | None = None,
    ):
        self.url = url
        self.verbose = bool(verbose)
        self.connection_timeout = connection_timeout
        self.network_timeout = network_timeout
        self._model_name: str | None = None

    def get_model_metadata(self, model_name: str) -> dict:
        self._model_name = model_name
        return {"name": model_name, "versions": ["1"], "outputs": [{"name": "SCORES"}]}

    def get_model_config(self, model_name: str) -> dict:
        self._model_name = model_name
        return {"max_batch_size": 64}

    def get_model_repository_index(self) -> list[dict]:
        name = self._model_name or "llama_3_2_nv_rerankqa_1b_v2"
        return [{"name": name}]

    def is_server_ready(self) -> bool:
        return True

    def is_model_ready(self, model_name: str) -> bool:
        return True

    def is_server_live(self) -> bool:
        return True

    def infer(
        self,
        model_name: str,
        inputs: list[InferInput],
        outputs: list[InferRequestedOutput],
        timeout: float | None = None,
    ) -> _FakeResponse:
        query_input = self._get_input(inputs, "QUERY")
        passages_input = self._get_input(inputs, "PASSAGES")
        if query_input is None or passages_input is None:
            raise InferenceServerException("Missing QUERY or PASSAGES input")
        query_text = self._first_text(query_input._data)
        passages = self._as_text_list(passages_input._data)
        scores = self._score_passages(query_text, passages)
        return _FakeResponse({"SCORES": scores.astype("float32")})

    @staticmethod
    def _get_input(inputs: list[InferInput], name: str) -> InferInput | None:
        for item in inputs:
            if item.name == name:
                return item
        return None

    @staticmethod
    def _first_text(array: np.ndarray | None) -> str:
        if array is None:
            return ""
        flattened = array.reshape(-1)
        if flattened.size == 0:
            return ""
        value = flattened[0]
        return (
            value.decode("utf-8")
            if isinstance(value, (bytes, bytearray))
            else str(value)
        )

    @staticmethod
    def _as_text_list(array: np.ndarray | None) -> list[str]:
        if array is None:
            return []
        values = array.reshape(-1)
        passages: list[str] = []
        for value in values:
            if isinstance(value, (bytes, bytearray)):
                passages.append(value.decode("utf-8"))
            else:
                passages.append(str(value))
        return passages

    @staticmethod
    def _score_passages(query: str, passages: list[str]) -> np.ndarray:
        scores: list[float] = []
        for index, passage in enumerate(passages):
            digest = hashlib.sha256(
                f"{query}::{passage}::{index}".encode("utf-8")
            ).digest()
            value = int.from_bytes(digest[:4], byteorder="big", signed=False) / float(
                2**32
            )
            scores.append((value * 2.0) - 1.0)
        return np.asarray(scores, dtype="float32")
