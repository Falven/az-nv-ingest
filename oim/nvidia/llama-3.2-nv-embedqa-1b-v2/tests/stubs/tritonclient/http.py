from __future__ import annotations

import numpy as np


class InferInput:
    """Minimal stub capturing input data."""

    def __init__(self, name: str, shape: list[int], datatype: str):
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
        return {
            "name": model_name,
            "outputs": [
                {"name": "EMBEDDINGS", "shape": [-1, 2048]},
                {"name": "TOKENS", "shape": [-1]},
            ],
        }

    def get_model_config(self, model_name: str) -> dict:
        self._model_name = model_name
        return {"max_batch_size": 30}

    def get_model_repository_index(self) -> list[dict]:
        name = self._model_name or "llama_3_2_nv_embedqa_1b_v2"
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
        text_input = self._get_input(inputs, "TEXT")
        dims_input = self._get_input(inputs, "DIMENSIONS")
        texts = (
            text_input._data.reshape(-1)
            if getattr(text_input, "_data", None) is not None
            else np.asarray([])
        )
        dim_value = 0
        if dims_input is not None and getattr(dims_input, "_data", None) is not None:
            dim_array = dims_input._data.reshape(-1)
            if dim_array.size:
                dim_value = int(dim_array[0])
        target_dim = dim_value if dim_value > 0 else 2048
        embeddings = []
        token_counts = []
        for value in texts:
            text = (
                value.decode("utf-8")
                if isinstance(value, (bytes, bytearray))
                else str(value)
            )
            rng = np.random.default_rng(abs(hash(text)) % 10_000)
            vector = rng.standard_normal(target_dim).astype("float32")
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            embeddings.append(vector)
            token_counts.append(max(len(text.split()), 1))
        emb_array = (
            np.stack(embeddings, axis=0).astype("float32")
            if embeddings
            else np.zeros((0, target_dim), dtype="float32")
        )
        token_array = np.asarray(token_counts, dtype="int32")
        return _FakeResponse({"EMBEDDINGS": emb_array, "TOKENS": token_array})

    @staticmethod
    def _get_input(inputs: list[InferInput], name: str) -> InferInput | None:
        for item in inputs:
            if item.name == name:
                return item
        return None
