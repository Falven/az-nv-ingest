from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np


class InferInput:
    """Minimal stub capturing input data."""

    def __init__(self, name: str, shape: List[int], datatype: str):
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self._data: Optional[np.ndarray] = None

    def set_data_from_numpy(self, array: np.ndarray, binary_data: bool = True) -> None:
        self._data = array


class InferRequestedOutput:
    """Placeholder output selector."""

    def __init__(self, name: str, binary_data: bool = True):
        self.name = name
        self.binary_data = binary_data


class _FakeResponse:
    """Response wrapper mimicking tritonclient response API."""

    def __init__(self, outputs: Dict[str, np.ndarray]):
        self._outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray | None:
        return self._outputs.get(name)


class InferenceServerClient:
    """Lightweight stand-in for tritonclient.grpc.InferenceServerClient."""

    def __init__(
        self,
        url: str,
        verbose: bool | int = False,
        network_timeout: float | None = None,
    ):
        self.url = url
        self.verbose = bool(verbose)
        self.network_timeout = network_timeout

    def close(self) -> None:
        return

    def is_server_live(self) -> bool:
        return True

    def is_server_ready(self) -> bool:
        return True

    def is_model_ready(self, model_name: str) -> bool:
        return True

    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        return {"name": model_name, "versions": ["1"], "inputs": [], "outputs": []}

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        return {"name": model_name, "max_batch_size": 8}

    def infer(
        self,
        model_name: str,
        inputs: List[InferInput],
        outputs: List[InferRequestedOutput],
        client_timeout: float | None = None,
    ) -> _FakeResponse:
        image_input = self._get_input(inputs, "INPUT_IMAGES")
        predictions: List[Dict[str, Any]] = []
        values = (
            image_input._data.reshape(-1)
            if image_input and image_input._data is not None
            else []
        )
        for index, _ in enumerate(values):
            offset = 0.05 * index
            predictions.append(
                {
                    "chart_title": [[0.1 + offset, 0.1, 0.4, 0.2 + offset, 0.92]],
                    "legend_title": [],
                    "other": [[0.6, 0.6, 0.9, 0.9, 0.55]],
                }
            )
        payload = np.array([json.dumps(item) for item in predictions], dtype=object)
        return _FakeResponse({"OUTPUT": payload})

    @staticmethod
    def _get_input(inputs: List[InferInput], name: str) -> InferInput | None:
        for item in inputs:
            if item.name == name:
                return item
        return None
