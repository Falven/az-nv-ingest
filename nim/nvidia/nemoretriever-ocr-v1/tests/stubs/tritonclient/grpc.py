from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np


class InferInput:
    """
    Minimal stub for tritonclient.grpc.InferInput.
    """

    def __init__(self, name: str, shape: List[int], datatype: str):
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self._data: Optional[np.ndarray] = None

    def set_data_from_numpy(self, array: np.ndarray, binary_data: bool = True) -> None:
        self._data = array


class InferRequestedOutput:
    """
    Placeholder output selector.
    """

    def __init__(self, name: str, binary_data: bool = True):
        self.name = name
        self.binary_data = binary_data


class _FakeResponse:
    """
    Response wrapper providing the as_numpy API.
    """

    def __init__(self, outputs: Dict[str, np.ndarray]):
        self._outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray | None:
        return self._outputs.get(name)


class InferenceServerClient:
    """
    Lightweight stand-in for tritonclient.grpc.InferenceServerClient used by contract tests.
    """

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

    def get_model_metadata(
        self, model_name: str, as_json: bool = False
    ) -> Dict[str, Any]:
        metadata = {"name": model_name, "versions": ["1"], "max_batch_size": 8}
        return metadata if as_json else metadata

    def infer(
        self,
        model_name: str,
        inputs: List[InferInput],
        outputs: List[InferRequestedOutput],
        client_timeout: float | None = None,
    ) -> _FakeResponse:
        batch_size = self._batch_size(inputs)
        payload: List[List[Any]] = []
        for index in range(batch_size):
            offset = 0.05 * index
            boxes = [
                [
                    [0.1 + offset, 0.1],
                    [0.9, 0.1],
                    [0.9, 0.9],
                    [0.1 + offset, 0.9],
                ]
            ]
            texts = [f"mock-{index}"]
            confidences = [0.99]
            payload.append(
                [
                    json.dumps(boxes),
                    json.dumps(texts),
                    json.dumps(confidences),
                ]
            )
        output_array = np.array(payload, dtype=object)
        return _FakeResponse({"OUTPUT": output_array})

    @staticmethod
    def _batch_size(inputs: List[InferInput]) -> int:
        for item in inputs:
            if item.name == "INPUT_IMAGE_URLS" and item._data is not None:
                return int(item._data.shape[0])
        return 0
