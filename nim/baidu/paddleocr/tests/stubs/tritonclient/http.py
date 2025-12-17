from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np


class InferInput:
    """Minimal stub capturing input payloads."""

    def __init__(self, name: str, shape: List[int], datatype: str):
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self._data: Optional[np.ndarray] = None

    def set_data_from_numpy(self, array: np.ndarray, binary_data: bool = True) -> None:
        self._data = array


class InferRequestedOutput:
    """Placeholder output selector."""

    def __init__(self, name: str):
        self.name = name


class _FakeResponse:
    """Response wrapper mimicking tritonclient response API."""

    def __init__(self, outputs: Dict[str, np.ndarray]):
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

    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        self._model_name = model_name
        return {"name": model_name, "max_batch_size": 8}

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        self._model_name = model_name
        return {"name": model_name, "max_batch_size": 8}

    def is_server_ready(self) -> bool:
        return True

    def is_model_ready(self, model_name: str) -> bool:
        self._model_name = model_name
        return True

    def infer(
        self,
        model_name: str,
        inputs: List[InferInput],
        outputs: List[InferRequestedOutput],
        timeout: float | None = None,
    ) -> _FakeResponse:
        _ = outputs  # unused in stub
        input_tensor = self._get_input(inputs, "INPUT")
        payloads: list[str] = []
        if (
            input_tensor is not None
            and getattr(input_tensor, "_data", None) is not None
        ):
            flattened = input_tensor._data.reshape(-1)
            for item in flattened:
                payloads.append(
                    item.decode("utf-8")
                    if isinstance(item, (bytes, bytearray))
                    else str(item)
                )
        responses = [
            self._fake_detection(index, url) for index, url in enumerate(payloads)
        ]
        return _FakeResponse({"OUTPUT": np.asarray(responses, dtype=object)})

    @staticmethod
    def _get_input(inputs: List[InferInput], name: str) -> InferInput | None:
        for item in inputs:
            if item.name == name:
                return item
        return None

    @staticmethod
    def _fake_detection(index: int, url: str) -> bytes:
        box = [
            {"x": 0.1, "y": 0.1},
            {"x": 0.9, "y": 0.1},
            {"x": 0.9, "y": 0.9},
            {"x": 0.1, "y": 0.9},
        ]
        payload = {
            "text_detections": [
                {
                    "bounding_box": {"points": box, "type": "quadrilateral"},
                    "text_prediction": {"text": f"{url}:{index}", "confidence": 1.0},
                }
            ]
        }
        return json.dumps(payload).encode("utf-8")
