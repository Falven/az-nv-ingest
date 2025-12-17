from __future__ import annotations

import numpy as np

from .utils import InferenceServerException


class InferInput:
    """
    Minimal stub capturing input data.
    """

    def __init__(self, name: str, shape: list[int], datatype: str):
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self._data: np.ndarray | None = None

    def set_data_from_numpy(self, array: np.ndarray, binary_data: bool = True) -> None:  # noqa: ARG002
        self._data = array


class InferRequestedOutput:
    """
    Placeholder output selector.
    """

    def __init__(self, name: str):
        self.name = name


class _FakeResponse:
    """
    Response wrapper mimicking tritonclient response API.
    """

    def __init__(self, outputs: dict[str, np.ndarray]):
        self._outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray | None:
        return self._outputs.get(name)


class InferenceServerClient:
    """
    Lightweight stand-in for tritonclient.http.InferenceServerClient.
    """

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

    def close(self) -> None:
        return

    def is_server_ready(self) -> bool:
        return True

    def is_model_ready(self, model_name: str) -> bool:  # noqa: ARG002
        return True

    def infer(
        self,
        model_name: str,  # noqa: ARG002
        inputs: list[InferInput],
        outputs: list[InferRequestedOutput] | None = None,  # noqa: ARG002
        timeout: float | None = None,  # noqa: ARG002
    ) -> _FakeResponse:
        image_input = self._get_input(inputs, "IMAGE_BYTES")
        user_input = self._get_input(inputs, "USER_PROMPT")
        system_input = self._get_input(inputs, "SYSTEM_PROMPT")
        max_tokens_input = self._get_input(inputs, "MAX_NEW_TOKENS")
        temperature_input = self._get_input(inputs, "TEMPERATURE")
        top_p_input = self._get_input(inputs, "TOP_P")

        if image_input is None or getattr(image_input, "_data", None) is None:
            raise InferenceServerException("IMAGE_BYTES input missing")

        user_prompt = self._first_text(user_input)
        system_prompt = self._first_text(system_input)
        max_tokens = self._first_int(max_tokens_input) or 512
        temperature = self._first_float(temperature_input) or 1.0
        top_p = self._first_float(top_p_input) or 1.0

        caption_parts = [
            part.strip()
            for part in (system_prompt, user_prompt)
            if isinstance(part, str) and part.strip()
        ]
        caption = " ".join(caption_parts) if caption_parts else "stub caption"
        if max_tokens > 0:
            caption = caption[:max_tokens]
        summary = f"[stub] top_p={top_p:g} temp={temperature:g} {caption}".strip()
        output_bytes = np.asarray([summary.encode("utf-8")], dtype=object)
        return _FakeResponse({"OUTPUT_TEXT": output_bytes})

    @staticmethod
    def _get_input(inputs: list[InferInput], name: str) -> InferInput | None:
        for item in inputs:
            if item.name == name:
                return item
        return None

    @staticmethod
    def _first_text(input_obj: InferInput | None) -> str | None:
        if input_obj is None or getattr(input_obj, "_data", None) is None:
            return None
        flattened = input_obj._data.reshape(-1)
        if flattened.size == 0:
            return None
        value = flattened[0]
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value).decode("utf-8", errors="replace")
        return str(value)

    @staticmethod
    def _first_int(input_obj: InferInput | None) -> int | None:
        if input_obj is None or getattr(input_obj, "_data", None) is None:
            return None
        flattened = input_obj._data.reshape(-1)
        if flattened.size == 0:
            return None
        try:
            return int(flattened[0])
        except Exception:
            return None

    @staticmethod
    def _first_float(input_obj: InferInput | None) -> float | None:
        if input_obj is None or getattr(input_obj, "_data", None) is None:
            return None
        flattened = input_obj._data.reshape(-1)
        if flattened.size == 0:
            return None
        try:
            return float(flattened[0])
        except Exception:
            return None
