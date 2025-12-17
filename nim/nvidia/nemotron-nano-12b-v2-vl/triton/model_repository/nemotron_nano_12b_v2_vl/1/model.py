from __future__ import annotations

import io
import json
import logging
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import triton_python_backend_utils as pb_utils  # type: ignore[reportMissingImports]
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

LOGGER = logging.getLogger("nemotron_nano_vlm_triton")
DEFAULT_SYSTEM_PROMPT = "/no_think"
DEFAULT_USER_PROMPT = "Caption the content of this image:"


def _get_param(params: Dict[str, Dict[str, str]], key: str, default: str) -> str:
    """
    Retrieve a string parameter from the Triton config parameters.
    """
    value = params.get(key, {}).get("string_value")
    return value if value is not None else default


def _select_device() -> str:
    """
    Choose the execution device based on CUDA availability.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def _select_dtype(device: str) -> torch.dtype:
    """
    Pick an appropriate dtype for the selected device.
    """
    if device == "cuda":
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _as_bytes(value: object) -> bytes:
    """
    Normalize a tensor element into bytes.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return str(value).encode("utf-8")


def _as_text(value: object, default: str) -> str:
    """
    Decode a tensor element into text with a fallback default.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        text = bytes(value).decode("utf-8", errors="replace")
    else:
        text = str(value)
    stripped = text.strip()
    return stripped if stripped else default


def _coerce_int(tensor: pb_utils.Tensor | None, default: int) -> int:
    """
    Extract an integer scalar from a tensor, falling back to default.
    """
    if tensor is None:
        return default
    values = tensor.as_numpy().reshape(-1)
    if values.size == 0:
        return default
    try:
        return int(values[0])
    except Exception:
        return default


def _coerce_float(tensor: pb_utils.Tensor | None, default: float) -> float:
    """
    Extract a float scalar from a tensor, falling back to default.
    """
    if tensor is None:
        return default
    values = tensor.as_numpy().reshape(-1)
    if values.size == 0:
        return default
    try:
        return float(values[0])
    except Exception:
        return default


def _first_value(tensor: pb_utils.Tensor | None) -> object | None:
    """
    Return the first element of a tensor when present.
    """
    if tensor is None:
        return None
    values = tensor.as_numpy().reshape(-1)
    if values.size == 0:
        return None
    return values[0]


class TritonPythonModel:
    """
    Triton Python backend implementing caption generation for Nemotron Nano VLM.
    """

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Load model assets at backend startup.
        """
        config = json.loads(args["model_config"])
        params = config.get("parameters", {})

        self._max_batch_size = int(config.get("max_batch_size", 0) or 0)
        model_id_param = _get_param(
            params,
            "model_id",
            "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        )
        default_tokens_param = _get_param(params, "default_max_tokens", "512")
        max_tokens_param = _get_param(params, "max_output_tokens", "1024")
        video_pruning_param = _get_param(params, "video_pruning_rate", "0.0")
        enable_mock_param = _get_param(params, "enable_mock_inference", "0")

        self._model_id = os.getenv("MODEL_ID", model_id_param)
        self._default_max_tokens = int(
            os.getenv("DEFAULT_MAX_TOKENS", default_tokens_param)
        )
        self._max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", max_tokens_param))
        self._video_pruning_rate = float(
            os.getenv("VIDEO_PRUNING_RATE", video_pruning_param)
        )
        self._enable_mock = os.getenv("ENABLE_MOCK_INFERENCE", enable_mock_param) == "1"
        self._system_prompt = os.getenv(
            "VLM_CAPTION_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT
        )
        self._user_prompt = os.getenv("VLM_CAPTION_PROMPT", DEFAULT_USER_PROMPT)

        self._device = _select_device()
        self._dtype = _select_dtype(self._device)
        self._tokenizer: AutoTokenizer | None = None
        self._processor: AutoProcessor | None = None
        self._model: AutoModelForCausalLM | None = None

        if self._enable_mock:
            LOGGER.info("Mock inference enabled; skipping model load.")
            return

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, trust_remote_code=True
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._processor = AutoProcessor.from_pretrained(
            self._model_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            trust_remote_code=True,
            torch_dtype=self._dtype,
        )
        self._model.to(self._device)
        self._model.eval()

        if hasattr(self._model, "video_pruning_rate"):
            try:
                self._model.video_pruning_rate = max(self._video_pruning_rate, 0.0)
            except Exception:
                LOGGER.debug(
                    "Model does not expose writable video_pruning_rate; skipping"
                )

        LOGGER.info(
            "Loaded %s to %s (dtype=%s)",
            self._model_id,
            self._device,
            self._dtype,
        )

    def execute(
        self, requests: Iterable[pb_utils.InferenceRequest]
    ) -> List[pb_utils.InferenceResponse]:
        """
        Handle one or more inference requests from Triton.
        """
        responses: List[pb_utils.InferenceResponse] = []
        for request in requests:
            try:
                (
                    system_prompt,
                    user_prompt,
                    image_bytes,
                    max_new_tokens,
                    temperature,
                    top_p,
                ) = self._parse_request(request)

                caption = (
                    self._mock_caption(user_prompt)
                    if self._enable_mock
                    else self._generate_caption(
                        system_prompt,
                        user_prompt,
                        image_bytes,
                        max_new_tokens,
                        temperature,
                        top_p,
                    )
                )
                output = pb_utils.Tensor(
                    "OUTPUT_TEXT",
                    np.asarray([caption.encode("utf-8")], dtype=object),
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[output]))
            except Exception as exc:  # pragma: no cover - surfaced to Triton
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))
                )
        return responses

    def _parse_request(
        self, request: pb_utils.InferenceRequest
    ) -> Tuple[str, str, bytes, int, float, float]:
        """
        Decode and validate incoming request tensors.
        """
        image_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_BYTES")
        system_tensor = pb_utils.get_input_tensor_by_name(request, "SYSTEM_PROMPT")
        user_tensor = pb_utils.get_input_tensor_by_name(request, "USER_PROMPT")
        max_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "MAX_NEW_TOKENS")
        temperature_tensor = pb_utils.get_input_tensor_by_name(request, "TEMPERATURE")
        top_p_tensor = pb_utils.get_input_tensor_by_name(request, "TOP_P")

        if image_tensor is None:
            raise ValueError("IMAGE_BYTES input is required")

        image_values = image_tensor.as_numpy().reshape(-1)
        if image_values.size == 0:
            raise ValueError("IMAGE_BYTES input must include at least one item")
        if self._max_batch_size > 0 and image_values.size > self._max_batch_size:
            raise ValueError(
                f"Batch size {image_values.size} exceeds limit {self._max_batch_size}"
            )
        image_bytes = _as_bytes(image_values[0])

        system_prompt = _as_text(_first_value(system_tensor), self._system_prompt)
        user_prompt = _as_text(_first_value(user_tensor), self._user_prompt)

        requested_tokens = _coerce_int(max_tokens_tensor, self._default_max_tokens)
        max_new_tokens = max(1, min(requested_tokens, self._max_output_tokens))
        temperature = max(0.0, _coerce_float(temperature_tensor, 1.0))
        top_p = _coerce_float(top_p_tensor, 1.0)
        if top_p <= 0.0:
            top_p = 1.0
        return (
            system_prompt,
            user_prompt,
            image_bytes,
            max_new_tokens,
            temperature,
            top_p,
        )

    def _generate_caption(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """
        Run caption generation synchronously.
        """
        if self._model is None or self._tokenizer is None or self._processor is None:
            raise RuntimeError("Model assets are not initialized")

        image = self._decode_image(image_bytes)
        inputs, prompt_length = self._prepare_inputs(system_prompt, user_prompt, image)
        generation_kwargs: Dict[str, object] = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pixel_values": inputs.pixel_values,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "temperature": temperature if temperature > 0.0 else None,
            "top_p": top_p,
            "eos_token_id": self._tokenizer.eos_token_id,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        generation_kwargs = {
            k: v for k, v in generation_kwargs.items() if v is not None
        }

        with torch.inference_mode():
            generated_ids = self._model.generate(**generation_kwargs)

        generated_sequence = generated_ids[0][prompt_length:]
        caption = self._tokenizer.decode(
            generated_sequence, skip_special_tokens=True
        ).strip()
        return caption if caption else self._user_prompt

    def _prepare_inputs(
        self, system_prompt: str, user_prompt: str, image: Image.Image
    ) -> Tuple[object, int]:
        """
        Build tokenized model inputs and track prompt length.
        """
        if self._tokenizer is None or self._processor is None:
            raise RuntimeError("Tokenizer or processor is missing")
        messages_for_template: List[Dict[str, object]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ""},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            messages_for_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._device)
        sequence_length = int(inputs.input_ids.shape[1])
        return inputs, sequence_length

    @staticmethod
    def _decode_image(image_bytes: bytes) -> Image.Image:
        """
        Decode raw image bytes into an RGB PIL image.
        """
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.mode != "RGB":
                return image.convert("RGB")
            return image.copy()

    @staticmethod
    def _mock_caption(user_prompt: str) -> str:
        """
        Produce a deterministic caption in mock mode.
        """
        snippet = user_prompt.strip() or DEFAULT_USER_PROMPT
        return f"[mock] Caption for image with prompt: {snippet}"
