from __future__ import annotations

import io
import json
import os
import wave
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import triton_python_backend_utils as pb_utils

try:
    import torch
    from transformers import AutoModelForCTC, AutoProcessor
except Exception as exc:  # pragma: no cover - surfaced by Triton
    raise RuntimeError(
        "Triton Python backend requires torch and transformers; ensure dependencies are installed."
    ) from exc

try:
    from deepmultilingualpunctuation import PunctuationModel
except Exception:
    PunctuationModel = None


LINEAR_PCM_ENCODING = 1


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


def _resample_audio(
    audio: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    """
    Resample audio to the requested rate using scipy when available.
    """
    if source_rate == target_rate:
        return audio
    try:
        from scipy.signal import resample_poly
    except Exception:
        original_indices = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
        target_length = max(
            1, int(audio.shape[0] * float(target_rate) / float(source_rate))
        )
        target_indices = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
        return np.interp(target_indices, original_indices, audio).astype(np.float32)
    return resample_poly(audio, target_rate, source_rate).astype(np.float32)


def _decode_wav(
    payload: bytes,
    target_sample_rate: int,
) -> Tuple[np.ndarray, float]:
    """
    Decode a WAV payload and resample to the target rate.
    """
    if not payload:
        raise ValueError("audio payload is empty")
    try:
        with wave.open(io.BytesIO(payload), "rb") as reader:
            source_rate = reader.getframerate()
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            frames = reader.readframes(reader.getnframes())
    except Exception as exc:
        raise ValueError("unable to parse WAV payload") from exc
    if not frames:
        raise ValueError("audio payload contained no samples")
    dtype = np.int16 if sample_width == 2 else np.int8
    data = np.frombuffer(frames, dtype=dtype)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    max_range = np.iinfo(dtype).max
    data = data.astype(np.float32) / float(max_range)
    data = _resample_audio(data, source_rate, target_sample_rate)
    duration_seconds = float(data.shape[0]) / float(target_sample_rate)
    return data, duration_seconds


def _decode_pcm(
    payload: bytes,
    source_rate: int,
    target_rate: int,
    channels: int,
) -> Tuple[np.ndarray, float]:
    """
    Decode raw PCM16 audio to mono float32 and resample.
    """
    if not payload:
        raise ValueError("audio payload is empty")
    if source_rate <= 0:
        raise ValueError("sample_rate_hertz must be provided for PCM audio")
    pcm = np.frombuffer(payload, dtype=np.int16)
    if pcm.size == 0:
        raise ValueError("audio payload contained no samples")
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    data = pcm.astype(np.float32) / 32768.0
    data = _resample_audio(data, source_rate, target_rate)
    duration_seconds = float(data.shape[0]) / float(target_rate)
    return data, duration_seconds


def _get_param(params: Dict[str, Dict[str, str]], key: str, default: str) -> str:
    """
    Retrieve a string parameter from the Triton model config parameters.
    """
    value = params.get(key, {}).get("string_value")
    return value if value is not None else default


def _round_robin_speakers(count: int, speakers: int) -> List[int]:
    """
    Assign speaker tags in a round-robin pattern.
    """
    if speakers <= 0:
        return [0 for _ in range(count)]
    return [((idx % speakers) + 1) for idx in range(count)]


class _PunctuationRestorer:
    """
    Optional punctuation restorer using deepmultilingualpunctuation.
    """

    def __init__(self, model_id: str):
        if PunctuationModel is None:
            raise RuntimeError("punctuation model dependency is missing")
        self._model = PunctuationModel(model_id)

    def restore(self, transcript: str) -> str:
        """
        Apply learned punctuation to the transcript.
        """
        cleaned = transcript.strip()
        if not cleaned:
            return transcript
        return self._model.restore_punctuation(cleaned)


class TritonPythonModel:
    """
    Triton Python backend implementing Parakeet CTC inference.
    """

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Load model assets and configuration at backend initialization time.
        """
        config = json.loads(args["model_config"])
        params = config.get("parameters", {})
        self._model_id = _get_param(params, "model_id", "nvidia/parakeet-ctc-1.1b")
        self._punctuation_model_id = _get_param(
            params, "punctuation_model_id", "kredor/punctuate-all"
        )
        self._target_sample_rate = int(
            _get_param(params, "target_sample_rate", "16000")
        )
        enable_mock_param = _get_param(params, "enable_mock_inference", "0")
        self._enable_mock = os.getenv("ENABLE_MOCK_INFERENCE", enable_mock_param) == "1"
        self._device = _select_device()
        self._dtype = _select_dtype(self._device)

        if self._enable_mock:
            self._processor = None
            self._model = None
            self._sample_rate = self._target_sample_rate
            self._time_offset_seconds = 1.0 / float(self._target_sample_rate)
            self._punctuation: Optional[_PunctuationRestorer] = None
            return

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForCTC.from_pretrained(
            self._model_id,
            torch_dtype=self._dtype,
        ).to(self._device)
        self._model.eval()
        sampling_rate = getattr(
            self._processor.feature_extractor, "sampling_rate", self._target_sample_rate
        )
        self._sample_rate = int(sampling_rate)
        logits_ratio = getattr(self._model.config, "inputs_to_logits_ratio", 1.0)
        self._time_offset_seconds = float(logits_ratio) / float(self._sample_rate)
        self._punctuation: Optional[_PunctuationRestorer] = None

    def execute(
        self, requests: Iterable[pb_utils.InferenceRequest]
    ) -> List[pb_utils.InferenceResponse]:
        """
        Handle one or more inference requests from Triton.
        """
        responses: List[pb_utils.InferenceResponse] = []
        for request in requests:
            try:
                audio, sample_rate, encoding, channels = self._parse_audio(request)
                (
                    enable_punctuation,
                    include_word_offsets,
                    enable_diarization,
                    max_speakers,
                ) = self._parse_options(request)
                transcript, offsets, processed = self._transcribe(
                    audio,
                    sample_rate,
                    encoding,
                    channels,
                    include_word_offsets,
                    enable_punctuation,
                    enable_diarization,
                    max_speakers,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor(
                                "TRANSCRIPT",
                                np.asarray([transcript.encode("utf-8")], dtype=object),
                            ),
                            pb_utils.Tensor(
                                "WORD_OFFSETS",
                                np.asarray(
                                    [json.dumps(offsets).encode("utf-8")],
                                    dtype=object,
                                ),
                            ),
                            pb_utils.Tensor(
                                "AUDIO_PROCESSED",
                                np.asarray([processed], dtype=np.float32),
                            ),
                        ]
                    )
                )
            except Exception as exc:  # pragma: no cover - surfaced by Triton
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))
                )
        return responses

    def _parse_audio(
        self, request: pb_utils.InferenceRequest
    ) -> Tuple[bytes, int, int, int]:
        """
        Decode audio input tensors and associated metadata.
        """
        audio_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO")
        if audio_tensor is None:
            raise ValueError("AUDIO input is required")
        audio_values = audio_tensor.as_numpy().reshape(-1)
        if audio_values.size == 0:
            raise ValueError("audio payload is empty")
        raw_audio = audio_values[0]
        if isinstance(raw_audio, str):
            audio_bytes = raw_audio.encode("utf-8")
        elif isinstance(raw_audio, (bytes, bytearray)):
            audio_bytes = bytes(raw_audio)
        else:
            raise ValueError("AUDIO input must be bytes")

        sample_rate = self._int_input(request, "SAMPLE_RATE", self._sample_rate)
        encoding = self._int_input(request, "ENCODING", 0)
        channels = self._int_input(request, "CHANNELS", 1)
        return audio_bytes, sample_rate, encoding, channels

    def _parse_options(
        self, request: pb_utils.InferenceRequest
    ) -> Tuple[bool, bool, bool, int]:
        """
        Parse optional inference flags from the request.
        """
        enable_punctuation = self._bool_input(request, "ENABLE_PUNCTUATION", True)
        include_word_offsets = self._bool_input(request, "ENABLE_WORD_OFFSETS", True)
        enable_diarization = self._bool_input(request, "ENABLE_DIARIZATION", False)
        max_speakers = self._int_input(request, "MAX_SPEAKER_COUNT", 1)
        return (
            enable_punctuation,
            include_word_offsets,
            enable_diarization,
            max_speakers,
        )

    def _int_input(
        self, request: pb_utils.InferenceRequest, name: str, default: int
    ) -> int:
        """
        Extract an integer scalar input, falling back to a default.
        """
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        values = tensor.as_numpy().reshape(-1)
        if values.size == 0:
            return default
        try:
            candidate = int(values[0])
            return candidate
        except Exception:
            return default

    def _bool_input(
        self, request: pb_utils.InferenceRequest, name: str, default: bool
    ) -> bool:
        """
        Extract a boolean flag from an INT32 input.
        """
        return bool(self._int_input(request, name, int(default)))

    def _transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        encoding: int,
        channels: int,
        include_word_offsets: bool,
        enable_punctuation: bool,
        enable_diarization: bool,
        max_speakers: int,
    ) -> Tuple[str, List[Dict[str, float]], float]:
        """
        Run the Parakeet CTC model (or mock) to produce a transcript and offsets.
        """
        audio, duration_seconds = self._decode_audio(
            audio_bytes, sample_rate, encoding, channels
        )
        if self._enable_mock:
            transcript = "mock transcript"
            offsets: List[Dict[str, float]] = []
            if include_word_offsets:
                offsets = [
                    {"word": "mock", "start": 0.0, "end": 0.5},
                    {"word": "transcript", "start": 0.5, "end": 1.2},
                ]
            if enable_diarization and offsets:
                speakers = _round_robin_speakers(len(offsets), max_speakers)
                for idx, speaker in enumerate(speakers):
                    offsets[idx]["speaker"] = speaker
            if enable_punctuation:
                transcript = self._punctuate(transcript)
            return transcript, offsets, duration_seconds

        assert self._processor is not None
        assert self._model is not None

        inputs = self._processor(
            audio,
            sampling_rate=self._sample_rate,
            return_tensors="pt",
            padding="longest",
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self._model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        decoded = self._processor.batch_decode(
            predicted_ids, output_word_offsets=include_word_offsets
        )
        transcript = decoded.text[0] if hasattr(decoded, "text") else decoded[0]
        offsets = self._word_offsets(decoded, include_word_offsets)
        if enable_diarization and offsets:
            speakers = _round_robin_speakers(len(offsets), max_speakers)
            for idx, speaker in enumerate(speakers):
                offsets[idx]["speaker"] = speaker
        if enable_punctuation:
            transcript = self._punctuate(transcript)
        return transcript, offsets, duration_seconds

    def _decode_audio(
        self, payload: bytes, sample_rate: int, encoding: int, channels: int
    ) -> Tuple[np.ndarray, float]:
        """
        Decode either WAV or PCM payloads into normalized audio.
        """
        effective_rate = sample_rate if sample_rate > 0 else self._sample_rate
        try:
            if encoding == LINEAR_PCM_ENCODING:
                return _decode_pcm(
                    payload, effective_rate, self._sample_rate, max(1, channels)
                )
            try:
                return _decode_wav(payload, self._sample_rate)
            except ValueError:
                return _decode_pcm(
                    payload, effective_rate, self._sample_rate, max(1, channels)
                )
        except Exception as exc:
            raise ValueError(f"failed to decode audio: {exc}") from exc

    def _word_offsets(
        self, decoded: Sequence[object], include_word_offsets: bool
    ) -> List[Dict[str, float]]:
        """
        Convert processor word offsets to second-based start/end values.
        """
        if not include_word_offsets:
            return []
        offsets: List[Dict[str, float]] = []
        decoded_offsets = getattr(decoded, "word_offsets", None)
        source: Sequence[Dict[str, float]] = []
        if decoded_offsets and len(decoded_offsets) > 0:
            source = decoded_offsets[0] or []
        for entry in source:
            word = entry.get("word")
            start_offset = entry.get("start_offset")
            end_offset = entry.get("end_offset")
            if word is None or start_offset is None or end_offset is None:
                continue
            offsets.append(
                {
                    "word": str(word),
                    "start": float(start_offset) * self._time_offset_seconds,
                    "end": float(end_offset) * self._time_offset_seconds,
                }
            )
        return offsets

    def _punctuate(self, transcript: str) -> str:
        """
        Apply punctuation using the configured model or a heuristic fallback.
        """
        cleaned = transcript.strip()
        if not cleaned:
            return transcript
        try:
            if self._punctuation is None and PunctuationModel is not None:
                self._punctuation = _PunctuationRestorer(self._punctuation_model_id)
            if self._punctuation is not None:
                return self._punctuation.restore(cleaned)
        except Exception:
            pass
        punctuated = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned
        if punctuated and punctuated[-1] not in {".", "!", "?"}:
            punctuated = f"{punctuated}."
        return punctuated
