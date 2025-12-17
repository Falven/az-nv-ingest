from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeAlias

import numpy as np
from fastapi import HTTPException, status
from oim_common.audio import pcm16_to_float32
from riva.client.proto import riva_asr_pb2 as rasr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tritonclient import http as triton_http
from tritonclient.utils import InferenceServerException

from .models import EndpointingOptions, RecognitionOptions, StreamingState
from .settings import ServiceSettings

LOGGER = logging.getLogger(__name__)

AUDIO_INPUT_NAME = "AUDIO"
SAMPLE_RATE_INPUT_NAME = "SAMPLE_RATE"
ENCODING_INPUT_NAME = "ENCODING"
CHANNELS_INPUT_NAME = "CHANNELS"
ENABLE_PUNCTUATION_INPUT_NAME = "ENABLE_PUNCTUATION"
ENABLE_WORD_OFFSETS_INPUT_NAME = "ENABLE_WORD_OFFSETS"
ENABLE_DIARIZATION_INPUT_NAME = "ENABLE_DIARIZATION"
MAX_SPEAKER_COUNT_INPUT_NAME = "MAX_SPEAKER_COUNT"

TRANSCRIPT_OUTPUT_NAME = "TRANSCRIPT"
WORD_OFFSETS_OUTPUT_NAME = "WORD_OFFSETS"
AUDIO_PROCESSED_OUTPUT_NAME = "AUDIO_PROCESSED"


class TritonASRClient:
    """
    Thin wrapper around Triton HTTP client for ASR inference.
    """

    def __init__(self, settings: ServiceSettings):
        self._settings = settings
        self._client = triton_http.InferenceServerClient(
            url=settings.triton_http_endpoint,
            verbose=bool(settings.log_verbose),
            connection_timeout=settings.triton_timeout,
            network_timeout=settings.triton_timeout,
        )
        self._model_name = settings.triton_model_name

    def is_ready(self) -> bool:
        """
        Report readiness based on Triton server and model status.
        """
        try:
            return bool(
                self._client.is_server_ready()
                and self._client.is_model_ready(model_name=self._model_name)
            )
        except Exception:
            return False

    def is_live(self) -> bool:
        """
        Report liveness from the Triton server.
        """
        try:
            return bool(self._client.is_server_live())
        except Exception:
            return False

    def model_metadata(self) -> Dict[str, Any]:
        """
        Fetch model metadata for Triton v2 endpoints.
        """
        return self._client.get_model_metadata(model_name=self._model_name)

    def model_config(self) -> Dict[str, Any]:
        """
        Fetch model config for Triton v2 endpoints.
        """
        return self._client.get_model_config(model_name=self._model_name)

    def repository_index(self) -> List[Dict[str, Any]]:
        """
        Return the model repository index.
        """
        return self._client.get_model_repository_index()

    def transcribe(
        self,
        audio_bytes: bytes,
        config: rasr.RecognitionConfig,
    ) -> rasr.SpeechRecognitionResult:
        """
        Execute a transcription request via Triton.
        """
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="audio payload is empty"
            )
        include_word_offsets, enable_punctuation, enable_diarization, max_speakers = (
            self._resolve_options(config)
        )
        inputs = self._build_inputs(
            audio_bytes,
            config,
            enable_punctuation,
            include_word_offsets,
            enable_diarization,
            max_speakers,
        )
        outputs = [
            triton_http.InferRequestedOutput(TRANSCRIPT_OUTPUT_NAME),
            triton_http.InferRequestedOutput(WORD_OFFSETS_OUTPUT_NAME),
            triton_http.InferRequestedOutput(AUDIO_PROCESSED_OUTPUT_NAME),
        ]
        response = self._infer(inputs, outputs)
        return self._to_result(response, include_word_offsets)

    @retry(
        retry=retry_if_exception_type(InferenceServerException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
        reraise=True,
    )
    def _infer(
        self,
        inputs: List[triton_http.InferInput],
        outputs: List[triton_http.InferRequestedOutput],
    ) -> triton_http.InferResult:
        """
        Issue an inference request to Triton.
        """
        return self._client.infer(
            model_name=self._model_name,
            inputs=inputs,
            outputs=outputs,
            timeout=self._settings.triton_timeout,
        )

    def _build_inputs(
        self,
        audio_bytes: bytes,
        config: rasr.RecognitionConfig,
        enable_punctuation: bool,
        include_word_offsets: bool,
        enable_diarization: bool,
        max_speakers: int,
    ) -> List[triton_http.InferInput]:
        """
        Build Triton HTTP inputs from the gRPC RecognitionConfig and payload.
        """
        audio_input = triton_http.InferInput(AUDIO_INPUT_NAME, [1], "BYTES")
        audio_input.set_data_from_numpy(np.asarray([audio_bytes], dtype=np.object_))

        sample_rate_value = (
            int(config.sample_rate_hertz) if config.sample_rate_hertz > 0 else 0
        )
        sample_rate_input = triton_http.InferInput(SAMPLE_RATE_INPUT_NAME, [1], "INT32")
        sample_rate_input.set_data_from_numpy(
            np.asarray([sample_rate_value], dtype=np.int32)
        )

        encoding_input = triton_http.InferInput(ENCODING_INPUT_NAME, [1], "INT32")
        encoding_input.set_data_from_numpy(
            np.asarray([int(config.encoding)], dtype=np.int32)
        )

        channels_input = triton_http.InferInput(CHANNELS_INPUT_NAME, [1], "INT32")
        channels_input.set_data_from_numpy(
            np.asarray([max(int(config.audio_channel_count or 1), 1)], dtype=np.int32)
        )

        punctuation_input = triton_http.InferInput(
            ENABLE_PUNCTUATION_INPUT_NAME, [1], "INT32"
        )
        punctuation_input.set_data_from_numpy(
            np.asarray([int(enable_punctuation)], dtype=np.int32)
        )

        word_offsets_input = triton_http.InferInput(
            ENABLE_WORD_OFFSETS_INPUT_NAME, [1], "INT32"
        )
        word_offsets_input.set_data_from_numpy(
            np.asarray([int(include_word_offsets)], dtype=np.int32)
        )

        diarization_input = triton_http.InferInput(
            ENABLE_DIARIZATION_INPUT_NAME, [1], "INT32"
        )
        diarization_input.set_data_from_numpy(
            np.asarray([int(enable_diarization)], dtype=np.int32)
        )

        max_speakers_input = triton_http.InferInput(
            MAX_SPEAKER_COUNT_INPUT_NAME, [1], "INT32"
        )
        max_speakers_input.set_data_from_numpy(
            np.asarray([max_speakers], dtype=np.int32)
        )

        return [
            audio_input,
            sample_rate_input,
            encoding_input,
            channels_input,
            punctuation_input,
            word_offsets_input,
            diarization_input,
            max_speakers_input,
        ]

    def _resolve_options(
        self, config: rasr.RecognitionConfig
    ) -> tuple[bool, bool, bool, int]:
        """
        Normalize RecognitionConfig flags for downstream request building.
        """
        include_word_offsets = bool(config.enable_word_time_offsets)
        enable_punctuation = bool(
            config.enable_automatic_punctuation
            or self._settings.enable_automatic_punctuation
        )
        enable_diarization = False
        max_speakers = 1
        if config.HasField("diarization_config"):
            enable_diarization = bool(
                config.diarization_config.enable_speaker_diarization
            )
            try:
                max_speakers = max(1, int(config.diarization_config.max_speaker_count))
            except Exception:
                max_speakers = 1
        return (
            include_word_offsets,
            enable_punctuation,
            enable_diarization,
            max_speakers,
        )

    def _to_result(
        self, response: triton_http.InferResult, include_word_offsets: bool
    ) -> rasr.SpeechRecognitionResult:
        """
        Convert Triton outputs to a Riva SpeechRecognitionResult.
        """
        transcript_array = response.as_numpy(TRANSCRIPT_OUTPUT_NAME)
        if transcript_array is None or transcript_array.size == 0:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Triton response missing transcript",
            )
        transcript_value = transcript_array.reshape(-1)[0]
        transcript = (
            transcript_value.decode("utf-8")
            if isinstance(transcript_value, (bytes, bytearray))
            else str(transcript_value)
        )

        offsets_payload = b"[]"
        offsets_array = response.as_numpy(WORD_OFFSETS_OUTPUT_NAME)
        if offsets_array is not None and offsets_array.size > 0:
            offsets_candidate = offsets_array.reshape(-1)[0]
            if isinstance(offsets_candidate, (bytes, bytearray)):
                offsets_payload = offsets_candidate
            else:
                offsets_payload = str(offsets_candidate).encode("utf-8")

        audio_processed = 0.0
        processed_array = response.as_numpy(AUDIO_PROCESSED_OUTPUT_NAME)
        if processed_array is not None and processed_array.size > 0:
            try:
                audio_processed = float(processed_array.reshape(-1)[0])
            except Exception:
                audio_processed = 0.0

        result = rasr.SpeechRecognitionResult()
        alternative = result.alternatives.add()
        alternative.transcript = transcript
        alternative.confidence = 0.0
        if include_word_offsets:
            alternative.words.extend(self._word_infos(offsets_payload))
        result.audio_processed = audio_processed
        return result

    def _word_infos(self, payload: bytes) -> List[rasr.WordInfo]:
        """
        Parse JSON word offsets into WordInfo messages.
        """
        try:
            decoded = payload.decode("utf-8")
            parsed = json.loads(decoded) if decoded else []
        except Exception:
            LOGGER.warning("Failed to parse word offsets payload", exc_info=True)
            return []
        infos: List[rasr.WordInfo] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            word = entry.get("word")
            start = entry.get("start")
            end = entry.get("end")
            if word is None or start is None or end is None:
                continue
            info = rasr.WordInfo()
            info.word = str(word)
            try:
                info.start_time = int(max(float(start), 0.0) * 1000.0)
                info.end_time = int(max(float(end), 0.0) * 1000.0)
            except Exception:
                continue
            speaker_tag = entry.get("speaker")
            if speaker_tag is not None:
                try:
                    info.speaker_tag = int(speaker_tag)
                except Exception:
                    info.speaker_tag = 0
            infos.append(info)
        return infos

    def validate_model_name(self, model_name: str) -> None:
        """
        Ensure the provided model name matches the configured Triton model.
        """
        if model_name != self._model_name:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown model '{model_name}'",
            )


class VadDetector:
    """
    Minimal VAD stub for streaming endpointing logic.
    """

    def __init__(self, frame_ms: int):
        self.frame_ms = max(1, int(frame_ms))

    def probabilities(
        self, audio: np.ndarray, sample_rate: int
    ) -> Tuple[List[float], float]:
        """
        Return uniform VAD probabilities for the provided audio.
        """
        frame_duration = float(self.frame_ms) / 1000.0
        if audio.size == 0 or sample_rate <= 0:
            return [], frame_duration
        total_seconds = float(audio.shape[0]) / float(sample_rate)
        frames = max(1, int(math.ceil(total_seconds / frame_duration)))
        return [1.0 for _ in range(frames)], frame_duration


class ParakeetASRBackend:
    """
    Lightweight ASR backend that can operate in mock mode or delegate to Triton.
    """

    def __init__(self, settings: ServiceSettings):
        self._settings = settings
        self.sample_rate = 16000
        self.time_offset_seconds = 1.0 / float(self.sample_rate)
        self.vad_detector = VadDetector(settings.vad_frame_ms)
        self._use_mock = bool(
            settings.enable_mock_inference or os.getenv("ENABLE_DRY_RUN") == "1"
        )
        self._triton: TritonASRClient | None = None
        if not self._use_mock:
            try:
                self._triton = TritonASRClient(settings)
            except Exception:
                LOGGER.warning(
                    "Falling back to mock inference; Triton client unavailable."
                )
                self._use_mock = True
        self._last_config: rasr.RecognitionConfig | None = None

    def decode_bytes(
        self, payload: bytes, config: rasr.RecognitionConfig
    ) -> Tuple[np.ndarray, float]:
        """
        Decode raw audio bytes into normalized float32 samples.
        """
        self._last_config = config
        if not payload:
            raise ValueError("audio payload is empty")
        audio_values = pcm16_to_float32(payload)
        if audio_values.size == 0:
            raise ValueError("audio payload contained no samples")
        normalized = audio_values
        sample_rate = (
            int(config.sample_rate_hertz)
            if config.sample_rate_hertz > 0
            else self.sample_rate
        )
        duration_seconds = (
            float(normalized.shape[0]) / float(sample_rate) if sample_rate > 0 else 0.0
        )
        return normalized, duration_seconds

    def _pcm_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        clipped = np.clip(audio, -1.0, 1.0)
        scaled = (clipped * 32767.0).astype(np.int16)
        return scaled.tobytes()

    def recognize_from_audio(
        self,
        audio: np.ndarray,
        include_word_offsets: bool,
        enable_diarization: bool,
        enable_punctuation: bool,
        max_speaker_count: int,
        offset_seconds: float,
        original_duration_seconds: float,
    ) -> rasr.SpeechRecognitionResult:
        """
        Run offline recognition using Triton when available; otherwise return a mock response.
        """
        config = self._last_config or rasr.RecognitionConfig()
        sample_rate = (
            int(config.sample_rate_hertz)
            if config.sample_rate_hertz > 0
            else self.sample_rate
        )
        if not self._use_mock and self._triton is not None:
            config.enable_word_time_offsets = bool(include_word_offsets)
            config.enable_automatic_punctuation = bool(enable_punctuation)
            if enable_diarization:
                config.diarization_config.enable_speaker_diarization = True
                config.diarization_config.max_speaker_count = max_speaker_count
            if config.sample_rate_hertz <= 0:
                config.sample_rate_hertz = sample_rate
            result = self._triton.transcribe(
                self._pcm_bytes(audio, sample_rate),
                config,
            )
            if not result.audio_processed:
                result.audio_processed = original_duration_seconds
            return result
        transcript = "mock transcript"
        offsets: List[Dict[str, float]] = []
        if include_word_offsets:
            offsets = [
                {"word": "mock", "start": offset_seconds, "end": offset_seconds + 0.5},
                {
                    "word": "transcript",
                    "start": offset_seconds + 0.5,
                    "end": offset_seconds + 1.0,
                },
            ]
        response = rasr.SpeechRecognitionResult()
        alternative = response.alternatives.add()
        alternative.transcript = (
            transcript if enable_punctuation else transcript.rstrip(".")
        )
        alternative.confidence = 0.0
        if include_word_offsets:
            alternative.words.extend(self.word_infos(offsets, None))
        response.audio_processed = (
            original_duration_seconds
            if original_duration_seconds > 0
            else float(audio.shape[0]) / float(sample_rate or 1)
        )
        return response

    def compute_logits(self, audio: np.ndarray) -> np.ndarray:
        """
        Produce deterministic mock logits for streaming responses.
        """
        if audio.size == 0:
            return np.zeros((1, 1), dtype=np.float32)
        flattened = audio.reshape(-1)
        return flattened.astype(np.float32).reshape(-1, 1)

    def decode_logits(
        self, logits: np.ndarray, include_word_offsets: bool
    ) -> Tuple[str, List[Dict[str, float]]]:
        """
        Convert mock logits into a transcript and optional offsets.
        """
        if logits.size == 0:
            return "", []
        transcript = "mock transcript"
        offsets: List[Dict[str, float]] = []
        if include_word_offsets:
            step = 0.5
            offsets = [
                {"word": "mock", "start": 0.0, "end": step},
                {"word": "transcript", "start": step, "end": 2 * step},
            ]
        return transcript, offsets

    def punctuate(self, transcript: str) -> str:
        """
        Apply lightweight punctuation when requested.
        """
        cleaned = transcript.strip()
        if not cleaned:
            return transcript
        if cleaned.endswith((".", "!", "?")):
            return cleaned
        return f"{cleaned}."

    def word_infos(
        self,
        offsets: Sequence[Dict[str, float]],
        speaker_assignments: Optional[Dict[int, int]],
    ) -> List[rasr.WordInfo]:
        """
        Convert offset dictionaries into Riva WordInfo messages.
        """
        infos: List[rasr.WordInfo] = []
        for index, entry in enumerate(offsets):
            word = str(entry.get("word", "")).strip()
            if not word:
                continue
            start = float(entry.get("start", 0.0))
            end = float(entry.get("end", start))
            info = rasr.WordInfo()
            info.word = word
            info.start_time = int(max(start, 0.0) * 1000.0)
            info.end_time = int(max(end, 0.0) * 1000.0)
            if speaker_assignments is not None:
                speaker = speaker_assignments.get(index)
                if speaker is not None:
                    try:
                        info.speaker_tag = int(speaker)
                    except Exception:
                        info.speaker_tag = 0
            infos.append(info)
        return infos


ASRBackend: TypeAlias = ParakeetASRBackend


def config_to_options(
    config: rasr.RecognitionConfig, settings: ServiceSettings
) -> RecognitionOptions:
    """
    Normalize a RecognitionConfig into RecognitionOptions with defaults applied.
    """
    sample_rate = (
        int(config.sample_rate_hertz) if config.sample_rate_hertz > 0 else 16000
    )
    language_code = config.language_code or settings.default_language_code
    include_word_offsets = bool(config.enable_word_time_offsets)
    enable_punctuation = bool(
        config.enable_automatic_punctuation
        or settings.enable_automatic_punctuation
        or settings.default_punctuation
    )
    enable_diarization = bool(
        config.HasField("diarization_config")
        and config.diarization_config.enable_speaker_diarization
    )
    max_speaker_count = (
        int(config.diarization_config.max_speaker_count)
        if config.HasField("diarization_config")
        and config.diarization_config.max_speaker_count > 0
        else settings.max_speaker_count
    )
    endpointing = EndpointingOptions(
        start_history_seconds=settings.endpoint_start_history_ms / 1000.0,
        start_threshold=settings.endpoint_start_threshold,
        stop_history_seconds=settings.endpoint_stop_history_ms / 1000.0,
        stop_threshold=settings.endpoint_stop_threshold,
        eou_history_seconds=settings.endpoint_eou_history_ms / 1000.0,
        eou_threshold=settings.endpoint_eou_threshold,
    )
    return RecognitionOptions(
        sample_rate=sample_rate,
        language_code=language_code,
        include_word_time_offsets=include_word_offsets,
        enable_punctuation=enable_punctuation,
        enable_diarization=enable_diarization,
        interim_results=False,
        channel_count=int(config.audio_channel_count or 1),
        max_speaker_count=max_speaker_count,
        endpointing=endpointing,
    )


def trim_audio_with_vad(
    audio: np.ndarray,
    sample_rate: int,
    vad_detector: VadDetector,
    endpointing: EndpointingOptions,
) -> Tuple[np.ndarray, float]:
    """
    Placeholder VAD trimming; returns the input audio unchanged.
    """
    if audio.size == 0:
        raise ValueError("audio payload is empty")
    _ = (sample_rate, vad_detector, endpointing)
    return audio, 0.0


def pipeline_vad_probabilities(state: StreamingState) -> List[float]:
    """
    Surface recent VAD probabilities collected during streaming.
    """
    if not state.vad_history:
        return []
    history = list(state.vad_history)
    return [prob for _, prob in history[-5:]]


def update_endpointing_state(
    state: StreamingState,
    endpointing: EndpointingOptions,
    vad_probs: Sequence[float],
    frame_duration: float,
) -> None:
    """
    Update streaming endpointing flags using simple thresholds.
    """
    if vad_probs:
        state.vad_history.extend(
            [
                (state.total_audio_seconds + (index * frame_duration), probability)
                for index, probability in enumerate(vad_probs)
            ]
        )
        if any(probability >= endpointing.start_threshold for probability in vad_probs):
            state.speech_started = True
            state.last_silence_time = None
        if state.speech_started and all(
            probability < endpointing.stop_threshold for probability in vad_probs
        ):
            state.last_silence_time = state.total_audio_seconds
    if (
        state.speech_started
        and state.total_audio_seconds >= endpointing.max_history_seconds
    ):
        state.endpoint_triggered = True


def estimate_speaker_segments(
    audio: np.ndarray, sample_rate: int, max_speakers: int
) -> List[Tuple[float, float]]:
    """
    Produce simple, evenly split speaker segments for diarization.
    """
    if audio.size == 0 or sample_rate <= 0 or max_speakers <= 0:
        return []
    duration_seconds = float(audio.shape[0]) / float(sample_rate)
    segment_length = duration_seconds / float(max_speakers)
    segments: List[Tuple[float, float]] = []
    start = 0.0
    for index in range(max_speakers):
        end = duration_seconds if index == max_speakers - 1 else start + segment_length
        segments.append((start, end))
        start = end
    return segments


def assign_speakers(
    offsets: Sequence[Dict[str, float]],
    segments: Sequence[Tuple[float, float]],
    time_offset_seconds: float,
) -> Dict[int, int]:
    """
    Map word offsets to speaker identifiers based on simple segment boundaries.
    """
    assignments: Dict[int, int] = {}
    if not offsets or not segments:
        return assignments
    _ = time_offset_seconds
    for index, entry in enumerate(offsets):
        start = float(entry.get("start", 0.0))
        end = float(entry.get("end", start))
        midpoint = (start + end) / 2.0
        for speaker_index, (segment_start, segment_end) in enumerate(segments):
            if segment_start <= midpoint <= segment_end:
                assignments[index] = speaker_index + 1
                break
        if index not in assignments:
            assignments[index] = (index % len(segments)) + 1
    return assignments


def create_asr_backend(settings: ServiceSettings) -> ASRBackend:
    """
    Factory for the ASR backend, preferring real inference unless mock is requested.
    """
    return ParakeetASRBackend(settings)
