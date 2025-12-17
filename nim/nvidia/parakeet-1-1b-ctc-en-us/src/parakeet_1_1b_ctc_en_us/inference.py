from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import numpy as np
from fastapi import HTTPException, status
from riva.client.proto import riva_asr_pb2 as rasr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tritonclient import http as triton_http
from tritonclient.utils import InferenceServerException

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
