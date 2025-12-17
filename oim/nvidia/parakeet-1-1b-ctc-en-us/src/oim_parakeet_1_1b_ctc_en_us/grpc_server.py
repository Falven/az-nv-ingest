from __future__ import annotations

import logging
import threading
import time
from concurrent import futures
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional, Sequence

import grpc
from opentelemetry import trace
from riva.client.proto import riva_asr_pb2 as rasr
from riva.client.proto import riva_asr_pb2_grpc as rasr_grpc
from tritonclient.grpc import model_config_pb2, service_pb2, service_pb2_grpc

from oim_common.auth import AuthInterceptor, AuthValidator
from .inference import (
    ASRBackend,
    assign_speakers,
    config_to_options,
    estimate_speaker_segments,
    pipeline_vad_probabilities,
    trim_audio_with_vad,
    update_endpointing_state,
)
from .metrics import RequestMetrics
from .models import StreamingState, TritonModelConfig
from .settings import ServiceSettings

LOGGER = logging.getLogger("parakeet-grpc")


class BusyError(RuntimeError):
    """
    Raised when concurrency limits are exceeded.
    """


class RequestLimiter:
    """
    Concurrency guard for general and streaming requests.
    """

    def __init__(self, max_concurrent_requests: int, max_streaming_sessions: int):
        self.general = threading.BoundedSemaphore(value=max(1, max_concurrent_requests))
        self.streaming = threading.BoundedSemaphore(
            value=max(1, max_streaming_sessions)
        )

    @contextmanager
    def limit(self, semaphore: threading.Semaphore) -> Iterator[None]:
        """
        Acquire the provided semaphore non-blocking and release afterwards.
        """
        acquired = semaphore.acquire(blocking=False)
        if not acquired:
            raise BusyError("server is busy")
        try:
            yield
        finally:
            semaphore.release()


def triton_model_metadata(
    config: TritonModelConfig,
) -> service_pb2.ModelMetadataResponse:
    response = service_pb2.ModelMetadataResponse()
    response.name = config.name
    response.platform = "riva-asr"
    response.versions.extend([config.version])
    tensor_input = response.inputs.add()
    tensor_input.name = "AUDIO"
    tensor_input.datatype = "BYTES"
    tensor_input.shape.extend([-1])
    tensor_output = response.outputs.add()
    tensor_output.name = "TRANSCRIPTS"
    tensor_output.datatype = "BYTES"
    tensor_output.shape.extend([-1])
    return response


def triton_model_config_response(
    config: TritonModelConfig,
    settings: ServiceSettings,
) -> model_config_pb2.ModelConfigResponse:
    response = model_config_pb2.ModelConfigResponse()
    response.config.name = config.name
    response.config.platform = "riva-asr"
    response.config.max_batch_size = max(1, config.max_batch_size)
    model_input = response.config.input.add()
    model_input.name = "AUDIO"
    model_input.data_type = model_config_pb2.TYPE_BYTES
    model_input.dims.extend([-1])
    model_output = response.config.output.add()
    model_output.name = "TRANSCRIPTS"
    model_output.data_type = model_config_pb2.TYPE_BYTES
    model_output.dims.extend([-1])
    response.config.parameters["language_code"].string_value = config.language_code
    response.config.parameters["sample_rate"].string_value = str(config.sample_rate)
    response.config.parameters["version"].string_value = config.version
    response.config.parameters["supports_punctuation"].string_value = "true"
    response.config.parameters["supports_diarization"].string_value = "true"
    response.config.parameters["max_concurrent_requests"].string_value = str(
        settings.max_concurrent_requests
    )
    response.config.parameters["max_streaming_sessions"].string_value = str(
        settings.max_streaming_sessions
    )
    response.config.parameters["endpoint_start_history_ms"].string_value = str(
        settings.endpoint_start_history_ms
    )
    response.config.parameters["endpoint_start_threshold"].string_value = str(
        settings.endpoint_start_threshold
    )
    response.config.parameters["endpoint_stop_history_ms"].string_value = str(
        settings.endpoint_stop_history_ms
    )
    response.config.parameters["endpoint_stop_threshold"].string_value = str(
        settings.endpoint_stop_threshold
    )
    response.config.parameters["endpoint_stop_history_eou_ms"].string_value = str(
        settings.endpoint_eou_history_ms
    )
    response.config.parameters["endpoint_stop_threshold_eou"].string_value = str(
        settings.endpoint_eou_threshold
    )
    response.config.parameters["vad_frame_ms"].string_value = str(settings.vad_frame_ms)
    return response


def triton_repository_index_response(
    config: TritonModelConfig, loaded: bool
) -> service_pb2.RepositoryIndexResponse:
    response = service_pb2.RepositoryIndexResponse()
    model_index = response.models.add()
    model_index.name = config.name
    model_index.version = config.version
    model_index.state = "READY" if loaded else "UNAVAILABLE"
    model_index.reason = "" if loaded else "Model is unloaded"
    return response


def triton_server_metadata_response(
    settings: ServiceSettings,
) -> service_pb2.ServerMetadataResponse:
    response = service_pb2.ServerMetadataResponse()
    response.name = settings.model_name
    response.version = settings.model_version
    response.extensions.extend(["classification", "sequence", "riva-asr"])
    return response


def triton_model_ready_response(loaded: bool) -> service_pb2.ModelReadyResponse:
    return service_pb2.ModelReadyResponse(ready=loaded)


def triton_server_ready_response(loaded: bool) -> service_pb2.ServerReadyResponse:
    return service_pb2.ServerReadyResponse(ready=loaded)


def triton_server_live_response() -> service_pb2.ServerLiveResponse:
    return service_pb2.ServerLiveResponse(live=True)


def triton_model_statistics_response(
    config: TritonModelConfig, loaded: bool
) -> service_pb2.ModelStatisticsResponse:
    response = service_pb2.ModelStatisticsResponse()
    if loaded:
        stats = response.model_stats.add()
        stats.name = config.name
        stats.version = config.version
    return response


class TritonControlService(service_pb2_grpc.GRPCInferenceServiceServicer):
    """
    Minimal Triton control-plane service for compatibility with discovery clients.
    """

    def __init__(self, config: TritonModelConfig, settings: ServiceSettings):
        self.config = config
        self.settings = settings
        self.loaded = True

    def _validate_model(self, name: Optional[str], context) -> None:
        if name and name not in {self.config.name}:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Model {name} not found")

    def ServerLive(self, request, context):  # noqa: N802
        return triton_server_live_response()

    def ServerReady(self, request, context):  # noqa: N802
        return triton_server_ready_response(self.loaded)

    def ServerMetadata(self, request, context):  # noqa: N802
        return triton_server_metadata_response(self.settings)

    def ModelReady(self, request, context):  # noqa: N802
        self._validate_model(request.name, context)
        return triton_model_ready_response(self.loaded)

    def ModelMetadata(self, request, context):  # noqa: N802
        self._validate_model(request.name, context)
        return triton_model_metadata(self.config)

    def ModelConfig(self, request, context):  # noqa: N802
        self._validate_model(request.name, context)
        return triton_model_config_response(self.config, self.settings)

    def RepositoryIndex(self, request, context):  # noqa: N802
        return triton_repository_index_response(self.config, self.loaded)

    def RepositoryModelLoad(self, request, context):  # noqa: N802
        self.loaded = True
        return service_pb2.RepositoryModelLoadResponse()

    def RepositoryModelUnload(self, request, context):  # noqa: N802
        self.loaded = False
        return service_pb2.RepositoryModelUnloadResponse()

    def ModelStatistics(self, request, context):  # noqa: N802
        self._validate_model(request.name, context)
        return triton_model_statistics_response(self.config, self.loaded)


class ParakeetServicer(rasr_grpc.RivaSpeechRecognitionServicer):
    """
    Implements the Riva SpeechRecognition service using the configured ASR backend.
    """

    def __init__(
        self,
        backend: ASRBackend,
        settings: ServiceSettings,
        metrics: RequestMetrics,
        limiter: RequestLimiter,
        tracer: Optional[trace.Tracer],
    ):
        self.backend = backend
        self.settings = settings
        self.logger = LOGGER
        self.metrics = metrics
        self.limiter = limiter
        self.tracer = tracer

    def _build_response(
        self, result: rasr.SpeechRecognitionResult
    ) -> rasr.RecognizeResponse:
        response = rasr.RecognizeResponse()
        response.results.append(result)
        return response

    def Recognize(
        self, request: rasr.RecognizeRequest, context
    ) -> rasr.RecognizeResponse:  # noqa: N802
        try:
            with (
                self.metrics.track("Recognize"),
                self.limiter.limit(self.limiter.general),
            ):
                span = self.tracer.start_span("Recognize") if self.tracer else None
                try:
                    config = (
                        request.config
                        if request.config.ListFields()
                        else rasr.RecognitionConfig()
                    )
                    options = config_to_options(config, self.settings)
                    audio, duration_seconds = self.backend.decode_bytes(
                        request.audio, config
                    )
                    trimmed_audio, offset_seconds = trim_audio_with_vad(
                        audio,
                        self.backend.sample_rate,
                        self.backend.vad_detector,
                        options.endpointing,
                    )
                    result = self.backend.recognize_from_audio(
                        trimmed_audio,
                        include_word_offsets=options.include_word_time_offsets,
                        enable_diarization=options.enable_diarization,
                        enable_punctuation=options.enable_punctuation,
                        max_speaker_count=options.max_speaker_count,
                        offset_seconds=offset_seconds,
                        original_duration_seconds=duration_seconds,
                    )
                    self.metrics.audio_seconds.labels(method="Recognize").inc(
                        result.audio_processed
                    )
                    if span:
                        span.set_attribute(
                            "audio.duration_seconds", result.audio_processed
                        )
                    return self._build_response(result)
                except ValueError as exc:
                    if span:
                        span.record_exception(exc)
                        span.end()
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                except Exception as exc:  # pragma: no cover - defensive guard
                    self.logger.exception("recognition failed: %s", exc)
                    if span:
                        span.record_exception(exc)
                        span.end()
                    context.abort(grpc.StatusCode.INTERNAL, "recognition failed")
                finally:
                    if span and span.is_recording():
                        span.end()
        except BusyError as exc:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))

    def StreamingRecognize(  # noqa: N802
        self,
        request_iterator: Iterable[rasr.StreamingRecognizeRequest],
        context,
    ) -> Iterator[rasr.StreamingRecognizeResponse]:
        try:
            with (
                self.metrics.track("StreamingRecognize"),
                self.limiter.limit(self.limiter.streaming),
            ):
                span = (
                    self.tracer.start_span("StreamingRecognize")
                    if self.tracer
                    else None
                )
                try:
                    config = rasr.RecognitionConfig()
                    interim_results = True
                    state: Optional[StreamingState] = None
                    config_seen = False
                    for message in request_iterator:
                        if message.HasField("streaming_config"):
                            config = message.streaming_config.config
                            interim_results = bool(
                                message.streaming_config.interim_results
                            )
                            state = StreamingState(
                                options=config_to_options(config, self.settings)
                            )
                            state.last_emit_time = time.perf_counter()
                            config_seen = True
                            continue
                        if not message.audio_content:
                            continue
                        if (
                            not config_seen
                            and config.encoding
                            == rasr.RecognitionConfig.AudioEncoding.LINEAR_PCM
                        ):
                            raise ValueError(
                                "streaming_config with sample_rate_hertz is required before PCM audio"
                            )
                        chunk_audio, duration_seconds = self.backend.decode_bytes(
                            message.audio_content, config
                        )
                        logits_chunk = self.backend.compute_logits(chunk_audio)
                        vad_probs, frame_duration = (
                            self.backend.vad_detector.probabilities(
                                chunk_audio,
                                self.backend.sample_rate,
                            )
                        )
                        if state is None:
                            state = StreamingState(
                                options=config_to_options(config, self.settings)
                            )
                            state.last_emit_time = time.perf_counter()
                        state.append(chunk_audio, logits_chunk, duration_seconds)
                        update_endpointing_state(
                            state,
                            state.options.endpointing,
                            vad_probs,
                            frame_duration,
                        )
                        self.metrics.audio_seconds.labels(
                            method="StreamingRecognize"
                        ).inc(duration_seconds)
                        now = time.perf_counter()
                        pipeline_vad = pipeline_vad_probabilities(state)
                        emit_interval_reached = (
                            now - state.last_emit_time
                        ) * 1000.0 >= float(self.settings.streaming_emit_interval_ms)
                        if (
                            interim_results
                            and state.speech_started
                            and emit_interval_reached
                            and not state.endpoint_triggered
                        ):
                            partial = self._build_streaming_response(
                                state,
                                is_final=False,
                                pipeline_vad=pipeline_vad,
                            )
                            if (
                                partial
                                and partial.results
                                and partial.results[0].alternatives
                            ):
                                transcript = (
                                    partial.results[0].alternatives[0].transcript
                                )
                                if transcript != state.last_transcript:
                                    yield partial
                                    state.last_transcript = transcript
                                state.last_emit_time = now
                        if state.endpoint_triggered and not state.final_emitted:
                            final_response = self._build_streaming_response(
                                state,
                                is_final=True,
                                pipeline_vad=pipeline_vad,
                            )
                            if span:
                                span.set_attribute(
                                    "audio.duration_seconds", state.total_audio_seconds
                                )
                            if final_response:
                                yield final_response
                            state.final_emitted = True
                            break
                    if state:
                        if span:
                            span.set_attribute(
                                "audio.duration_seconds", state.total_audio_seconds
                            )
                        if state.final_emitted:
                            return
                        final_response = self._build_streaming_response(
                            state,
                            is_final=True,
                            pipeline_vad=pipeline_vad_probabilities(state),
                        )
                        if final_response:
                            yield final_response
                except ValueError as exc:
                    if span:
                        span.record_exception(exc)
                        span.end()
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                except Exception as exc:  # pragma: no cover - defensive guard
                    self.logger.exception("streaming recognition failed: %s", exc)
                    if span:
                        span.record_exception(exc)
                        span.end()
                    context.abort(
                        grpc.StatusCode.INTERNAL, "streaming recognition failed"
                    )
                finally:
                    if span and span.is_recording():
                        span.end()
        except BusyError as exc:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))

    def _build_streaming_response(
        self,
        state: StreamingState,
        is_final: bool,
        pipeline_vad: Optional[Sequence[float]] = None,
    ) -> Optional[rasr.StreamingRecognizeResponse]:
        merged_logits = state.merged_logits()
        if merged_logits is None:
            return None
        transcript, offsets = self.backend.decode_logits(
            merged_logits,
            state.options.include_word_time_offsets,
        )
        if state.options.enable_punctuation:
            transcript = self.backend.punctuate(transcript)
        speaker_assignments: Optional[dict[int, int]] = None
        if state.options.enable_diarization and state.options.include_word_time_offsets:
            merged_audio = state.merged_audio()
            if merged_audio is not None:
                segments = estimate_speaker_segments(
                    merged_audio,
                    self.backend.sample_rate,
                    max(1, state.options.max_speaker_count),
                )
                speaker_assignments = assign_speakers(
                    offsets,
                    segments,
                    self.backend.time_offset_seconds,
                )
        streaming_response = rasr.StreamingRecognizeResponse()
        response_result = streaming_response.results.add()
        alternative = rasr.SpeechRecognitionAlternative()
        alternative.transcript = transcript
        alternative.confidence = 0.0
        if state.options.include_word_time_offsets:
            alternative.words.extend(
                self.backend.word_infos(
                    offsets,
                    speaker_assignments,
                )
            )
        response_result.alternatives.append(alternative)
        response_result.is_final = is_final
        response_result.channel_tag = 0
        response_result.audio_processed = state.total_audio_seconds
        response_result.stability = 1.0 if is_final else 0.9
        if pipeline_vad:
            pipeline_states = rasr.PipelineStates()
            pipeline_states.vad_probabilities.extend(pipeline_vad)
            response_result.pipeline_states.CopyFrom(pipeline_states)
        return streaming_response

    def GetRivaSpeechRecognitionConfig(  # noqa: N802
        self,
        request: rasr.RivaSpeechRecognitionConfigRequest,
        context,
    ) -> rasr.RivaSpeechRecognitionConfigResponse:
        response = rasr.RivaSpeechRecognitionConfigResponse()
        config = response.model_config.add()
        config.model_name = self.settings.model_name
        config.parameters["version"] = self.settings.model_version
        config.parameters["sample_rate_hz"] = str(self.backend.sample_rate)
        config.parameters["language_code"] = self.settings.default_language_code
        config.parameters["max_batch_size"] = str(self.settings.max_batch_size)
        config.parameters["max_concurrent_requests"] = str(
            self.settings.max_concurrent_requests
        )
        config.parameters["max_streaming_sessions"] = str(
            self.settings.max_streaming_sessions
        )
        config.parameters["supports_punctuation"] = "true"
        config.parameters["supports_diarization"] = "true"
        config.parameters["default_punctuation"] = str(
            self.settings.default_punctuation
        )
        config.parameters["default_diarization"] = str(
            self.settings.default_diarization
        )
        config.parameters["endpoint_start_history_ms"] = str(
            self.settings.endpoint_start_history_ms
        )
        config.parameters["endpoint_start_threshold"] = str(
            self.settings.endpoint_start_threshold
        )
        config.parameters["endpoint_stop_history_ms"] = str(
            self.settings.endpoint_stop_history_ms
        )
        config.parameters["endpoint_stop_threshold"] = str(
            self.settings.endpoint_stop_threshold
        )
        config.parameters["endpoint_stop_history_eou_ms"] = str(
            self.settings.endpoint_eou_history_ms
        )
        config.parameters["endpoint_stop_threshold_eou"] = str(
            self.settings.endpoint_eou_threshold
        )
        config.parameters["vad_frame_ms"] = str(self.settings.vad_frame_ms)
        return response


def create_triton_config(
    settings: ServiceSettings, sample_rate: int
) -> TritonModelConfig:
    """
    Build a TritonModelConfig from service settings.
    """
    return TritonModelConfig(
        name=settings.model_name,
        version=settings.model_version,
        max_batch_size=settings.max_batch_size,
        sample_rate=sample_rate,
        language_code=settings.default_language_code,
    )


def create_grpc_server(
    backend: ASRBackend,
    settings: ServiceSettings,
    metrics: RequestMetrics,
    limiter: RequestLimiter,
    tracer: Optional[trace.Tracer],
    auth_validator: AuthValidator,
) -> tuple[grpc.Server, TritonControlService]:
    """
    Configure the gRPC server with both Riva ASR and Triton control-plane services.
    """
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=settings.max_workers),
        interceptors=[AuthInterceptor(auth_validator)],
        options=[
            ("grpc.max_concurrent_streams", settings.max_streaming_sessions),
        ],
    )
    rasr_grpc.add_RivaSpeechRecognitionServicer_to_server(
        ParakeetServicer(backend, settings, metrics, limiter, tracer),
        grpc_server,
    )
    triton_service = TritonControlService(
        create_triton_config(settings, backend.sample_rate), settings
    )
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(
        triton_service, grpc_server
    )
    grpc_server.add_insecure_port(f"[::]:{settings.grpc_port}")
    return grpc_server, triton_service
