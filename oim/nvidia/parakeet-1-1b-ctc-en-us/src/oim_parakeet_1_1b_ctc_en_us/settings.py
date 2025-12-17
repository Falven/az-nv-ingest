from __future__ import annotations

from typing import List

from oim_common.settings import CommonSettings
from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict


class ServiceSettings(CommonSettings):
    """
    Runtime configuration for the Parakeet ASR NIM.
    """

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    model_id: str = Field("nvidia/parakeet-ctc-1.1b", alias="MODEL_ID")
    model_name: str = Field("parakeet-1-1b-ctc-en-us", alias="MODEL_NAME")
    model_version: str = Field("1.1.0", alias="MODEL_VERSION")

    http_port: int = Field(9000, alias="HTTP_PORT")
    grpc_port: int = Field(50051, alias="GRPC_PORT")
    max_workers: int = Field(4, alias="MAX_WORKERS")
    max_concurrent_requests: int = Field(16, alias="MAX_CONCURRENT_REQUESTS")
    max_streaming_sessions: int = Field(8, alias="MAX_STREAMING_SESSIONS")
    max_batch_size: int = Field(4, alias="MAX_BATCH_SIZE")

    default_punctuation: bool = Field(True, alias="DEFAULT_PUNCTUATION")
    default_diarization: bool = Field(False, alias="DEFAULT_DIARIZATION")
    streaming_emit_interval_ms: int = Field(400, alias="STREAMING_EMIT_INTERVAL_MS")
    endpoint_start_history_ms: int = Field(240, alias="ENDPOINT_START_HISTORY_MS")
    endpoint_start_threshold: float = Field(0.6, alias="ENDPOINT_START_THRESHOLD")
    endpoint_stop_history_ms: int = Field(1200, alias="ENDPOINT_STOP_HISTORY_MS")
    endpoint_stop_threshold: float = Field(0.35, alias="ENDPOINT_STOP_THRESHOLD")
    endpoint_eou_history_ms: int = Field(1600, alias="ENDPOINT_EOU_HISTORY_MS")
    endpoint_eou_threshold: float = Field(0.2, alias="ENDPOINT_EOU_THRESHOLD")
    vad_frame_ms: int = Field(30, alias="VAD_FRAME_MS")
    max_speaker_count: int = Field(2, alias="MAX_SPEAKER_COUNT")
    punctuation_model_id: str = Field("kredor/punctuate-all", alias="PUNCTUATION_MODEL_ID")

    metrics_namespace: str = Field("parakeet", alias="METRICS_NAMESPACE")

    triton_http_endpoint: str = Field(
        "http://127.0.0.1:8003", alias="TRITON_HTTP_ENDPOINT"
    )
    triton_model_name: str = Field("parakeet_1_1b_ctc_en_us", alias="TRITON_MODEL_NAME")
    triton_timeout: float = Field(60.0, alias="TRITON_TIMEOUT_SECONDS")

    default_language_code: str = Field("en-US", alias="LANGUAGE_CODE")
    enable_automatic_punctuation: bool = Field(
        True, alias="ENABLE_AUTOMATIC_PUNCTUATION"
    )
    allow_unauth_health: bool = Field(True, alias="ALLOW_UNAUTH_HEALTH")
    log_verbose: int = Field(0, alias="NIM_TRITON_LOG_VERBOSE")

    additional_auth_tokens: List[str] = Field(default_factory=list, alias="AUTH_TOKENS")

    otel_endpoint: str | None = Field(None, alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str | None = Field("parakeet-asr", alias="OTEL_SERVICE_NAME")
    enable_mock_inference: bool = Field(False, alias="ENABLE_MOCK_INFERENCE")

    @field_validator("additional_auth_tokens", mode="before")
    @classmethod
    def split_tokens(cls, value: object) -> List[str]:
        """
        Normalize comma-separated strings or iterables into a clean token list.
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError("AUTH_TOKENS must be a string or iterable")

    @property
    def auth_tokens(self) -> set[str]:
        """
        Combine NGC tokens with any explicit AUTH_TOKENS list.
        """
        return self.resolved_auth_tokens().union(self.additional_auth_tokens)

    @property
    def auth_required(self) -> bool:
        """
        Determine whether endpoints must enforce bearer authentication.
        """
        return self.require_auth or bool(self.auth_tokens)
