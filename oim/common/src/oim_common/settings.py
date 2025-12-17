from __future__ import annotations

from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CommonSettings(BaseSettings):
    """
    Base settings shared across NIM services.
    """

    model_config = SettingsConfigDict(
        populate_by_name=True, extra="ignore", cli_parse_args=False
    )

    log_level: str = Field("INFO", alias="LOG_LEVEL")
    require_auth: bool = Field(True, alias="NIM_REQUIRE_AUTH")
    auth_token: str | None = Field(None, alias="NGC_API_KEY")
    fallback_auth_token: str | None = Field(None, alias="NIM_NGC_API_KEY")
    nvidia_auth_token: str | None = Field(None, alias="NVIDIA_API_KEY")

    def resolved_auth_tokens(self) -> set[str]:
        """
        Collect all configured bearer tokens, omitting empty entries.

        Returns:
            A set of non-null bearer tokens from the supported environment variables.
        """
        return {
            token
            for token in (
                self.auth_token,
                self.fallback_auth_token,
                self.nvidia_auth_token,
            )
            if token is not None
        }


class HttpServerSettings(CommonSettings):
    """
    Shared HTTP server settings including auth, metrics, and telemetry knobs.
    """

    http_port: int = Field(8000, alias="NIM_HTTP_API_PORT")
    metrics_port: int | None = Field(8002, alias="NIM_METRICS_PORT")
    rate_limit: int | None = Field(None, alias="NIM_TRITON_RATE_LIMIT")
    enable_otel: bool = Field(False, alias="NIM_ENABLE_OTEL")
    otel_endpoint: str | None = Field(None, alias="NIM_OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str | None = Field(None, alias="NIM_OTEL_SERVICE_NAME")
    log_verbose: int = Field(0, alias="NIM_TRITON_LOG_VERBOSE")


class TritonHttpSettings(CommonSettings):
    """
    Triton HTTP client configuration shared across NIM shims.
    """

    triton_http_endpoint: str = Field(
        "http://127.0.0.1:8003", alias="TRITON_HTTP_ENDPOINT"
    )
    triton_model_name: str = Field("model", alias="TRITON_MODEL_NAME")
    triton_timeout: float = Field(30.0, alias="TRITON_TIMEOUT_SECONDS")
    max_batch_size: Optional[int] = Field(None, alias="NIM_TRITON_MAX_BATCH_SIZE")
    rate_limit: Optional[int] = Field(None, alias="NIM_TRITON_RATE_LIMIT")
    log_verbose: int = Field(0, alias="NIM_TRITON_LOG_VERBOSE")


class MetricsSettings(CommonSettings):
    """
    Prometheus metrics exposure configuration.
    """

    metrics_port: int | None = Field(
        8002,
        alias="NIM_TRITON_METRICS_PORT",
        validation_alias=AliasChoices("NIM_TRITON_METRICS_PORT", "NIM_METRICS_PORT"),
    )


class OtelSettings(CommonSettings):
    """
    OpenTelemetry configuration shared across services.
    """

    enable_otel: bool = Field(False, alias="NIM_ENABLE_OTEL")
    otel_endpoint: str | None = Field(None, alias="NIM_OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str | None = Field(None, alias="NIM_OTEL_SERVICE_NAME")
