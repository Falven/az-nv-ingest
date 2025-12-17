from __future__ import annotations

from oim_common.settings import CommonSettings
from pydantic import AliasChoices, Field
from pydantic_settings import SettingsConfigDict


class ServiceSettings(CommonSettings):
    """
    Runtime configuration for the llama-3.2-nv-rerankqa-1b-v2 NIM service.
    """

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    model_id: str = Field("nvidia/llama-3.2-nv-rerankqa-1b-v2", alias="MODEL_ID")
    model_version: str = Field("1.8.0", alias="MODEL_VERSION")
    http_port: int = Field(8000, alias="NIM_HTTP_API_PORT")
    metrics_port: int | None = Field(
        8002,
        alias="NIM_TRITON_METRICS_PORT",
        validation_alias=AliasChoices("NIM_TRITON_METRICS_PORT", "NIM_METRICS_PORT"),
    )
    triton_http_endpoint: str = Field(
        "http://127.0.0.1:8003", alias="TRITON_HTTP_ENDPOINT"
    )
    triton_model_name: str = Field(
        "llama_3_2_nv_rerankqa_1b_v2", alias="TRITON_MODEL_NAME"
    )
    triton_timeout: float = Field(30.0, alias="TRITON_TIMEOUT_SECONDS")
    max_batch_size: int | None = Field(64, alias="NIM_TRITON_MAX_BATCH_SIZE")
    rate_limit: int | None = Field(None, alias="NIM_TRITON_RATE_LIMIT")
    enable_otel: bool = Field(False, alias="NIM_ENABLE_OTEL")
    otel_endpoint: str | None = Field(None, alias="NIM_OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str | None = Field(
        "llama-3.2-nv-rerankqa-1b-v2", alias="NIM_OTEL_SERVICE_NAME"
    )
    log_verbose: int = Field(0, alias="NIM_TRITON_LOG_VERBOSE")

    @property
    def auth_tokens(self) -> set[str]:
        """
        Resolve all configured bearer tokens for request authorization.
        """
        return self.resolved_auth_tokens()

    @property
    def auth_required(self) -> bool:
        """
        Determine whether HTTP endpoints should enforce bearer auth.
        """
        return self.require_auth
