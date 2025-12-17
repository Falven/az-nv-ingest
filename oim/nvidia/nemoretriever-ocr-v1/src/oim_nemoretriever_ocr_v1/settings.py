from __future__ import annotations

from typing import Literal

from oim_common.settings import CommonSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict

MergeLevel = Literal["word", "sentence", "paragraph"]

DEFAULT_MODEL_NAME = "scene_text_wrapper"
DEFAULT_MODEL_VERSION = "1.1.0"


class ServiceSettings(CommonSettings):
    """
    Runtime configuration for the nemoretriever-ocr-v1 NIM service.
    """

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    http_port: int = Field(8000, alias="NIM_HTTP_API_PORT")
    grpc_port: int = Field(8001, alias="NIM_TRITON_GRPC_PORT")
    metrics_port: int = Field(8002, alias="NIM_TRITON_METRICS_PORT")
    merge_level: MergeLevel = Field("paragraph", alias="MERGE_LEVEL")
    max_batch_size: int = Field(8, alias="NIM_TRITON_MAX_BATCH_SIZE")
    model_dir: str = Field("./checkpoints", alias="MODEL_DIR")
    request_timeout_seconds: float = Field(30.0, alias="REQUEST_TIMEOUT_SECONDS")
    model_version: str = Field(DEFAULT_MODEL_VERSION, alias="MODEL_VERSION")
    model_name: str = Field(DEFAULT_MODEL_NAME, alias="OCR_MODEL_NAME")
    triton_grpc_url: str = Field("localhost:8001", alias="TRITON_GRPC_URL")
    triton_http_url: str = Field("http://127.0.0.1:8003", alias="TRITON_HTTP_URL")
    enable_mock_inference: bool = Field(False, alias="ENABLE_MOCK_INFERENCE")
    otel_endpoint: str | None = Field(None, alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str = Field("nemoretriever-ocr-v1", alias="OTEL_SERVICE_NAME")
    require_auth: bool = Field(False, alias="NIM_REQUIRE_AUTH")

    @property
    def auth_tokens(self) -> set[str]:
        """
        Resolve all configured bearer tokens for request authorization.

        Returns:
            Set of bearer tokens sourced from supported environment variables.
        """
        return self.resolved_auth_tokens()

    @property
    def auth_required(self) -> bool:
        """
        Determine whether HTTP/gRPC endpoints should enforce bearer auth.

        Returns:
            True when authentication is mandatory for incoming requests.
        """
        return self.require_auth or bool(self.auth_tokens)
