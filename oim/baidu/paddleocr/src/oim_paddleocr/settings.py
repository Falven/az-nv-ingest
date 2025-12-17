from __future__ import annotations

from typing import Optional, Set

from oim_common.settings import CommonSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class ServiceSettings(CommonSettings):
    """
    PaddleOCR service settings.
    """

    http_port: int = Field(8000, alias="NIM_HTTP_API_PORT")
    grpc_port: int = Field(8001, alias="NIM_TRITON_GRPC_PORT")
    metrics_port: int = Field(8002, alias="NIM_TRITON_METRICS_PORT")
    max_batch_size: int = Field(2, alias="NIM_TRITON_MAX_BATCH_SIZE")
    enable_model_control: bool = Field(False, alias="NIM_TRITON_ENABLE_MODEL_CONTROL")
    rate_limit: Optional[int] = Field(None, alias="NIM_TRITON_RATE_LIMIT")
    triton_log_verbose: int = Field(0, alias="NIM_TRITON_LOG_VERBOSE")
    request_timeout_seconds: float = Field(30.0, alias="REQUEST_TIMEOUT_SECONDS")
    use_layout_detection: bool = Field(True, alias="USE_LAYOUT_DETECTION")
    use_chart_recognition: bool = Field(True, alias="USE_CHART_RECOGNITION")
    format_block_content: bool = Field(False, alias="FORMAT_BLOCK_CONTENT")
    merge_layout_blocks: bool = Field(True, alias="MERGE_LAYOUT_BLOCKS")
    vl_rec_backend: Optional[str] = Field(None, alias="VL_REC_BACKEND")
    vl_rec_server_url: Optional[str] = Field(None, alias="VL_REC_SERVER_URL")
    model_id: str = Field("baidu/paddleocr", alias="MODEL_ID")
    model_version: str = Field("1.5.0", alias="MODEL_VERSION")
    short_name: str = Field("paddleocr", alias="MODEL_SHORT_NAME")
    model_name: str = Field("paddle", alias="OCR_MODEL_NAME")
    enable_otel: bool = Field(False, alias="NIM_ENABLE_OTEL")
    otel_service_name: str = Field("paddleocr", alias="NIM_OTEL_SERVICE_NAME")
    otel_endpoint: Optional[str] = Field(None, alias="NIM_OTEL_EXPORTER_OTLP_ENDPOINT")

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    def allowed_tokens(self) -> Set[str]:
        return self.resolved_auth_tokens()
