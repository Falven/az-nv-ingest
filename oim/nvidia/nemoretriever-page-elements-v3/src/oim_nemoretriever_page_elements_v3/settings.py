from __future__ import annotations

import sys
from pathlib import Path

from oim_common.settings import CommonSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict

DEFAULT_MODEL_ID = "nemoretriever-page-elements-v3"
DEFAULT_MODEL_VERSION = "1.7.0"
DEFAULT_CONF_THRESHOLD = 0.01
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_SCORE_THRESHOLD = 0.1


class ServiceSettings(CommonSettings):
    """
    Runtime configuration for the nemoretriever-page-elements-v3 NIM service.
    """

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    model_id: str = Field(DEFAULT_MODEL_ID, alias="MODEL_ID")
    model_version: str = Field(DEFAULT_MODEL_VERSION, alias="YOLOX_TAG")
    http_port: int = Field(8000, alias="NIM_HTTP_API_PORT")
    grpc_port: int = Field(8001, alias="NIM_GRPC_API_PORT")
    metrics_port: int | None = Field(8002, alias="NIM_METRICS_PORT")
    triton_http_port: int = Field(8100, alias="NIM_TRITON_HTTP_PORT")
    triton_metrics_port: int = Field(8003, alias="NIM_TRITON_METRICS_PORT")
    triton_model_repository: str = Field(
        "/app/triton/model_repository", alias="TRITON_MODEL_REPOSITORY"
    )
    triton_model_name: str = Field("pipeline", alias="TRITON_MODEL_NAME")
    max_batch_size: int = Field(8, alias="NIM_TRITON_MAX_BATCH_SIZE")
    default_conf_threshold: float = Field(
        DEFAULT_CONF_THRESHOLD, alias="YOLOX_CONF_THRESHOLD"
    )
    default_iou_threshold: float = Field(
        DEFAULT_IOU_THRESHOLD, alias="YOLOX_IOU_THRESHOLD"
    )
    default_score_threshold: float = Field(
        DEFAULT_SCORE_THRESHOLD, alias="YOLOX_MIN_SCORE"
    )
    device: str | None = Field(None, alias="DEVICE")
    request_timeout_seconds: float = Field(30.0, alias="REQUEST_TIMEOUT_SECONDS")
    enable_mock_inference: bool = Field(False, alias="ENABLE_MOCK_INFERENCE")
    log_verbose: int = Field(0, alias="NIM_TRITON_LOG_VERBOSE")
    rate_limit: int | None = Field(None, alias="NIM_TRITON_RATE_LIMIT")
    enable_otel: bool = Field(False, alias="NIM_ENABLE_OTEL")
    otel_endpoint: str | None = Field(None, alias="NIM_OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str | None = Field(None, alias="NIM_OTEL_SERVICE_NAME")
    venv_path: str = Field("/opt/venv")

    @property
    def auth_tokens(self) -> set[str]:
        """Resolve configured bearer tokens."""
        return self.resolved_auth_tokens()

    @property
    def auth_required(self) -> bool:
        """Return whether auth should be enforced for requests."""
        return self.require_auth or bool(self.auth_tokens)

    @property
    def effective_log_level(self) -> str:
        """Compute the effective log level from verbosity and configured level."""
        return "DEBUG" if self.log_verbose else self.log_level

    @property
    def triton_grpc_endpoint(self) -> str:
        """Endpoint used by the HTTP shim to reach Triton locally."""
        return f"localhost:{self.grpc_port}"

    @property
    def python_site_packages(self) -> str:
        """Site-packages path for the embedded virtual environment."""
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        return str(Path(self.venv_path) / "lib" / version / "site-packages")

    @property
    def triton_environment(self) -> dict[str, str]:
        """Environment variables provided to the Triton server process."""
        python_path = [
            self.python_site_packages,
            str(Path(__file__).resolve().parent.parent),
        ]
        env = {
            "PYTHONPATH": ":".join(python_path),
            "DEFAULT_SCORE_THRESHOLD": str(self.default_score_threshold),
            "DEFAULT_CONF_THRESHOLD": str(self.default_conf_threshold),
            "DEFAULT_IOU_THRESHOLD": str(self.default_iou_threshold),
            "ENABLE_MOCK_INFERENCE": "1" if self.enable_mock_inference else "0",
        }
        if self.device:
            env["DEVICE"] = self.device
        return env
