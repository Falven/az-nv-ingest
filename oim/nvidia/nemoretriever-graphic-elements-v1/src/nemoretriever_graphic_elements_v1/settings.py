from __future__ import annotations

import sys
from pathlib import Path

from common.settings import CommonSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict

DEFAULT_MODEL_NAME = "nemoretriever-graphic-elements-v1"
DEFAULT_MODEL_VERSION = "1.0.0"
DEFAULT_CONF_THRESHOLD = 0.01
DEFAULT_IOU_THRESHOLD = 0.25
DEFAULT_SCORE_THRESHOLD = 0.1


class ServiceSettings(CommonSettings):
    """Runtime configuration for the graphic elements NIM service."""

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    model_name: str = Field(DEFAULT_MODEL_NAME, alias="MODEL_ID")
    model_version: str = Field(DEFAULT_MODEL_VERSION, alias="YOLOX_TAG")
    http_port: int = Field(8000, alias="NIM_HTTP_API_PORT")
    grpc_port: int = Field(8001, alias="NIM_GRPC_API_PORT")
    metrics_port: int | None = Field(8002, alias="NIM_METRICS_PORT")
    triton_http_port: int = Field(8100, alias="NIM_TRITON_HTTP_PORT")
    triton_metrics_port: int = Field(8003, alias="NIM_TRITON_METRICS_PORT")
    triton_model_repository: str = Field(
        "/app/triton/model_repository", alias="TRITON_MODEL_REPOSITORY"
    )
    triton_server_bin: str = Field(
        "/opt/tritonserver/bin/tritonserver", alias="TRITON_SERVER_BIN"
    )
    triton_model_name: str = Field("yolox_ensemble", alias="TRITON_MODEL_NAME")
    max_batch_size: int = Field(8, alias="NIM_TRITON_MAX_BATCH_SIZE")
    threshold: float = Field(DEFAULT_SCORE_THRESHOLD, alias="THRESHOLD")
    device: str | None = Field(None, alias="DEVICE")
    request_timeout_seconds: float = Field(30.0, alias="REQUEST_TIMEOUT_SECONDS")
    enable_mock_inference: bool = Field(False, alias="ENABLE_MOCK_INFERENCE")
    log_verbose: int = Field(0, alias="NIM_TRITON_LOG_VERBOSE")
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
            "DEFAULT_SCORE_THRESHOLD": str(self.threshold),
            "DEFAULT_CONF_THRESHOLD": str(DEFAULT_CONF_THRESHOLD),
            "DEFAULT_IOU_THRESHOLD": str(DEFAULT_IOU_THRESHOLD),
            "ENABLE_MOCK_INFERENCE": "1" if self.enable_mock_inference else "0",
        }
        if self.device:
            env["DEVICE"] = self.device
        return env
