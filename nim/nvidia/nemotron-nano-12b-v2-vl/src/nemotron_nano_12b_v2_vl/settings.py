from __future__ import annotations

import sys
from pathlib import Path

from common.settings import CommonSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict

DEFAULT_SYSTEM_PROMPT = "/no_think"
DEFAULT_USER_PROMPT = "Caption the content of this image:"


class ServiceSettings(CommonSettings):
    """
    Runtime configuration for the Nemotron Nano VLM NIM.
    """

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    model_id: str = Field(
        "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", alias="MODEL_ID"
    )
    served_model_name: str = Field(
        "nvidia/nemotron-nano-12b-v2-vl", alias="SERVED_MODEL_NAME"
    )
    model_version: str = Field("1.0.0", alias="MODEL_VERSION")
    http_port: int = Field(8000, alias="NIM_HTTP_API_PORT")
    metrics_port: int | None = Field(8002, alias="NIM_METRICS_PORT")
    triton_http_port: int = Field(8003, alias="NIM_TRITON_HTTP_PORT")
    triton_grpc_port: int = Field(8004, alias="NIM_TRITON_GRPC_PORT")
    triton_metrics_port: int = Field(8005, alias="NIM_TRITON_METRICS_PORT")
    triton_model_repository: str = Field(
        "/app/triton/model_repository", alias="TRITON_MODEL_REPOSITORY"
    )
    triton_server_bin: str = Field(
        "/opt/tritonserver/bin/tritonserver", alias="TRITON_SERVER_BIN"
    )
    triton_model_name: str = Field("nemotron_nano_12b_v2_vl", alias="TRITON_MODEL_NAME")
    triton_timeout: float = Field(120.0, alias="TRITON_TIMEOUT_SECONDS")
    log_verbose: int = Field(0, alias="NIM_TRITON_LOG_VERBOSE")
    rate_limit: int | None = Field(None, alias="NIM_TRITON_RATE_LIMIT")
    default_max_tokens: int = Field(512, alias="DEFAULT_MAX_TOKENS")
    max_output_tokens: int = Field(1024, alias="MAX_OUTPUT_TOKENS")
    max_batch_size: int = Field(1, alias="NIM_TRITON_MAX_BATCH_SIZE")
    system_prompt: str = Field(DEFAULT_SYSTEM_PROMPT, alias="VLM_CAPTION_SYSTEM_PROMPT")
    user_prompt: str = Field(DEFAULT_USER_PROMPT, alias="VLM_CAPTION_PROMPT")
    video_pruning_rate: float = Field(0.0, alias="VIDEO_PRUNING_RATE")
    enable_otel: bool = Field(False, alias="NIM_ENABLE_OTEL")
    otel_endpoint: str | None = Field(None, alias="NIM_OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str | None = Field(None, alias="NIM_OTEL_SERVICE_NAME")
    enable_mock_inference: bool = Field(False, alias="ENABLE_MOCK_INFERENCE")
    venv_path: str = Field("/opt/venv")

    @property
    def auth_tokens(self) -> set[str]:
        """
        Resolve configured bearer tokens, omitting empty entries.

        Returns:
            A set of non-empty bearer tokens.
        """
        return self.resolved_auth_tokens()

    @property
    def auth_required(self) -> bool:
        """
        Determine whether endpoints should enforce authentication.

        Returns:
            True when authentication is mandatory either because it is explicitly
            required or because tokens are configured.
        """
        return self.require_auth or bool(self.auth_tokens)

    @property
    def logging_level(self) -> str:
        """
        Compute the effective logging level.

        Returns:
            Upper-cased logging level string.
        """
        if self.log_verbose > 0:
            return "DEBUG"
        return self.log_level.upper()

    @property
    def triton_http_endpoint(self) -> str:
        """
        Endpoint for the local Triton HTTP server.
        """
        return f"http://127.0.0.1:{self.triton_http_port}"

    @property
    def python_site_packages(self) -> str:
        """
        Compute the site-packages path for the embedded virtual environment.
        """
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        return str(Path(self.venv_path) / "lib" / version / "site-packages")

    @property
    def triton_environment(self) -> dict[str, str]:
        """
        Environment variables passed to the Triton server process.
        """
        env = {
            "PYTHONPATH": ":".join(
                [
                    self.python_site_packages,
                    str(Path(__file__).resolve().parent.parent),
                ]
            ),
            "DEFAULT_MAX_TOKENS": str(self.default_max_tokens),
            "MAX_OUTPUT_TOKENS": str(self.max_output_tokens),
            "MODEL_ID": self.model_id,
            "VLM_CAPTION_SYSTEM_PROMPT": self.system_prompt,
            "VLM_CAPTION_PROMPT": self.user_prompt,
            "VIDEO_PRUNING_RATE": str(self.video_pruning_rate),
            "ENABLE_MOCK_INFERENCE": "1" if self.enable_mock_inference else "0",
        }
        return env
