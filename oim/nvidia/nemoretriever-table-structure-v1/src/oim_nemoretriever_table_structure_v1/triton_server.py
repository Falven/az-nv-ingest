from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tritonclient.grpc import InferenceServerClient, InferenceServerException

from .errors import TritonStartupError
from .settings import ServiceSettings

logger = logging.getLogger(__name__)


class TritonServer:
    """Manage the embedded Triton server process."""

    def __init__(self, settings: ServiceSettings) -> None:
        """Initialize with service settings."""
        self._settings = settings
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        """Start Triton if it is not already running and wait for readiness."""
        if self._process is not None:
            return

        if not Path(self._settings.triton_server_bin).exists():
            raise TritonStartupError(
                f"Triton binary not found at {self._settings.triton_server_bin}"
            )
        if not Path(self._settings.triton_model_repository).exists():
            raise TritonStartupError(
                f"Model repository missing at {self._settings.triton_model_repository}"
            )

        cmd = [
            self._settings.triton_server_bin,
            f"--model-repository={self._settings.triton_model_repository}",
            f"--grpc-port={self._settings.grpc_port}",
            f"--http-port={self._settings.triton_http_port}",
            f"--metrics-port={self._settings.triton_metrics_port}",
            "--model-control-mode=none",
            f"--log-verbose={self._settings.log_verbose}",
        ]

        env = os.environ.copy()
        env.update(self._settings.triton_environment)
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
        )
        try:
            await self._wait_for_live()
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the Triton server process."""
        if self._process is None:
            return
        process = self._process
        self._process = None
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    @retry(
        retry=retry_if_exception_type(TritonStartupError),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        stop=stop_after_attempt(20),
        reraise=True,
    )
    async def _wait_for_live(self) -> None:
        """Poll the server until it reports live/ready."""
        if self._process is None:
            raise TritonStartupError("Triton process was not started.")
        if self._process.returncode is not None:
            raise TritonStartupError("Triton process exited prematurely.")
        try:
            await asyncio.to_thread(self._assert_live_once)
        except TritonStartupError:
            raise
        except Exception as exc:
            raise TritonStartupError(f"Triton liveness check failed: {exc}") from exc

    def _assert_live_once(self) -> None:
        """Single liveness probe using the Triton gRPC client."""
        try:
            client = InferenceServerClient(
                url=self._settings.triton_grpc_endpoint,
                verbose=False,
                network_timeout=self._settings.request_timeout_seconds,
            )
            live = bool(client.is_server_live())
            ready = bool(client.is_server_ready())
            client.close()
        except InferenceServerException as exc:
            raise TritonStartupError(str(exc)) from exc
        if not live or not ready:
            raise TritonStartupError("Triton is not live yet.")
