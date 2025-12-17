from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
import urllib.request

from .settings import ServiceSettings

logger = logging.getLogger(__name__)


class TritonServer:
    """
    Helper that manages a Triton server process for local inference.
    """

    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings
        self._process: subprocess.Popen[bytes] | None = None
        self._started = False

    async def start(self) -> None:
        """
        Launch Triton if it is not already running.
        """
        if await self._is_ready():
            logger.info("Triton server already running; skipping launch.")
            return
        if self._process is not None:
            return
        cmd = [
            self._settings.triton_server_bin,
            f"--model-repository={self._settings.triton_model_repository}",
            f"--http-port={self._settings.triton_http_port}",
            f"--grpc-port={self._settings.triton_grpc_port}",
            f"--metrics-port={self._settings.triton_metrics_port}",
            "--model-control-mode=none",
            f"--log-verbose={self._settings.log_verbose}",
        ]
        env = os.environ.copy()
        env.update(self._settings.triton_environment)
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._started = True
        await self._wait_ready()

    async def stop(self) -> None:
        """
        Stop the managed Triton process when started by this helper.
        """
        if not self._started or self._process is None:
            return
        self._process.terminate()
        try:
            await asyncio.wait_for(asyncio.to_thread(self._process.wait), timeout=10.0)
        except asyncio.TimeoutError:
            self._process.kill()
        self._process = None
        self._started = False

    async def _wait_ready(self) -> None:
        """
        Poll until Triton reports ready or times out.
        """
        deadline = time.monotonic() + self._settings.triton_timeout
        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                raise RuntimeError("Triton server exited during startup")
            if await self._is_ready():
                return
            await asyncio.sleep(0.5)
        raise RuntimeError("Triton server did not become ready in time")

    async def _is_ready(self) -> bool:
        """
        Check Triton readiness via the HTTP health endpoint.
        """
        url = f"http://127.0.0.1:{self._settings.triton_http_port}/v2/health/ready"
        return await asyncio.to_thread(self._probe_ready, url)

    @staticmethod
    def _probe_ready(url: str) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=2):  # nosec: B310 - health check
                return True
        except Exception:
            return False
