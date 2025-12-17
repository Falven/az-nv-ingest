from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
STUB_PATH = PROJECT_ROOT / "tests" / "stubs"
MODEL_NAME = "nvidia/nemotron-nano-12b-v2-vl"
TEST_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _HealthHandler(BaseHTTPRequestHandler):
    """
    Tiny HTTP handler that marks Triton as ready for tests.
    """

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/v2/health/ready") or self.path.startswith(
            "/v2/health/live"
        ):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:  # noqa: A003
        return


def start_triton_health_server(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def http_json(
    method: str,
    url: str,
    payload: Dict[str, Any] | None,
    headers: Dict[str, str],
    timeout: float = 5.0,
) -> Tuple[int, Dict[str, Any] | None]:
    body = None
    request_headers = dict(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(
        url, data=body, headers=request_headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        parsed = json.loads(raw) if raw else None
        return exc.code, parsed
    except urllib.error.URLError:
        return 0, None


def wait_for_ready(base_url: str, timeout: float = 30.0) -> None:
    ready_url = f"{base_url}/health/ready"
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, payload = http_json("GET", ready_url, None, {}, timeout=3.0)
        if status == 200 and payload and payload.get("ready") is True:
            return
        time.sleep(0.5)
    raise RuntimeError("Service failed to become ready")


@pytest.fixture(scope="module")
def service() -> Iterator[Dict[str, str]]:
    http_port = find_free_port()
    triton_port = find_free_port()
    triton_grpc_port = find_free_port()
    triton_metrics_port = find_free_port()
    health_server = start_triton_health_server(triton_port)
    env = os.environ.copy()
    env.update(
        {
            "NIM_HTTP_API_PORT": str(http_port),
            "NIM_TRITON_HTTP_PORT": str(triton_port),
            "NIM_TRITON_GRPC_PORT": str(triton_grpc_port),
            "NIM_TRITON_METRICS_PORT": str(triton_metrics_port),
            "NIM_METRICS_PORT": "0",
            "NIM_REQUIRE_AUTH": "0",
            "ENABLE_MOCK_INFERENCE": "1",
        }
    )
    path_parts = [str(STUB_PATH), str(SRC_PATH)]
    existing_path = env.get("PYTHONPATH")
    if existing_path:
        path_parts.append(existing_path)
    env["PYTHONPATH"] = ":".join(path_parts)

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "nemotron_nano_12b_v2_vl.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(http_port),
    ]
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    base_url = f"http://127.0.0.1:{http_port}/v1"
    try:
        wait_for_ready(base_url)
        yield {"base_url": base_url}
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        health_server.shutdown()
        health_server.server_close()


def test_health_and_metadata(service: Dict[str, str]) -> None:
    status, ready_payload = http_json(
        "GET", f"{service['base_url']}/health/ready", None, {}
    )
    assert status == 200
    assert ready_payload == {"ready": True, "error": None}

    status, live_payload = http_json(
        "GET", f"{service['base_url']}/health/live", None, {}
    )
    assert status == 200
    assert live_payload == {"live": True}

    status, models_payload = http_json("GET", f"{service['base_url']}/models", None, {})
    assert status == 200
    assert models_payload is not None
    assert models_payload.get("data")
    assert models_payload["data"][0]["id"] == MODEL_NAME

    status, metadata_payload = http_json(
        "GET", f"{service['base_url']}/metadata", None, {}
    )
    assert status == 200
    assert metadata_payload is not None
    assert metadata_payload.get("id") == MODEL_NAME
    max_output_tokens = metadata_payload.get("max_output_tokens")
    assert isinstance(max_output_tokens, int) and max_output_tokens >= 1


def test_chat_completions_contract(service: Dict[str, str]) -> None:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "/no_think"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Caption the content of this image:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE}"},
                    },
                ],
            },
        ],
        "max_tokens": 64,
        "temperature": 0.5,
        "top_p": 1.0,
        "stream": False,
    }
    status, body = http_json(
        "POST",
        f"{service['base_url']}/chat/completions",
        payload,
        {},
        timeout=10.0,
    )
    assert status == 200, body
    assert body is not None
    assert body.get("model") == MODEL_NAME
    assert body.get("object") == "chat.completion"
    choices = body.get("choices")
    assert isinstance(choices, list)
    assert len(choices) >= 1
    first = choices[0]
    assert first.get("finish_reason") == "stop"
    message = first.get("message") or {}
    assert message.get("role") == "assistant"
    assert isinstance(message.get("content"), str)
    assert message["content"]
