from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUB_PATH = PROJECT_ROOT / "tests" / "stubs"
MODEL_ID = "nvidia/llama-3.2-nv-embedqa-1b-v2"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


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


def wait_for_ready(base_url: str, token: str) -> None:
    deadline = time.time() + 30.0
    headers = {"Authorization": f"Bearer {token}"}
    ready_url = f"{base_url}/health/ready"
    while time.time() < deadline:
        status, payload = http_json(
            "GET", ready_url, None, headers=headers, timeout=3.0
        )
        if status == 200 and payload and payload.get("ready") is True:
            return
        time.sleep(0.5)
    raise RuntimeError("Service failed to become ready")


@pytest.fixture(scope="module")
def service() -> Iterator[Dict[str, Any]]:
    port = find_free_port()
    token = "test-token"
    env = os.environ.copy()
    env.update(
        {
            "NIM_HTTP_API_PORT": str(port),
            "TRITON_HTTP_ENDPOINT": "http://127.0.0.1:18003",
            "NVIDIA_API_KEY": token,
            "NGC_API_KEY": token,
            "NIM_TRITON_MAX_BATCH_SIZE": "8",
            "ENABLE_MOCK_INFERENCE": "1",
        }
    )
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{STUB_PATH}:{existing_path}" if existing_path else str(STUB_PATH)
    )
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "oim_llama_3_2_nv_embedqa_1b_v2.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    base_url = f"http://127.0.0.1:{port}/v1"
    try:
        wait_for_ready(base_url, token)
        yield {"base_url": base_url, "token": token}
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def test_health_and_metadata(service: Dict[str, Any]) -> None:
    headers = {"Authorization": f"Bearer {service['token']}"}
    status, ready_payload = http_json(
        "GET", f"{service['base_url']}/health/ready", None, headers=headers
    )
    assert status == 200
    assert ready_payload == {"ready": True}

    status, models_payload = http_json(
        "GET", f"{service['base_url']}/models", None, headers=headers
    )
    assert status == 200
    assert models_payload is not None
    assert models_payload.get("data") and models_payload["data"][0]["id"] == MODEL_ID

    status, metadata_payload = http_json(
        "GET", f"{service['base_url']}/metadata", None, headers=headers
    )
    assert status == 200
    assert metadata_payload is not None
    assert metadata_payload.get("id") == MODEL_ID
    assert metadata_payload.get("modelInfo")
    short_name = metadata_payload["modelInfo"][0]["shortName"]
    assert short_name.startswith(MODEL_ID)


def test_embeddings_contract(service: Dict[str, Any]) -> None:
    headers = {"Authorization": f"Bearer {service['token']}"}
    payload = {
        "model": MODEL_ID,
        "input": ["example passage text", "second sample"],
        "input_type": "passage",
        "truncate": "END",
        "dimensions": 8,
    }
    status, body = http_json(
        "POST",
        f"{service['base_url']}/embeddings",
        payload,
        headers=headers,
        timeout=10.0,
    )
    assert status == 200
    assert body is not None
    assert body.get("object") == "list"
    assert body.get("model") == MODEL_ID
    data = body.get("data")
    assert isinstance(data, list)
    assert len(data) == len(payload["input"])
    for index, item in enumerate(data):
        assert item["object"] == "embedding"
        assert item["index"] == index
        assert item["model"] == MODEL_ID
        embedding = item["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) == payload["dimensions"]
    usage = body.get("usage") or {}
    assert isinstance(usage.get("prompt_tokens"), int)
    assert usage["prompt_tokens"] >= 1
    assert usage["total_tokens"] == usage["prompt_tokens"]
