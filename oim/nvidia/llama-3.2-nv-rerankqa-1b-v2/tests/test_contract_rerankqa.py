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
MODEL_ID = "nvidia/llama-3.2-nv-rerankqa-1b-v2"


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


def wait_for_ready(base_url: str) -> None:
    deadline = time.time() + 30.0
    ready_url = f"{base_url}/health/ready"
    while time.time() < deadline:
        status, payload = http_json("GET", ready_url, None, headers={}, timeout=3.0)
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
    src_path = PROJECT_ROOT / "src"
    combined_path = f"{STUB_PATH}:{src_path}"
    env["PYTHONPATH"] = (
        f"{combined_path}:{existing_path}" if existing_path else combined_path
    )
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "oim_llama_3_2_nv_rerankqa_1b_v2.server:app",
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
        wait_for_ready(base_url)
        yield {"base_url": base_url, "token": token}
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def test_health_and_metadata(service: Dict[str, Any]) -> None:
    headers = {"Authorization": f"Bearer {service['token']}"}
    status, live_payload = http_json(
        "GET", f"{service['base_url']}/health/live", None, headers=headers
    )
    assert status == 200
    assert live_payload == {"live": True}

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
    info = metadata_payload.get("modelInfo") or []
    assert info
    assert info[0].get("shortName", "").startswith(MODEL_ID)
    assert isinstance(info[0].get("maxBatchSize"), int)


def _assert_rankings(body: Dict[str, Any], expected_count: int) -> None:
    rankings = body.get("rankings")
    assert isinstance(rankings, list)
    assert len(rankings) == expected_count
    indices = {item.get("index") for item in rankings}
    assert indices == set(range(expected_count))
    logits = [item.get("logit") for item in rankings]
    assert all(isinstance(value, float) for value in logits)
    assert logits == sorted(logits, reverse=True)


def test_ranking_contract(service: Dict[str, Any]) -> None:
    headers = {"Authorization": f"Bearer {service['token']}"}
    payload = {
        "model": MODEL_ID,
        "query": {"text": "which way should I go?"},
        "passages": [
            {"text": "two roads diverged in a yellow wood"},
            {"text": "then took the other"},
        ],
        "truncate": "END",
    }
    status, body = http_json(
        "POST", f"{service['base_url']}/ranking", payload, headers=headers, timeout=10.0
    )
    assert status == 200
    assert body is not None
    _assert_rankings(body, expected_count=len(payload["passages"]))


def test_compatibility_reranking(service: Dict[str, Any]) -> None:
    headers = {"Authorization": f"Bearer {service['token']}"}
    payload = {
        "query": {"text": "what path is best?"},
        "passages": [
            {"text": "keep left"},
            {"text": "turn right"},
            {"text": "walk forward"},
        ],
        "truncate": "END",
    }
    status, body = http_json(
        "POST",
        f"{service['base_url']}/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
        payload,
        headers=headers,
        timeout=10.0,
    )
    assert status == 200
    assert body is not None
    _assert_rankings(body, expected_count=len(payload["passages"]))
