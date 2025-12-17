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

import grpc
import pytest
from riva.client.proto import riva_asr_pb2 as rasr
from riva.client.proto import riva_asr_pb2_grpc as rasr_grpc

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(
    method: str, url: str, payload: Dict[str, Any] | None, timeout: float = 5.0
) -> Tuple[int, Dict[str, Any] | None]:
    body = None
    headers: Dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
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
    ready_url = f"{base_url}/v1/health/ready"
    while time.time() < deadline:
        status, payload = http_json("GET", ready_url, None, timeout=3.0)
        if status == 200 and payload and payload.get("status") == "READY":
            return
        time.sleep(0.5)
    raise RuntimeError("Service failed to become ready")


@pytest.fixture(scope="module")
def service() -> Iterator[Dict[str, Any]]:
    http_port = find_free_port()
    grpc_port = find_free_port()
    env = os.environ.copy()
    env.update(
        {
            "HTTP_PORT": str(http_port),
            "GRPC_PORT": str(grpc_port),
            "TRITON_HTTP_ENDPOINT": "http://127.0.0.1:8003",
            "NIM_REQUIRE_AUTH": "0",
            "ALLOW_UNAUTH_HEALTH": "1",
            "ENABLE_MOCK_INFERENCE": "1",
        }
    )
    existing_path = env.get("PYTHONPATH", "")
    module_paths = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT.parent / "common" / "src",
    ]
    joined_paths = os.pathsep.join(str(path) for path in module_paths)
    env["PYTHONPATH"] = (
        os.pathsep.join([joined_paths, existing_path])
        if existing_path
        else joined_paths
    )
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "oim_parakeet_1_1b_ctc_en_us.server:app",
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
    base_url = f"http://127.0.0.1:{http_port}"
    try:
        wait_for_ready(base_url)
        yield {"base_url": base_url, "grpc_port": grpc_port}
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def test_health_and_metadata(service: Dict[str, Any]) -> None:
    status, live_payload = http_json(
        "GET", f"{service['base_url']}/v1/health/live", None
    )
    assert status == 200
    assert live_payload == {"status": "SERVING"}

    status, ready_payload = http_json(
        "GET", f"{service['base_url']}/v1/health/ready", None
    )
    assert status == 200
    assert ready_payload is not None
    assert ready_payload.get("status") == "READY"
    assert ready_payload.get("model") == "parakeet-1-1b-ctc-en-us"

    status, metadata_payload = http_json(
        "GET", f"{service['base_url']}/v1/metadata", None
    )
    assert status == 200
    assert metadata_payload is not None
    assert metadata_payload.get("id") == "parakeet-1-1b-ctc-en-us"
    model_info = metadata_payload.get("modelInfo") or []
    assert isinstance(model_info, list) and model_info
    short_name = model_info[0].get("shortName")
    assert isinstance(short_name, str) and short_name.startswith(
        "parakeet-1-1b-ctc-en-us"
    )


def test_grpc_recognize_mock(service: Dict[str, Any]) -> None:
    channel = grpc.insecure_channel(f"127.0.0.1:{service['grpc_port']}")
    stub = rasr_grpc.RivaSpeechRecognitionStub(channel)
    config = rasr.RecognitionConfig(
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_word_time_offsets=True,
    )
    request = rasr.RecognizeRequest(config=config, audio=b"\x00\x00" * 160)
    response = stub.Recognize(request, timeout=5)
    channel.close()

    assert response.results, "Expected at least one recognition result"
    alternative = response.results[0].alternatives[0]
    assert alternative.transcript
    assert alternative.words
    assert alternative.words[0].start_time >= 0
