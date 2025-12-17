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
MODEL_SHORT_NAME = "nemoretriever-ocr-v1"
IMAGE_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
    ready_url = f"{base_url}/v1/health/ready"
    while time.time() < deadline:
        status, payload = http_json("GET", ready_url, None, {}, timeout=3.0)
        if status == 200 and payload is not None and payload.get("status") == "ready":
            return
        time.sleep(0.5)
    raise RuntimeError("Service failed to become ready")


@pytest.fixture(scope="module")
def service() -> Iterator[Dict[str, Any]]:
    http_port = find_free_port()
    grpc_port = find_free_port()
    metrics_port = find_free_port()
    token = "test-token"
    env = os.environ.copy()
    env.update(
        {
            "NIM_HTTP_API_PORT": str(http_port),
            "NIM_TRITON_GRPC_PORT": str(grpc_port),
            "NIM_TRITON_METRICS_PORT": str(metrics_port),
            "TRITON_GRPC_URL": f"127.0.0.1:{grpc_port}",
            "TRITON_HTTP_URL": f"http://127.0.0.1:{metrics_port}",
            "NIM_TRITON_MAX_BATCH_SIZE": "4",
            "NVIDIA_API_KEY": token,
            "NGC_API_KEY": token,
            "NIM_REQUIRE_AUTH": "1",
        }
    )
    existing_path = env.get("PYTHONPATH", "")
    module_paths = [
        STUB_PATH,
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
        "nemoretriever_ocr_v1.server:app",
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
        yield {"base_url": base_url, "token": token}
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def test_health_and_metadata(service: Dict[str, Any]) -> None:
    status, live_payload = http_json(
        "GET", f"{service['base_url']}/v1/health/live", None, {}
    )
    assert status == 200
    assert live_payload == {"status": "live"}

    status, ready_payload = http_json(
        "GET", f"{service['base_url']}/v1/health/ready", None, {}
    )
    assert status == 200
    assert ready_payload == {"status": "ready"}

    headers = {"Authorization": f"Bearer {service['token']}"}
    status, metadata_payload = http_json(
        "GET", f"{service['base_url']}/v1/metadata", None, headers
    )
    assert status == 200
    assert metadata_payload is not None
    model_info = metadata_payload.get("modelInfo")
    assert isinstance(model_info, list)
    assert len(model_info) == 1
    info = model_info[0]
    assert info.get("shortName") == MODEL_SHORT_NAME
    assert info.get("version")
    assert info.get("id")


def test_infer_contract(service: Dict[str, Any]) -> None:
    headers = {"Authorization": f"Bearer {service['token']}"}
    payload = {
        "input": [
            {"type": "image_url", "url": IMAGE_DATA_URL},
            {"type": "image_url", "url": IMAGE_DATA_URL},
        ],
        "merge_levels": ["paragraph", "paragraph"],
    }
    status, body = http_json(
        "POST",
        f"{service['base_url']}/v1/infer",
        payload,
        headers,
        timeout=10.0,
    )
    assert status == 200
    assert body is not None
    results = body.get("data")
    assert isinstance(results, list)
    assert len(results) == len(payload["input"])
    first = results[0]
    assert isinstance(first, dict)
    detections = first.get("text_detections")
    assert isinstance(detections, list)
    assert detections, "expected at least one detection"
    detection = detections[0]
    bbox = detection.get("bounding_box")
    assert isinstance(bbox, dict)
    points = bbox.get("points")
    assert isinstance(points, list)
    assert len(points) == 4
    for point in points:
        assert "x" in point and "y" in point
        assert 0.0 <= float(point["x"]) <= 1.0
        assert 0.0 <= float(point["y"]) <= 1.0
    prediction = detection.get("text_prediction")
    assert isinstance(prediction, dict)
    assert isinstance(prediction.get("text"), str)
    confidence = prediction.get("confidence")
    assert isinstance(confidence, (float, int))
    assert 0.0 <= float(confidence) <= 1.0
