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
MODEL_ID = "nemoretriever-table-structure-v1"
IMAGE_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
LABELS = ["border", "cell", "row", "column", "header"]


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
        if status == 200 and payload is not None and payload.get("ready") is True:
            return
        time.sleep(0.5)
    raise RuntimeError("Service failed to become ready")


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Dict[str, Any]]:
    http_port = find_free_port()
    grpc_port = find_free_port()
    triton_http_port = find_free_port()
    triton_metrics_port = find_free_port()
    metrics_port = find_free_port()
    token = "test-token"

    fake_triton = tmp_path_factory.mktemp("triton_bin") / "tritonserver"
    fake_triton.write_text("#!/usr/bin/env bash\nwhile true; do sleep 3600; done\n")
    fake_triton.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "NIM_HTTP_API_PORT": str(http_port),
            "NIM_GRPC_API_PORT": str(grpc_port),
            "NIM_TRITON_HTTP_PORT": str(triton_http_port),
            "NIM_TRITON_METRICS_PORT": str(triton_metrics_port),
            "NIM_METRICS_PORT": str(metrics_port),
            "TRITON_MODEL_REPOSITORY": str(
                PROJECT_ROOT / "triton" / "model_repository"
            ),
            "TRITON_SERVER_BIN": str(fake_triton),
            "ENABLE_MOCK_INFERENCE": "1",
            "NIM_TRITON_MAX_BATCH_SIZE": "4",
            "NVIDIA_API_KEY": token,
            "NGC_API_KEY": token,
            "NIM_REQUIRE_AUTH": "1",
        }
    )
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        os.pathsep.join([str(STUB_PATH), existing_path])
        if existing_path
        else str(STUB_PATH)
    )
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "oim_nemoretriever_table_structure_v1.server:app",
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
    headers = {"Authorization": f"Bearer {service['token']}"}
    status, live_payload = http_json(
        "GET", f"{service['base_url']}/v1/health/live", None, headers
    )
    assert status == 200
    assert live_payload == {"live": True}

    status, ready_payload = http_json(
        "GET", f"{service['base_url']}/v1/health/ready", None, headers
    )
    assert status == 200
    assert ready_payload == {"ready": True}

    status, metadata_payload = http_json(
        "GET", f"{service['base_url']}/v1/metadata", None, headers
    )
    assert status == 200
    assert metadata_payload is not None
    assert metadata_payload.get("id") == MODEL_ID
    model_info = metadata_payload.get("modelInfo")
    assert isinstance(model_info, list) and len(model_info) > 0
    info = model_info[0]
    assert info.get("name") == MODEL_ID
    assert isinstance(info.get("shortName"), str)
    assert isinstance(info.get("maxBatchSize"), int)


def test_infer_contract(service: Dict[str, Any]) -> None:
    headers = {"Authorization": f"Bearer {service['token']}"}
    payload = {
        "input": [
            {"type": "image_url", "url": IMAGE_DATA_URL},
            {"type": "image_url", "url": IMAGE_DATA_URL},
        ]
    }
    status, body = http_json(
        "POST",
        f"{service['base_url']}/v1/infer",
        payload,
        headers=headers,
        timeout=10.0,
    )
    assert status == 200
    assert body is not None
    data = body.get("data")
    assert isinstance(data, list)
    assert len(data) == len(payload["input"])

    first = data[0]
    assert isinstance(first, dict)
    boxes = first.get("bounding_boxes")
    assert isinstance(boxes, dict)
    for label in LABELS:
        assert label in boxes
        assert isinstance(boxes[label], list)
    populated_labels = [label for label, items in boxes.items() if len(items) > 0]
    assert len(populated_labels) > 0
    sample_box = boxes[populated_labels[0]][0]
    assert {"x_min", "y_min", "x_max", "y_max", "confidence"} <= set(sample_box.keys())
    assert 0.0 <= float(sample_box["confidence"]) <= 1.0
    assert 0.0 <= float(sample_box["x_min"]) <= float(sample_box["x_max"]) <= 1.0
    assert 0.0 <= float(sample_box["y_min"]) <= float(sample_box["y_max"]) <= 1.0
