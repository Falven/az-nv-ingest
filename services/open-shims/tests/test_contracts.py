from __future__ import annotations

import json
from typing import Dict
from typing import List

import pytest
from fastapi.testclient import TestClient

from open_shims.app import ShimKind, create_app
from open_shims.constants import PARSE_ARGUMENTS, sample_chat_completion_request, sample_infer_request


def make_client(kind: ShimKind) -> TestClient:
    return TestClient(create_app(kind))


@pytest.mark.parametrize(
    ("kind", "expected_classes"),
    [
        (ShimKind.PAGE_ELEMENTS, ["table", "chart", "title", "infographic"]),
        (
            ShimKind.GRAPHIC_ELEMENTS,
            ["chart_title", "x_title", "y_title", "xlabel", "ylabel", "other", "legend_label", "legend_title", "mark_label", "value_label"],
        ),
        (ShimKind.TABLE_STRUCTURE, ["border", "cell", "row", "column", "header"]),
    ],
)
def test_yolox_shim_contract(kind: ShimKind, expected_classes: List[str]) -> None:
    client = make_client(kind)
    resp = client.post("/v1/infer", json=sample_infer_request())
    assert resp.status_code == 200
    body: Dict[str, object] = resp.json()
    assert body.get("model") == "open-shim"
    data = body.get("data")
    assert isinstance(data, list) and len(data) == 1
    detections = data[0]
    boxes = detections.get("bounding_boxes")
    assert isinstance(boxes, dict)
    for cls in expected_classes:
        assert cls in boxes


def test_ocr_shim_contract() -> None:
    client = make_client(ShimKind.OCR)
    resp = client.post("/v1/infer", json=sample_infer_request())
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("model") == "open-shim"
    data = body.get("data")
    assert isinstance(data, list) and len(data) == 1
    text_detections = data[0].get("text_detections")
    assert isinstance(text_detections, list) and len(text_detections) >= 1
    detection = text_detections[0]
    assert "text_prediction" in detection
    assert "bounding_box" in detection
    assert "points" in detection["bounding_box"]


def test_parse_shim_contract() -> None:
    client = make_client(ShimKind.PARSE)
    request_body = sample_chat_completion_request()
    resp = client.post("/v1/chat/completions", json=request_body)
    assert resp.status_code == 200
    body = resp.json()
    choices = body.get("choices")
    assert isinstance(choices, list) and len(choices) == 1
    tool_calls = choices[0]["message"]["tool_calls"]
    assert isinstance(tool_calls, list) and len(tool_calls) == 1
    arguments = tool_calls[0]["function"]["arguments"]
    parsed_arguments = json.loads(arguments)
    assert parsed_arguments == PARSE_ARGUMENTS


@pytest.mark.parametrize("kind", list(ShimKind))
def test_health_and_sample(kind: ShimKind) -> None:
    client = make_client(kind)
    health = client.get("/health")
    assert health.status_code == 200
    ready = client.get("/v1/health/ready")
    assert ready.status_code == 200

    sample = client.get("/sample")
    assert sample.status_code == 200
    payload = sample.json()
    if kind == ShimKind.PARSE:
        assert "messages" in payload
    else:
        assert "input" in payload
