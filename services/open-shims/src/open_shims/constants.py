from __future__ import annotations

import copy
import json
from typing import Any
from typing import Dict
from typing import List

SAMPLE_IMAGE_URL = "data:image/png;base64,ZmFrZV9pbWFnZQ=="


def _bbox(x_min: float, y_min: float, x_max: float, y_max: float, confidence: float) -> Dict[str, float]:
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "confidence": confidence,
    }


PAGE_BOXES: Dict[str, List[Dict[str, float]]] = {
    "table": [_bbox(0.08, 0.12, 0.62, 0.36, 0.93)],
    "chart": [_bbox(0.55, 0.45, 0.9, 0.82, 0.88)],
    "title": [_bbox(0.1, 0.05, 0.6, 0.1, 0.74)],
    "infographic": [],
}

GRAPHIC_BOXES: Dict[str, List[Dict[str, float]]] = {
    "chart_title": [_bbox(0.12, 0.08, 0.72, 0.16, 0.91)],
    "x_title": [_bbox(0.32, 0.88, 0.78, 0.93, 0.86)],
    "y_title": [_bbox(0.08, 0.28, 0.13, 0.78, 0.82)],
    "xlabel": [_bbox(0.34, 0.82, 0.82, 0.86, 0.83)],
    "ylabel": [_bbox(0.14, 0.3, 0.18, 0.76, 0.81)],
    "other": [],
    "legend_label": [_bbox(0.72, 0.22, 0.88, 0.34, 0.79)],
    "legend_title": [_bbox(0.72, 0.18, 0.88, 0.21, 0.8)],
    "mark_label": [],
    "value_label": [],
}

TABLE_BOXES: Dict[str, List[Dict[str, float]]] = {
    "border": [_bbox(0.08, 0.18, 0.86, 0.74, 0.95)],
    "cell": [
        _bbox(0.1, 0.2, 0.45, 0.28, 0.93),
        _bbox(0.45, 0.2, 0.82, 0.28, 0.91),
        _bbox(0.1, 0.28, 0.45, 0.36, 0.9),
        _bbox(0.45, 0.28, 0.82, 0.36, 0.89),
    ],
    "row": [
        _bbox(0.1, 0.2, 0.82, 0.28, 0.9),
        _bbox(0.1, 0.28, 0.82, 0.36, 0.88),
    ],
    "column": [
        _bbox(0.1, 0.2, 0.45, 0.72, 0.86),
        _bbox(0.45, 0.2, 0.82, 0.72, 0.84),
    ],
    "header": [_bbox(0.1, 0.18, 0.82, 0.24, 0.92)],
}

OCR_TEXT_DETECTIONS: List[Dict[str, Any]] = [
    {
        "text_prediction": {"text": "Sample heading", "confidence": 0.98},
        "bounding_box": {
            "points": [
                {"x": 0.08, "y": 0.06},
                {"x": 0.62, "y": 0.06},
                {"x": 0.62, "y": 0.12},
                {"x": 0.08, "y": 0.12},
            ]
        },
    },
    {
        "text_prediction": {"text": "Quarterly revenue increased by 12%.", "confidence": 0.95},
        "bounding_box": {
            "points": [
                {"x": 0.1, "y": 0.2},
                {"x": 0.74, "y": 0.2},
                {"x": 0.74, "y": 0.26},
                {"x": 0.1, "y": 0.26},
            ]
        },
    },
]

PARSE_ARGUMENTS: List[Dict[str, Any]] = [
    {
        "type": "Title",
        "bbox": {"xmin": 0.08, "ymin": 0.05, "xmax": 0.7, "ymax": 0.12},
        "text": "Quarterly Business Review",
    },
    {
        "type": "Text",
        "bbox": {"xmin": 0.1, "ymin": 0.16, "xmax": 0.82, "ymax": 0.34},
        "text": "Revenue grew by 12% quarter-over-quarter with strong performance in cloud services.",
    },
    {
        "type": "Table",
        "bbox": {"xmin": 0.12, "ymin": 0.36, "xmax": 0.84, "ymax": 0.62},
        "text": "\\begin{tabular}{lrr}Region & Q1 & Q2 \\\\ North & 10 & 12 \\\\ West & 8 & 9 \\end{tabular}",
    },
    {
        "type": "Picture",
        "bbox": {"xmin": 0.14, "ymin": 0.66, "xmax": 0.86, "ymax": 0.9},
        "text": "Bar chart summarizing regional growth.",
    },
]


def build_bounding_box_payload(boxes: Dict[str, List[Dict[str, float]]], batch_size: int) -> Dict[str, Any]:
    return {
        "model": "open-shim",
        "data": [copy.deepcopy({"bounding_boxes": boxes}) for _ in range(batch_size)],
    }


def build_ocr_payload(text_detections: List[Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
    return {"model": "open-shim", "data": [copy.deepcopy({"text_detections": text_detections}) for _ in range(batch_size)]}


def build_parse_payload(arguments: List[Dict[str, Any]]) -> Dict[str, Any]:
    arguments_json = json.dumps(arguments)
    return {
        "id": "shim-chatcmpl-1",
        "object": "chat.completion",
        "model": "nemotron-parse-1.1-shim",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "shim-function",
                            "type": "function",
                            "function": {"name": "document_parse", "arguments": arguments_json},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def sample_infer_request() -> Dict[str, Any]:
    return {"input": [{"type": "image_url", "url": SAMPLE_IMAGE_URL}]}


def sample_chat_completion_request() -> Dict[str, Any]:
    return {
        "model": "nemotron-parse-1.1-shim",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": SAMPLE_IMAGE_URL}}],
            }
        ],
    }
