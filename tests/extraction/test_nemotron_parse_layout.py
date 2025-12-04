import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide a lightweight psutil stub to avoid pulling the full dependency into unit tests.
if "psutil" not in sys.modules:
    psutil_stub = ModuleType("psutil")
    psutil_stub.__spec__ = ModuleSpec("psutil", loader=None)
    psutil_stub.virtual_memory = lambda: SimpleNamespace(total=0)
    sys.modules["psutil"] = psutil_stub

from nv_ingest.framework.util.patches.nemotron_parse import (  # noqa: E402
    NemotronParseModelInterface,
    annotate_table_continuity,
)
from nv_ingest.framework.orchestration.ray.util.pipeline.stage_builders import get_nim_service  # noqa: E402
from nv_ingest_api.internal.enums.common import ContentDescriptionEnum, ContentTypeEnum, TableFormatEnum  # noqa: E402


def _base_table_entry(page_number: int, layout_id):
    return [
        ContentTypeEnum.STRUCTURED,
        {
            "content_metadata": {
                "type": ContentTypeEnum.STRUCTURED,
                "description": ContentDescriptionEnum.PDF_TABLE,
                "page_number": page_number,
                "hierarchy": {"page_count": 3, "page": page_number, "line": -1, "span": -1},
                "subtype": ContentTypeEnum.TABLE,
            },
            "table_metadata": {
                "caption": "",
                "table_format": TableFormatEnum.IMAGE,
                "table_content": "dummy",
                "table_content_format": TableFormatEnum.PSEUDO_MARKDOWN,
                "table_location": (0, 0, 10, 10),
                "table_location_max_dimensions": (1024, 1280),
                "custom_content": {"layout_id": layout_id},
            },
        },
        f"uuid-{page_number}",
    ]


@pytest.fixture
def golden_continuity():
    fixture_path = Path(__file__).parent.parent / "data" / "golden_multi_page_tables.json"
    return json.loads(fixture_path.read_text())


def test_annotate_table_continuity_matches_golden(golden_continuity):
    extracted = [
        _base_table_entry(0, "table-1"),
        _base_table_entry(1, "table-1"),
        _base_table_entry(2, None),
    ]

    annotate_table_continuity(extracted)
    observed = [deepcopy(item[1]["table_metadata"]["custom_content"]) for item in extracted]

    assert observed == golden_continuity


def test_model_interface_parses_output_layout():
    interface = NemotronParseModelInterface()
    response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {
                            "type": "output_layout",
                            "layout": [
                                {
                                    "type": "Table",
                                    "text": "LATEX",
                                    "bbox": [0.1, 0.2, 0.3, 0.4],
                                    "page": 0,
                                    "layout_id": "table-123",
                                }
                            ],
                        }
                    ]
                }
            }
        ]
    }

    parsed = interface.parse_output(response, protocol="http")
    assert len(parsed) == 1
    assert parsed[0][0]["layout_id"] == "table-123"
    assert parsed[0][0]["bbox"] == (0.1, 0.2, 0.3, 0.4)
    assert parsed[0][0]["page"] == 0


def test_get_nim_service_prefers_shim(monkeypatch):
    monkeypatch.setenv("NEMORETRIEVER_PARSE_HTTP_ENDPOINT", "http://nim.example/v1")
    monkeypatch.setenv("NEMORETRIEVER_PARSE_SHIM_HTTP_ENDPOINT", "http://shim.example/v1")
    grpc, http, _, protocol = get_nim_service("nemoretriever_parse")

    assert http == "http://shim.example/v1"
    assert protocol == "http"
    assert grpc == ""
