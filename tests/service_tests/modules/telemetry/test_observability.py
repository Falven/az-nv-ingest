# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from az_nv_ingest.azure import observability


class FakeAzureMonitorExporter:
    def __init__(self, connection_string=None, credential=None):
        self.connection_string = connection_string
        self.credential = credential

    @classmethod
    def from_connection_string(cls, connection_string, credential=None):
        return cls(connection_string=connection_string, credential=credential)


class FakeOtlpExporter:
    def __init__(self, endpoint, insecure, headers=None):
        self.endpoint = endpoint
        self.insecure = insecure
        self.headers = headers or {}


def test_build_trace_exporter_prefers_app_insights(monkeypatch):
    monkeypatch.setattr(observability, "AzureMonitorTraceExporter", FakeAzureMonitorExporter)

    env = {"APPLICATIONINSIGHTS_CONNECTION_STRING": "InstrumentationKey=test-key"}

    exporter = observability.build_trace_exporter(env=env)

    assert isinstance(exporter, FakeAzureMonitorExporter)
    assert exporter.connection_string == "InstrumentationKey=test-key"


def test_build_trace_exporter_defaults_to_otlp(monkeypatch):
    monkeypatch.setattr(observability, "AzureMonitorTraceExporter", FakeAzureMonitorExporter)
    monkeypatch.setattr(observability, "OTLPSpanExporter", FakeOtlpExporter)

    env = {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "https://collector:4317",
        "OTEL_EXPORTER_OTLP_HEADERS": "api-key=value",
    }

    exporter = observability.build_trace_exporter(env=env)

    assert isinstance(exporter, FakeOtlpExporter)
    assert exporter.endpoint == "https://collector:4317"
    assert exporter.insecure is False
    assert exporter.headers == {"api-key": "value"}
