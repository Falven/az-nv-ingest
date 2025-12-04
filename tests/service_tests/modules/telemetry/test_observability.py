# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from az_nv_ingest.azure import observability
from opentelemetry import trace


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


class FakeSpanProcessor:
    def __init__(self, exporter):
        self.exporter = exporter

    def shutdown(self):
        return None

    def force_flush(self, *_, **__):
        return True


class FakeTracerProvider:
    def __init__(self, resource):
        self.resource = resource
        self.span_processors = []

    def add_span_processor(self, processor):
        self.span_processors.append(processor)


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


def test_configure_tracer_is_idempotent(monkeypatch):
    calls = {"exporter": 0}

    def fake_build_trace_exporter(**_kwargs):
        calls["exporter"] += 1
        return FakeOtlpExporter(endpoint="http://collector:4317", insecure=True)

    def fake_batch_processor(exporter):
        return FakeSpanProcessor(exporter)

    holder: dict[str, object] = {}

    def fake_set_tracer_provider(provider):
        holder["provider"] = provider

    def fake_get_tracer_provider():
        return holder.get("provider")

    def fake_get_tracer(name):
        holder["last_tracer"] = name
        return f"tracer:{name}"

    monkeypatch.setattr(observability, "TracerProvider", FakeTracerProvider)
    monkeypatch.setattr(observability, "build_trace_exporter", fake_build_trace_exporter)
    monkeypatch.setattr(observability, "BatchSpanProcessor", fake_batch_processor)
    monkeypatch.setattr(observability, "_TRACER_INITIALIZED", False)
    monkeypatch.setattr(observability.trace, "set_tracer_provider", fake_set_tracer_provider)
    monkeypatch.setattr(observability.trace, "get_tracer_provider", fake_get_tracer_provider)
    monkeypatch.setattr(observability.trace, "get_tracer", fake_get_tracer)

    observability.configure_tracer(env={"OTEL_SERVICE_NAME": "svc"})
    provider = holder["provider"]
    span_processors_first = provider.span_processors

    observability.configure_tracer(env={"OTEL_SERVICE_NAME": "svc"})
    span_processors_second = provider.span_processors

    assert calls["exporter"] == 1
    assert span_processors_first == span_processors_second


def test_configure_tracer_sets_role_name(monkeypatch):
    holder: dict[str, object] = {}

    monkeypatch.setattr(observability, "TracerProvider", FakeTracerProvider)
    monkeypatch.setattr(observability, "build_trace_exporter", lambda **_: FakeOtlpExporter("http://collector", True))
    monkeypatch.setattr(observability, "BatchSpanProcessor", lambda exporter: FakeSpanProcessor(exporter))
    monkeypatch.setattr(observability, "_TRACER_INITIALIZED", False)
    monkeypatch.setattr(observability.trace, "set_tracer_provider", lambda provider: holder.update({"provider": provider}))
    monkeypatch.setattr(observability.trace, "get_tracer_provider", lambda: holder.get("provider"))
    monkeypatch.setattr(observability.trace, "get_tracer", lambda name: f"tracer:{name}")

    env = {
        "OTEL_SERVICE_NAME": "svc",
        "APPLICATIONINSIGHTS_ROLE_NAME": "ingest-role",
        "HOSTNAME": "pod-1",
    }

    observability.configure_tracer(env=env)
    provider = holder["provider"]
    attributes = provider.resource.attributes

    assert attributes["service.namespace"] == "ingest-role"
    assert attributes["service.instance.id"] == "pod-1"
