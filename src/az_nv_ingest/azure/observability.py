# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for opt-in Application Insights + OTLP tracing configuration."""

from __future__ import annotations

import os
from typing import Dict, Mapping, MutableMapping, Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

try:  # Azure Monitor is optional until configured
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
except ImportError:  # pragma: no cover - import guarded by dependency management
    AzureMonitorTraceExporter = None  # type: ignore

_TRACER_INITIALIZED = False


def _coalesce_env(env: Mapping[str, str], keys: tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = env.get(key)
        if value:
            return value.strip()
    return None


def resolve_app_insights_connection_string(env: Mapping[str, str] | None = None) -> Optional[str]:
    env = env or os.environ

    connection_string = _coalesce_env(
        env,
        (
            "APPLICATIONINSIGHTS_CONNECTION_STRING",
            "APPINSIGHTS_CONNECTION_STRING",
        ),
    )
    if connection_string:
        return connection_string

    instrumentation_key = _coalesce_env(
        env,
        (
            "APPLICATIONINSIGHTS_INSTRUMENTATION_KEY",
            "APPINSIGHTS_INSTRUMENTATIONKEY",
            "APPINSIGHTS_INSTRUMENTATION_KEY",
        ),
    )
    if instrumentation_key:
        return f"InstrumentationKey={instrumentation_key}"

    return None


def parse_otlp_headers(raw_headers: str | None) -> Dict[str, str]:
    if not raw_headers:
        return {}

    parsed: Dict[str, str] = {}
    for pair in raw_headers.split(","):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            parsed[key] = value
    return parsed


def _is_insecure_endpoint(endpoint: str) -> bool:
    endpoint = endpoint.strip()
    if endpoint.startswith("https://"):
        return False
    return True


def build_trace_exporter(
    env: Mapping[str, str] | None = None,
    connection_string: str | None = None,
    otlp_endpoint: str | None = None,
    otlp_headers: Mapping[str, str] | None = None,
    credential=None,
):
    env = env or os.environ

    resolved_connection_string = connection_string or resolve_app_insights_connection_string(env)
    if resolved_connection_string:
        if AzureMonitorTraceExporter is None:
            raise RuntimeError("azure-monitor-opentelemetry-exporter is required for App Insights telemetry")
        return AzureMonitorTraceExporter.from_connection_string(
            resolved_connection_string,
            credential=credential,
        )

    endpoint = otlp_endpoint or env.get("OTEL_EXPORTER_OTLP_ENDPOINT", "otel-collector:4317")
    headers = otlp_headers or parse_otlp_headers(env.get("OTEL_EXPORTER_OTLP_HEADERS", ""))

    exporter_kwargs: Dict[str, object] = {
        "endpoint": endpoint,
        "insecure": _is_insecure_endpoint(endpoint),
    }
    if headers:
        exporter_kwargs["headers"] = headers

    return OTLPSpanExporter(**exporter_kwargs)


def configure_tracer(
    service_name: str = "nv-ingest",
    env: MutableMapping[str, str] | None = None,
    connection_string: str | None = None,
    otlp_endpoint: str | None = None,
    otlp_headers: Mapping[str, str] | None = None,
    credential=None,
) -> trace.Tracer:
    env = env or os.environ
    global _TRACER_INITIALIZED

    current_provider = trace.get_tracer_provider()
    if _TRACER_INITIALIZED:
        return trace.get_tracer(env.get("OTEL_SERVICE_NAME", service_name))

    if isinstance(current_provider, TracerProvider) and (
        getattr(current_provider, "_az_nv_ingest_configured", False) or getattr(current_provider, "span_processors", [])
    ):
        return trace.get_tracer(env.get("OTEL_SERVICE_NAME", service_name))

    exporter = build_trace_exporter(
        env=env,
        connection_string=connection_string,
        otlp_endpoint=otlp_endpoint,
        otlp_headers=otlp_headers,
        credential=credential,
    )

    resource_attributes = {"service.name": env.get("OTEL_SERVICE_NAME", service_name)}
    role_name = env.get("APPLICATIONINSIGHTS_ROLE_NAME")
    if role_name:
        resource_attributes["service.namespace"] = role_name
    instance_id = env.get("OTEL_SERVICE_INSTANCE_ID") or env.get("HOSTNAME")
    if instance_id:
        resource_attributes["service.instance.id"] = instance_id

    provider = TracerProvider(resource=Resource(attributes=resource_attributes))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    provider._az_nv_ingest_configured = True  # type: ignore[attr-defined]
    trace.set_tracer_provider(provider)
    _TRACER_INITIALIZED = True
    return trace.get_tracer(env.get("OTEL_SERVICE_NAME", service_name))
