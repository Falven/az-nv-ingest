from __future__ import annotations

import logging
from typing import Mapping, Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except Exception:  # pragma: no cover - optional dependency
    trace = None
    otel_metrics = None
    OTLPMetricExporter = None
    OTLPSpanExporter = None
    MeterProvider = None
    PeriodicExportingMetricReader = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None


def _endpoint(base: str | None, suffix: str) -> str:
    host = base or "http://localhost:4318"
    trimmed = host.rstrip("/")
    return f"{trimmed}/v1/{suffix}"


def configure_tracer(
    *,
    enabled: bool,
    service_name: str,
    otel_endpoint: str | None,
    resource_attributes: Optional[Mapping[str, str]] = None,
    enable_metrics: bool = True,
) -> Optional["trace.Tracer"]:
    """
    Configure OpenTelemetry tracing and metrics when requested.
    """
    if not enabled and not otel_endpoint:
        return None
    if (
        trace is None
        or otel_metrics is None
        or OTLPMetricExporter is None
        or OTLPSpanExporter is None
        or MeterProvider is None
        or PeriodicExportingMetricReader is None
        or Resource is None
        or TracerProvider is None
        or BatchSpanProcessor is None
    ):
        logger.warning(
            "OpenTelemetry requested but dependencies are unavailable; skipping setup."
        )
        return None
    try:
        attributes = {"service.name": service_name, **(resource_attributes or {})}
        resource = Resource.create(attributes)
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=_endpoint(otel_endpoint, "traces"),
                    insecure=True,
                )
            )
        )
        trace.set_tracer_provider(tracer_provider)

        if enable_metrics:
            metric_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=_endpoint(otel_endpoint, "metrics"),
                    insecure=True,
                )
            )
            meter_provider = MeterProvider(
                resource=resource, metric_readers=[metric_reader]
            )
            otel_metrics.set_meter_provider(meter_provider)
        return trace.get_tracer(service_name)
    except Exception:
        logger.exception("Failed to configure OpenTelemetry")
        return None
