from __future__ import annotations

import logging
from typing import Optional

from .settings import ServiceSettings

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

logger = logging.getLogger(__name__)


def configure_tracer(settings: ServiceSettings) -> Optional["trace.Tracer"]:
    """
    Configure OpenTelemetry tracing and metrics when enabled.
    """
    if not settings.enable_otel and not settings.otel_endpoint:
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
        resource = Resource.create(
            {"service.name": settings.otel_service_name or settings.model_id}
        )
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=settings.otel_endpoint
                    or "http://localhost:4318/v1/traces",
                    insecure=True,
                )
            )
        )
        trace.set_tracer_provider(tracer_provider)

        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=settings.otel_endpoint or "http://localhost:4318/v1/metrics",
                insecure=True,
            )
        )
        meter_provider = MeterProvider(
            resource=resource, metric_readers=[metric_reader]
        )
        otel_metrics.set_meter_provider(meter_provider)
        return trace.get_tracer(settings.otel_service_name or settings.model_id)
    except Exception:
        logger.exception("Failed to configure OTEL")
        return None
