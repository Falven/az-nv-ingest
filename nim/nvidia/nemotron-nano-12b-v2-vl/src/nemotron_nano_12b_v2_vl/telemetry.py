from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry import trace

from .settings import ServiceSettings

logger = logging.getLogger(__name__)


def configure_tracer(settings: ServiceSettings) -> Optional["trace.Tracer"]:
    """
    Configure OpenTelemetry tracing and metrics when enabled.

    Args:
        settings: Service configuration with OTEL options.

    Returns:
        Tracer instance when configured; otherwise ``None``.
    """
    if not settings.enable_otel:
        return None
    try:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ModuleNotFoundError:
        logger.info("OpenTelemetry dependencies not installed; skipping tracing.")
        return None
    try:
        resource = Resource.create(
            {"service.name": settings.otel_service_name or settings.served_model_name}
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
        return trace.get_tracer(
            settings.otel_service_name or settings.served_model_name
        )
    except Exception:  # pragma: no cover - surfaced to logs
        logger.exception("Failed to configure OpenTelemetry")
        return None
