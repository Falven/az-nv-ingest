from __future__ import annotations

import logging
from typing import Optional

from .settings import ServiceSettings

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except Exception:  # pragma: no cover - optional dependency
    trace = None
    OTLPSpanExporter = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None

logger = logging.getLogger(__name__)


def configure_tracer(settings: ServiceSettings) -> Optional["trace.Tracer"]:
    """
    Initialize an OpenTelemetry tracer when an OTEL endpoint is configured.

    Args:
        settings: Service settings containing OTEL endpoint and service name.

    Returns:
        Configured tracer instance when OTEL is enabled and dependencies exist;
        otherwise ``None``.
    """
    if settings.otel_endpoint is None:
        return None
    if (
        trace is None
        or OTLPSpanExporter is None
        or TracerProvider is None
        or BatchSpanProcessor is None
    ):
        logger.warning(
            "OpenTelemetry requested but dependencies are unavailable; skipping OTEL setup."
        )
        return None
    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(settings.otel_service_name)
