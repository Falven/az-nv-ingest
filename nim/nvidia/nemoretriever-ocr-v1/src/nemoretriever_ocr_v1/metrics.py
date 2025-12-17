from __future__ import annotations

import logging
from typing import Literal

from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
    start_http_server,
)

from .settings import ServiceSettings

logger = logging.getLogger(__name__)

REQUEST_COUNTER = Counter(
    "nim_ocr_requests_total", "Total OCR inference requests", ["protocol"]
)
REQUEST_LATENCY = Histogram(
    "nim_ocr_request_latency_seconds",
    "Latency for OCR inference requests",
    ["protocol"],
)
BATCH_SIZE = Histogram("nim_ocr_batch_size", "Observed OCR batch sizes", ["protocol"])

_METRICS_STARTED = False


def start_metrics_server(settings: ServiceSettings) -> None:
    """
    Start a Prometheus metrics endpoint unless auth tokens are configured.

    Args:
        settings: Service settings controlling the metrics port and auth policy.
    """
    global _METRICS_STARTED
    if _METRICS_STARTED:
        return
    if settings.auth_required:
        _METRICS_STARTED = True
        logger.info(
            "Metrics port is disabled when authentication is required; use /metrics endpoints instead.",
        )
        return
    start_http_server(settings.metrics_port)
    _METRICS_STARTED = True
    logger.info("Started metrics server on :%d/metrics", settings.metrics_port)


def render_metrics() -> Response:
    """
    Render collected metrics for FastAPI endpoints.

    Returns:
        HTTP response containing Prometheus-formatted metrics.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def observe_request(
    protocol: Literal["http", "grpc"], batch_size: int, duration_seconds: float
) -> None:
    """
    Record request counters and histograms for a completed inference call.

    Args:
        protocol: Protocol label for the request.
        batch_size: Number of items processed in the request.
        duration_seconds: Duration of the request in seconds.
    """
    REQUEST_COUNTER.labels(protocol).inc()
    REQUEST_LATENCY.labels(protocol).observe(duration_seconds)
    BATCH_SIZE.labels(protocol).observe(float(batch_size))
