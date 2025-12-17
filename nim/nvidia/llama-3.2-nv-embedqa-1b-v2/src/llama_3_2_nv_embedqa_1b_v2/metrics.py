from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator, Literal

from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
)

from .settings import ServiceSettings

logger = logging.getLogger(__name__)

REQUEST_COUNTER = Counter(
    "nim_requests_total", "Total requests served", ["protocol", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "nim_request_latency_seconds",
    "Latency for requests",
    ["protocol", "endpoint"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
)
INFLIGHT_GAUGE = Gauge(
    "nim_requests_in_flight", "Concurrent in-flight requests", ["protocol", "endpoint"]
)

_METRICS_STARTED = False


def start_metrics_server(settings: ServiceSettings) -> None:
    """
    Start a Prometheus metrics HTTP exporter when permitted by auth policy.

    Args:
        settings: Service settings governing the metrics port and auth rules.
    """
    global _METRICS_STARTED
    if _METRICS_STARTED or settings.metrics_port is None:
        return
    if settings.auth_required:
        _METRICS_STARTED = True
        logger.info(
            "Skipping standalone metrics server because authentication is required."
        )
        return
    start_http_server(settings.metrics_port)
    _METRICS_STARTED = True
    logger.info("Metrics server listening on port %s", settings.metrics_port)


def metrics_response() -> Response:
    """
    Render collected metrics for FastAPI endpoints.

    Returns:
        HTTP response with Prometheus-formatted metrics.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def record_request(
    protocol: Literal["http", "grpc"], endpoint: str, status: str, started_at: float
) -> None:
    """
    Record counters and latency for a completed request.

    Args:
        protocol: Protocol label (``"http"`` or ``"grpc"``).
        endpoint: Resolved endpoint path.
        status: HTTP/gRPC status code as a string.
        started_at: Monotonic time captured at request start.
    """
    duration = time.perf_counter() - started_at
    REQUEST_COUNTER.labels(protocol=protocol, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(protocol=protocol, endpoint=endpoint).observe(duration)


@contextmanager
def track_inflight(protocol: Literal["http", "grpc"], endpoint: str) -> Iterator[None]:
    """
    Context manager tracking in-flight request counts.

    Args:
        protocol: Protocol label (``"http"`` or ``"grpc"``).
        endpoint: Resolved endpoint path.
    """
    INFLIGHT_GAUGE.labels(protocol=protocol, endpoint=endpoint).inc()
    try:
        yield
    finally:
        INFLIGHT_GAUGE.labels(protocol=protocol, endpoint=endpoint).dec()
