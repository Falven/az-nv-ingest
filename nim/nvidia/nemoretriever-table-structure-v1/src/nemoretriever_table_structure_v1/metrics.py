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
    "nim_requests_total",
    "Total requests served",
    ["protocol", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "nim_request_latency_seconds",
    "Latency for requests",
    ["protocol", "endpoint"],
    buckets=(
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
    ),
)
INFLIGHT_GAUGE = Gauge(
    "nim_requests_in_flight",
    "Concurrent in-flight requests",
    ["protocol", "endpoint"],
)

_METRICS_STARTED = False


def start_metrics_server(settings: ServiceSettings) -> None:
    """
    Start the standalone Prometheus metrics server when allowed.

    Metrics server is disabled when authentication is required; use the
    FastAPI /metrics endpoint in that scenario.

    Args:
        settings: Service configuration.
    """
    global _METRICS_STARTED
    if _METRICS_STARTED:
        return
    if settings.auth_required or settings.metrics_port is None:
        _METRICS_STARTED = True
        if settings.metrics_port is not None:
            logger.info(
                "Metrics port is disabled when authentication is required; rely on /metrics.",
            )
        return
    start_http_server(settings.metrics_port)
    _METRICS_STARTED = True
    logger.info("Metrics server listening on port %s", settings.metrics_port)


def render_metrics() -> Response:
    """
    Render collected metrics for FastAPI endpoints.

    Returns:
        HTTP response containing Prometheus-formatted metrics.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def record_request(
    endpoint: str, protocol: Literal["http", "grpc"], status: str, start_time: float
) -> None:
    """
    Record request counters and latency for an endpoint.

    Args:
        endpoint: Logical endpoint name.
        protocol: Protocol label.
        status: Status label (e.g., success, error).
        start_time: Perf counter captured at request start.
    """
    REQUEST_COUNTER.labels(protocol=protocol, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(protocol=protocol, endpoint=endpoint).observe(
        time.perf_counter() - start_time
    )


@contextmanager
def track_inflight(protocol: Literal["http", "grpc"], endpoint: str) -> Iterator[None]:
    """
    Track in-flight requests with a Prometheus gauge.

    Args:
        protocol: Protocol label.
        endpoint: Endpoint label.

    Yields:
        Execution context for the active request.
    """
    INFLIGHT_GAUGE.labels(protocol=protocol, endpoint=endpoint).inc()
    try:
        yield
    finally:
        INFLIGHT_GAUGE.labels(protocol=protocol, endpoint=endpoint).dec()
