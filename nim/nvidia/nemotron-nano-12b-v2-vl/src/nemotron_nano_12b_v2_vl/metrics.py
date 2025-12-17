from __future__ import annotations

import logging
from contextlib import contextmanager

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
    buckets=(0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
)
INFLIGHT_GAUGE = Gauge(
    "nim_requests_in_flight", "Concurrent in-flight requests", ["protocol", "endpoint"]
)

_METRICS_STARTED = False


def start_metrics_server(settings: ServiceSettings) -> None:
    """
    Start the Prometheus metrics HTTP server when permitted.

    Args:
        settings: Service configuration controlling metrics exposure.
    """
    global _METRICS_STARTED
    if _METRICS_STARTED:
        return
    if settings.metrics_port is None:
        logger.info("Metrics server disabled because no port is configured.")
        _METRICS_STARTED = True
        return
    if settings.auth_required:
        logger.info(
            "Skipping standalone metrics server because authentication is enabled; use /metrics with auth."
        )
        _METRICS_STARTED = True
        return
    start_http_server(settings.metrics_port)
    _METRICS_STARTED = True
    logger.info("Metrics server listening on port %s", settings.metrics_port)


def render_metrics() -> Response:
    """
    Render collected metrics for FastAPI endpoints.

    Returns:
        Response with Prometheus-formatted metrics.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def observe_http_request(endpoint: str, status: str, duration_seconds: float) -> None:
    """
    Record counters and latency for an HTTP request.

    Args:
        endpoint: Path template associated with the request.
        status: HTTP status code label.
        duration_seconds: Elapsed time in seconds.
    """
    REQUEST_COUNTER.labels(protocol="http", endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(protocol="http", endpoint=endpoint).observe(duration_seconds)


@contextmanager
def track_inflight(endpoint: str):
    """
    Context manager that tracks in-flight HTTP requests.

    Args:
        endpoint: Endpoint label used in metrics.
    """
    INFLIGHT_GAUGE.labels(protocol="http", endpoint=endpoint).inc()
    try:
        yield
    finally:
        INFLIGHT_GAUGE.labels(protocol="http", endpoint=endpoint).dec()
