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
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
)
INFLIGHT_GAUGE = Gauge(
    "nim_requests_in_flight",
    "Concurrent in-flight requests",
    ["protocol", "endpoint"],
)
OCR_REQUEST_COUNTER = Counter(
    "nim_ocr_requests_total",
    "Total OCR inference requests",
    ["protocol"],
)
OCR_REQUEST_LATENCY = Histogram(
    "nim_ocr_request_latency_seconds",
    "Latency for OCR inference requests",
    ["protocol"],
)
OCR_BATCH_SIZE = Histogram(
    "nim_ocr_batch_size",
    "Observed OCR batch sizes",
    ["protocol"],
)

_METRICS_STARTED = False


def start_metrics_server(
    metrics_port: int | None,
    auth_required: bool,
) -> None:
    """
    Start a standalone Prometheus metrics server when safe to expose.
    """
    global _METRICS_STARTED
    if _METRICS_STARTED or metrics_port is None:
        return
    if auth_required:
        _METRICS_STARTED = True
        logger.info(
            "Skipping standalone metrics server because authentication is required; use the /metrics endpoint."
        )
        return
    start_http_server(metrics_port)
    _METRICS_STARTED = True
    logger.info("Metrics server listening on port %s", metrics_port)


def metrics_response() -> Response:
    """
    Render collected metrics for HTTP exposure.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def record_request(
    protocol: Literal["http", "grpc"],
    endpoint: str,
    status: str,
    started_at: float,
) -> None:
    """
    Record counters and latency for a completed request using a start time.
    """
    duration = time.perf_counter() - started_at
    REQUEST_COUNTER.labels(protocol=protocol, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(protocol=protocol, endpoint=endpoint).observe(duration)


def observe_http_request(endpoint: str, status: str, duration_seconds: float) -> None:
    """
    Record counters and latency when duration is precomputed.
    """
    REQUEST_COUNTER.labels(protocol="http", endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(protocol="http", endpoint=endpoint).observe(duration_seconds)


def observe_ocr_request(
    protocol: Literal["http", "grpc"], batch_size: int, duration_seconds: float
) -> None:
    """
    Record OCR-specific counters and histograms.
    """
    OCR_REQUEST_COUNTER.labels(protocol).inc()
    OCR_REQUEST_LATENCY.labels(protocol).observe(duration_seconds)
    OCR_BATCH_SIZE.labels(protocol).observe(float(batch_size))


@contextmanager
def track_inflight(protocol: Literal["http", "grpc"], endpoint: str) -> Iterator[None]:
    """
    Track in-flight requests with a Prometheus gauge.
    """
    INFLIGHT_GAUGE.labels(protocol=protocol, endpoint=endpoint).inc()
    try:
        yield
    finally:
        INFLIGHT_GAUGE.labels(protocol=protocol, endpoint=endpoint).dec()
