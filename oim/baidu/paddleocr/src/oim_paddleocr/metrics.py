from __future__ import annotations

import logging

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
    start_http_server,
)
from starlette.responses import Response

from .settings import ServiceSettings

logger = logging.getLogger("paddleocr-metrics")

REQUEST_COUNTER = Counter(
    "nim_paddleocr_requests_total", "Total inference requests", ["protocol"]
)
REQUEST_LATENCY = Histogram(
    "nim_paddleocr_request_latency_seconds",
    "Latency for inference requests",
    ["protocol"],
)
BATCH_SIZE = Histogram("nim_paddleocr_batch_size", "Observed batch sizes", ["protocol"])

_started = False


def start_metrics_server(settings: ServiceSettings) -> None:
    global _started
    if _started:
        return
    if settings.require_auth:
        logger.info("Metrics auth enabled; skipping standalone metrics server.")
        _started = True
        return
    start_http_server(settings.metrics_port)
    _started = True
    logger.info("Started metrics server on :%d/metrics", settings.metrics_port)


def render_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
