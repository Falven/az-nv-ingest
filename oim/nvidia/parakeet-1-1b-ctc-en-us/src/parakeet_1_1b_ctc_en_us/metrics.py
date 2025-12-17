from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response


class RequestMetrics:
    """
    Prometheus metrics wrapper with a scoped tracking context.
    """

    def __init__(self, namespace: str):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "requests_total",
            "Total requests by method and status",
            ["method", "status"],
            namespace=namespace,
            registry=self.registry,
        )
        self.latency = Histogram(
            "request_duration_seconds",
            "Latency per method",
            ["method"],
            namespace=namespace,
            registry=self.registry,
        )
        self.active = Gauge(
            "active_requests",
            "In-flight requests by method",
            ["method"],
            namespace=namespace,
            registry=self.registry,
        )
        self.audio_seconds = Counter(
            "audio_processed_seconds_total",
            "Total audio seconds processed",
            ["method"],
            namespace=namespace,
            registry=self.registry,
        )

    @contextmanager
    def track(self, method: str) -> Iterator[None]:
        """
        Track latency, inflight, and status counters for the named method.

        Args:
            method: Logical operation name (e.g., ``Recognize``).

        Yields:
            None. Metrics are recorded on exit.
        """
        start = time.perf_counter()
        self.active.labels(method=method).inc()
        status = "ok"
        try:
            yield
        except Exception:  # pragma: no cover - defensive guard
            status = "error"
            raise
        finally:
            elapsed = time.perf_counter() - start
            self.latency.labels(method=method).observe(elapsed)
            self.requests.labels(method=method, status=status).inc()
            self.active.labels(method=method).dec()

    def render(self) -> Response:
        """
        Render the Prometheus registry as an HTTP response.

        Returns:
            A Starlette Response containing the latest metrics snapshot.
        """
        body = generate_latest(self.registry)
        return Response(
            content=body,
            media_type=CONTENT_TYPE_LATEST,
            headers={"Content-Length": str(len(body))},
        )
