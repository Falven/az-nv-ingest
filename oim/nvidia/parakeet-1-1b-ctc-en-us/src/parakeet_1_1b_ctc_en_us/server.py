from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from common.logging import configure_logging
from fastapi import FastAPI

from .auth import AuthValidator, build_http_auth_dependency
from .grpc_server import RequestLimiter, create_grpc_server
from .http_api import create_router
from .inference import create_asr_backend
from .metrics import RequestMetrics
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

LOGGER = logging.getLogger(__name__)
APP_START_TIME = time.time()


def configure_tracer(settings: ServiceSettings) -> Optional["trace.Tracer"]:
    """
    Configure an OTLP tracer when an endpoint is provided.
    """
    endpoint = settings.otel_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return None
    if (
        trace is None
        or OTLPSpanExporter is None
        or TracerProvider is None
        or BatchSpanProcessor is None
    ):
        LOGGER.warning(
            "OpenTelemetry requested but not available; skipping OTEL setup."
        )
        return None
    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": settings.model_version,
            "service.instance.id": settings.model_name,
        }
    )
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(settings.otel_service_name)


settings = ServiceSettings()
configure_logging(settings.log_level)
metrics = RequestMetrics(namespace=settings.metrics_namespace)
auth_validator = AuthValidator(settings)
public_health_paths = {"/", "/v1/health/live", "/v1/health/ready"}
auth_dependency = build_http_auth_dependency(
    auth_validator,
    allow_unauthenticated_health=settings.allow_unauth_health,
    public_paths=public_health_paths,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize shared components and start the gRPC server alongside FastAPI.
    """
    tracer = configure_tracer(settings)
    limiter = RequestLimiter(
        settings.max_concurrent_requests, settings.max_streaming_sessions
    )
    backend = create_asr_backend(settings)
    grpc_server, triton_service = create_grpc_server(
        backend,
        settings,
        metrics,
        limiter,
        tracer,
        auth_validator,
    )
    app.state.grpc_server = grpc_server
    app.state.triton_service = triton_service
    app.state.triton_config = triton_service.config
    grpc_server.start()
    LOGGER.info("gRPC listening on %s", settings.grpc_port)
    try:
        yield
    finally:
        grpc_server.stop(grace=5)
        grpc_server.wait_for_termination(timeout=5)


app = FastAPI(
    title=settings.model_name,
    version=settings.model_version,
    lifespan=lifespan,
)
app.include_router(create_router(settings, metrics, auth_dependency, APP_START_TIME))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.http_port)
