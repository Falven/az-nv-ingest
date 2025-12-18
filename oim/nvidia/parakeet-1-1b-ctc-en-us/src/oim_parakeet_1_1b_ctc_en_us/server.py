from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Optional

from fastapi import FastAPI
from oim_common.auth import AuthValidator, build_http_auth_dependency
from oim_common.logging import configure_logging
from oim_common.telemetry import configure_tracer

from .grpc_server import RequestLimiter, create_grpc_server
from .http_api import create_router
from .inference import create_asr_backend
from .metrics import RequestMetrics
from .settings import ServiceSettings

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

LOGGER = logging.getLogger(__name__)
APP_START_TIME = time.time()


def _configure_tracer(settings: ServiceSettings) -> Optional["Tracer"]:
    endpoint = settings.otel_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return None
    return configure_tracer(
        enabled=True,
        service_name=settings.otel_service_name or settings.model_name,
        otel_endpoint=endpoint,
        resource_attributes={
            "service.version": settings.model_version,
            "service.instance.id": settings.model_name,
        },
        enable_metrics=False,
    )


settings = ServiceSettings()
configure_logging(settings.log_level)
metrics = RequestMetrics(namespace=settings.metrics_namespace)
auth_validator = AuthValidator(settings)
public_health_paths = {"/", "/v1/health/live", "/v1/health/ready"}
auth_dependency = build_http_auth_dependency(
    settings,
    allow_unauthenticated_paths=public_health_paths
    if settings.allow_unauth_health
    else None,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize shared components and start the gRPC server alongside FastAPI.
    """
    tracer = _configure_tracer(settings)
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
