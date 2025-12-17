from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import Depends, FastAPI
from oim_common.auth import build_http_auth_dependency
from oim_common.fastapi import (
    add_health_routes,
    add_metadata_routes,
    add_metrics_route,
    add_root_route,
    add_triton_routes,
    configure_service_tracer,
    install_http_middleware,
    start_background_metrics,
)
from oim_common.logging import configure_logging
from oim_common.rate_limit import AsyncRateLimiter

from .inference import TritonEmbeddingClient
from .models import EmbeddingsRequest
from .settings import ServiceSettings

settings = ServiceSettings()
configure_logging(settings.log_level if not settings.log_verbose else "DEBUG")
logger = logging.getLogger("llama-3.2-nv-embedqa-1b-v2")
tracer = configure_service_tracer(
    enabled=settings.enable_otel,
    model_id=settings.model_id,
    model_version=settings.model_version,
    otel_endpoint=settings.otel_endpoint,
)
triton_client = TritonEmbeddingClient(settings)
async_rate_limiter = AsyncRateLimiter(settings.rate_limit)
auth_dependency = Depends(build_http_auth_dependency(settings))


async def _lifespan(app: FastAPI):
    """
    Start background services at application lifespan startup.
    """
    start_background_metrics(settings)
    yield


app = FastAPI(
    title=settings.model_id, version=settings.model_version, lifespan=_lifespan
)


install_http_middleware(app, tracer)
add_health_routes(
    app,
    live_dependency=[auth_dependency],
    ready_dependency=[auth_dependency],
    ready_check=triton_client.is_ready,
)
add_metadata_routes(
    app,
    model_id=settings.model_id,
    model_version=settings.model_version,
    auth_dependency=[auth_dependency],
)
add_triton_routes(
    app,
    triton_model_name=settings.triton_model_name,
    client=triton_client,
    auth_dependency=[auth_dependency],
)
add_metrics_route(app, auth_dependency=[auth_dependency])
add_root_route(app, message=settings.model_id, auth_dependency=[auth_dependency])


@app.post("/v1/embeddings", dependencies=[auth_dependency])
async def embeddings(request_body: EmbeddingsRequest) -> Dict[str, Any]:
    """
    Compute embeddings for provided input texts.
    """
    async with async_rate_limiter.limit():
        return triton_client.embed(request_body)
