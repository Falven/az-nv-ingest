from __future__ import annotations

from typing import Dict, List

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
from oim_common.logging import configure_logging, get_logger
from oim_common.rate_limit import AsyncRateLimiter

from .inference import TritonRerankClient
from .models import RankingRequest
from .settings import ServiceSettings

settings = ServiceSettings()
configure_logging(settings.log_level if not settings.log_verbose else "DEBUG")
logger = get_logger("llama-3.2-nv-rerankqa-1b-v2")
tracer = configure_service_tracer(
    enabled=settings.enable_otel,
    model_id=settings.model_id,
    model_version=settings.model_version,
    otel_endpoint=settings.otel_endpoint,
)
triton_client = TritonRerankClient(settings)
async_rate_limiter = AsyncRateLimiter(settings.rate_limit)
auth_dependency = (
    [Depends(build_http_auth_dependency(settings))] if settings.auth_required else []
)


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
    ready_check=triton_client.is_ready,
)
add_metadata_routes(
    app,
    model_id=settings.model_id,
    model_version=settings.model_version,
    auth_dependency=auth_dependency,
    max_batch_size=triton_client.max_batch_size,
)
add_triton_routes(
    app,
    triton_model_name=settings.triton_model_name,
    client=triton_client,
    auth_dependency=auth_dependency,
)
add_metrics_route(app, auth_dependency=auth_dependency)
add_root_route(app, message=settings.model_id, auth_dependency=auth_dependency)


@app.post("/v1/ranking", dependencies=auth_dependency)
async def ranking(request_body: RankingRequest) -> Dict[str, List[Dict[str, float]]]:
    """
    Primary HTTP ranking endpoint.
    """
    async with async_rate_limiter.limit():
        return triton_client.rerank(request_body)


@app.post(
    "/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
    dependencies=auth_dependency,
)
async def compatibility_ranking(
    request_body: RankingRequest,
) -> Dict[str, List[Dict[str, float]]]:
    """
    Backward-compatible alias for ranking.
    """
    async with async_rate_limiter.limit():
        return triton_client.rerank(request_body)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.http_port)
