from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import Any, Dict, List

from oim_common.logging import configure_logging
from oim_common.metrics import (
    metrics_response,
    record_request,
    start_metrics_server,
    track_inflight,
)
from oim_common.rate_limit import AsyncRateLimiter
from oim_common.telemetry import configure_tracer
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from .auth import require_http_auth
from .inference import TritonRerankClient
from .models import RankingRequest
from .settings import ServiceSettings

settings = ServiceSettings()
configure_logging(settings.log_level if not settings.log_verbose else "DEBUG")
logger = logging.getLogger("llama-3.2-nv-rerankqa-1b-v2")
tracer = configure_tracer(
    enabled=settings.enable_otel,
    service_name=settings.otel_service_name or settings.model_id,
    otel_endpoint=settings.otel_endpoint,
)
triton_client = TritonRerankClient(settings)
async_rate_limiter = AsyncRateLimiter(settings.rate_limit)
auth_dependency = (
    [Depends(require_http_auth(settings))] if settings.auth_required else []
)


async def _lifespan(app: FastAPI):
    """
    Start background services at application lifespan startup.
    """
    start_metrics_server(settings.metrics_port, settings.auth_required)
    yield


app = FastAPI(
    title=settings.model_id, version=settings.model_version, lifespan=_lifespan
)


@app.middleware("http")
async def _http_tracing(request: Request, call_next):
    """
    Capture request metrics and optional tracing around each HTTP call.
    """
    endpoint = request.url.path
    status_label = "200"
    ctx_manager = (
        tracer.start_as_current_span(f"http {request.method.lower()} {endpoint}")
        if tracer
        else nullcontext()
    )
    started = time.perf_counter()
    with track_inflight("http", endpoint):
        try:
            with ctx_manager:
                response = await call_next(request)
                status_label = str(response.status_code)
                return response
        except HTTPException as exc:
            status_label = str(exc.status_code)
            raise
        except Exception:
            status_label = "500"
            raise
        finally:
            record_request("http", endpoint, status_label, started)


@app.get("/v1/health/live")
async def live() -> Dict[str, bool]:
    """
    Liveness probe endpoint.
    """
    return {"live": True}


@app.get("/v1/health/ready")
async def ready() -> Dict[str, bool]:
    """
    Readiness probe reflecting Triton model state.
    """
    return {"ready": triton_client.is_ready()}


@app.get("/v1/models", dependencies=auth_dependency)
async def list_models() -> Dict[str, Any]:
    """
    List available model identifiers.
    """
    return {"object": "list", "data": [{"id": settings.model_id}]}


@app.get("/v1/metadata", dependencies=auth_dependency)
async def metadata() -> Dict[str, Any]:
    """
    Return model metadata compatible with nv-ingest expectations.
    """
    short_name = f"{settings.model_id}:{settings.model_version}"
    response: Dict[str, Any] = {
        "id": settings.model_id,
        "name": settings.model_id,
        "version": settings.model_version,
        "modelInfo": [
            {
                "name": settings.model_id,
                "version": settings.model_version,
                "shortName": short_name,
            }
        ],
    }
    if triton_client.max_batch_size is not None:
        response["modelInfo"][0]["maxBatchSize"] = triton_client.max_batch_size
    return response


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


@app.get("/metrics", dependencies=auth_dependency)
async def metrics_endpoint() -> Response:
    """
    Expose Prometheus metrics via FastAPI when auth is enabled.
    """
    return metrics_response()


@app.get("/")
async def root() -> JSONResponse:
    """
    Default root endpoint mirroring upstream behavior.
    """
    return JSONResponse({"message": settings.model_id}, status_code=200)


@app.get("/v2/health/live", dependencies=auth_dependency)
async def triton_live() -> Dict[str, bool]:
    """
    Triton-compatible liveness probe.
    """
    return {"live": triton_client.is_live()}


@app.get("/v2/health/ready", dependencies=auth_dependency)
async def triton_ready() -> Dict[str, bool]:
    """
    Triton-compatible readiness probe.
    """
    return {"ready": triton_client.is_ready()}


@app.get("/v2/models", dependencies=auth_dependency)
async def list_triton_models() -> Dict[str, Any]:
    """
    List models for Triton v2 compatibility.
    """
    index = triton_client.repository_index()
    return {"models": index}


@app.get("/v2/models/{model_name}", dependencies=auth_dependency)
async def model_metadata_http(model_name: str) -> Dict[str, Any]:
    """
    Expose model metadata for Triton v2 compatibility.
    """
    _validate_model_name_param(model_name)
    return triton_client.model_metadata()


@app.get("/v2/models/{model_name}/config", dependencies=auth_dependency)
async def model_config_http(model_name: str) -> Dict[str, Any]:
    """
    Expose model config for Triton v2 compatibility.
    """
    _validate_model_name_param(model_name)
    return triton_client.model_config()


@app.get("/v2/models/{model_name}/ready", dependencies=auth_dependency)
async def model_ready_http(model_name: str) -> Dict[str, bool]:
    """
    Report readiness for a specific model.
    """
    _validate_model_name_param(model_name)
    return {"ready": triton_client.is_ready()}


def _validate_model_name_param(model_name: str) -> None:
    """
    Guard against unknown model names in HTTP routes.
    """
    if model_name != settings.triton_model_name:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.http_port)
