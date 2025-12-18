from __future__ import annotations

import logging
import time
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from oim_common.auth import build_http_auth_dependency
from oim_common.fastapi import (
    add_health_routes,
    add_metadata_routes,
    add_metrics_route,
    add_root_route,
    add_triton_routes,
    configure_service_tracer,
    start_background_metrics,
)
from oim_common.logging import configure_logging
from oim_common.metrics import record_request, track_inflight
from oim_common.rate_limit import AsyncRateLimiter

from .clients.triton_client import TritonClient
from .errors import InferenceError, InvalidImageError, TritonInferenceError
from .inference import encode_request_images, format_http_predictions
from .models import InferRequest
from .settings import ServiceSettings

settings = ServiceSettings()
configure_logging(settings.effective_log_level)
logger = logging.getLogger(settings.model_id)
rate_limiter = AsyncRateLimiter(settings.rate_limit)
tracer = configure_service_tracer(
    enabled=settings.enable_otel,
    model_id=settings.otel_service_name or settings.model_id,
    model_version=settings.model_version,
    otel_endpoint=settings.otel_endpoint,
)
triton_client: TritonClient | None = None
auth_dependency = Depends(build_http_auth_dependency(settings))


async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown tasks for the FastAPI app."""
    global triton_client
    start_background_metrics(settings)
    triton_client = TritonClient(settings)
    await triton_client.wait_for_model_ready()
    yield
    if triton_client is not None:
        triton_client.close()
        triton_client = None


app = FastAPI(
    title=settings.model_id, version=settings.model_version, lifespan=_lifespan
)


def _ensure_client() -> TritonClient:
    """Return an initialized Triton client or raise 503."""
    if triton_client is None:
        raise HTTPException(status_code=503, detail="Triton client is not ready")
    return triton_client


def _ready_check() -> bool:
    """Readiness probe indicating model availability."""
    client = triton_client
    return bool(client and client.is_ready())


class _TritonRouteClient:
    """Adapter used by shared Triton FastAPI routes."""

    def is_live(self) -> bool:
        client = triton_client
        return bool(client and client.is_live())

    def is_ready(self) -> bool:
        client = triton_client
        return bool(client and client.is_ready())

    def repository_index(self) -> list[dict[str, object]]:
        client = triton_client
        if client is None:
            return []
        return client.repository_index()

    def model_metadata(self) -> dict[str, object]:
        client = _ensure_client()
        try:
            return client.model_metadata()
        except TritonInferenceError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    def model_config(self) -> dict[str, object]:
        client = _ensure_client()
        try:
            return client.model_config()
        except TritonInferenceError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc


add_health_routes(app, ready_check=_ready_check)
add_metadata_routes(
    app,
    model_id=settings.model_id,
    model_version=settings.model_version,
    auth_dependency=[auth_dependency],
    max_batch_size=settings.max_batch_size,
    include_models_endpoint=False,
)
add_metrics_route(
    app,
    auth_dependency=[auth_dependency],
    include_triton_route=True,
)
add_triton_routes(
    app,
    triton_model_name=settings.triton_model_name,
    client=_TritonRouteClient(),
    include_repository_index=False,
    health_dependency=[],
    metadata_dependency=[auth_dependency],
    model_ready_dependency=[],
)
add_root_route(app, message=settings.model_id)


@app.post("/v1/infer", dependencies=[auth_dependency])
async def infer(request_body: InferRequest) -> JSONResponse:
    """HTTP inference endpoint returning bounding boxes."""
    client = _ensure_client()
    start_time = time.perf_counter()
    endpoint = "/v1/infer"
    if len(request_body.input) > settings.max_batch_size:
        record_request("http", endpoint, "batch_limit", start_time)
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(request_body.input)} exceeds limit {settings.max_batch_size}",
        )

    try:
        images = await encode_request_images(
            request_body.input, settings.request_timeout_seconds
        )
    except InvalidImageError as exc:
        record_request("http", endpoint, "invalid_payload", start_time)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InferenceError as exc:
        record_request("http", endpoint, "error", start_time)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        async with rate_limiter.limit():
            with track_inflight("http", endpoint):
                if tracer:
                    with tracer.start_as_current_span("http_infer"):
                        predictions = await client.infer(images)
                else:
                    predictions = await client.infer(images)
    except TritonInferenceError as exc:
        record_request("http", endpoint, "error", start_time)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced to client
        logger.exception("Model inference failed.")
        record_request("http", endpoint, "error", start_time)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response_items = format_http_predictions(predictions)
    record_request("http", endpoint, "success", start_time)
    return JSONResponse({"data": response_items}, status_code=200)
