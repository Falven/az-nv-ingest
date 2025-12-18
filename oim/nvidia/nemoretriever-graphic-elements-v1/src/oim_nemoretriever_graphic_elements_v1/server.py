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
    start_background_metrics,
)
from oim_common.logging import configure_logging
from oim_common.metrics import record_request, track_inflight

from .clients.triton_client import TritonClient
from .errors import InferenceError, InvalidImageError, TritonInferenceError
from .inference import encode_request_images, format_http_predictions
from .models import InferRequest
from .settings import ServiceSettings

settings = ServiceSettings()
configure_logging(settings.effective_log_level)
logger = logging.getLogger(settings.model_name)
triton_client: TritonClient = TritonClient(settings)
auth_dependency = Depends(build_http_auth_dependency(settings))


async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown tasks for the FastAPI app."""
    start_background_metrics(settings)
    await triton_client.wait_for_model_ready()
    yield
    triton_client.close()


app = FastAPI(
    title=settings.model_name, version=settings.model_version, lifespan=_lifespan
)


def _ensure_client() -> TritonClient:
    """Return the active Triton client."""
    return triton_client


def _ready_check() -> bool:
    """Readiness probe indicating model availability."""
    return bool(triton_client.is_ready())


add_health_routes(app, ready_check=_ready_check)
add_metadata_routes(
    app,
    model_id=settings.model_name,
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
add_root_route(app, message=f"{settings.model_name} HTTP")


@app.get("/v2/health/live")
async def triton_live() -> JSONResponse:
    """Triton-compatible liveness probe."""
    return JSONResponse({"live": triton_client.is_live()}, status_code=200)


@app.get("/v2/health/ready")
async def triton_ready() -> JSONResponse:
    """Triton-compatible readiness probe."""
    return JSONResponse({"ready": triton_client.is_ready()}, status_code=200)


@app.get("/v2/models/{model_name}", dependencies=[auth_dependency])
async def model_metadata_http(model_name: str) -> JSONResponse:
    """Expose Triton model metadata over HTTP."""
    if model_name != settings.triton_model_name:
        raise HTTPException(status_code=404, detail="Unknown model")
    client = _ensure_client()
    try:
        payload = client.model_metadata()
    except TritonInferenceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(payload)


@app.get("/v2/models/{model_name}/config", dependencies=[auth_dependency])
async def model_config_http(model_name: str) -> JSONResponse:
    """Expose Triton model configuration over HTTP."""
    if model_name != settings.triton_model_name:
        raise HTTPException(status_code=404, detail="Unknown model")
    client = _ensure_client()
    try:
        payload = client.model_config()
    except TritonInferenceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(payload)


@app.get("/v2/models/{model_name}/ready")
async def model_ready_http(model_name: str) -> JSONResponse:
    """Report model readiness for Triton parity endpoints."""
    if model_name != settings.triton_model_name:
        raise HTTPException(status_code=404, detail="Unknown model")
    return JSONResponse({"ready": triton_client.is_ready()})


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
        with track_inflight("http", endpoint):
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.http_port)
