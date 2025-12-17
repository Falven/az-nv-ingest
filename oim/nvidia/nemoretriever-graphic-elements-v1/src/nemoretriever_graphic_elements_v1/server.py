from __future__ import annotations

import logging
import time
from typing import Optional

from common.logging import configure_logging
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .auth import require_http_auth
from .clients.triton_client import TritonClient
from .errors import (
    InvalidImageError,
    InferenceError,
    TritonInferenceError,
    TritonStartupError,
)
from .inference import encode_request_images, format_http_predictions
from .metrics import (
    record_request,
    render_metrics,
    start_metrics_server,
    track_inflight,
)
from .models import InferRequest
from .settings import ServiceSettings
from .triton_server import TritonServer

settings = ServiceSettings()
configure_logging(settings.effective_log_level)
logger = logging.getLogger(settings.model_name)
triton_server = TritonServer(settings)
triton_client: Optional[TritonClient] = None
auth_dependency = Depends(require_http_auth(settings))


async def _lifespan(_app: FastAPI):
    """Manage startup and shutdown tasks for the FastAPI app."""
    global triton_client
    start_metrics_server(settings)
    try:
        await triton_server.start()
    except TritonStartupError as exc:
        logger.error("Failed to start Triton: %s", exc)
        raise
    triton_client = TritonClient(settings)
    await triton_client.wait_for_model_ready()
    yield
    if triton_client is not None:
        triton_client.close()
        triton_client = None
    await triton_server.stop()


app = FastAPI(
    title=settings.model_name, version=settings.model_version, lifespan=_lifespan
)


def _ensure_client() -> TritonClient:
    """Return an initialized Triton client or raise 503."""
    if triton_client is None:
        raise HTTPException(status_code=503, detail="Triton client is not ready")
    return triton_client


@app.get("/v1/health/live")
async def live() -> JSONResponse:
    """Liveness probe for container orchestration."""
    return JSONResponse({"live": True}, status_code=200)


@app.get("/v1/health/ready")
async def ready() -> JSONResponse:
    """Readiness probe indicating model availability."""
    client = triton_client
    is_ready = bool(client and client.is_model_ready())
    return JSONResponse({"ready": is_ready}, status_code=200)


@app.get("/v2/health/live")
async def triton_live() -> JSONResponse:
    """Triton-compatible liveness probe."""
    client = triton_client
    is_live = bool(client and client.is_server_ready())
    return JSONResponse({"live": is_live}, status_code=200)


@app.get("/v2/health/ready")
async def triton_ready() -> JSONResponse:
    """Triton-compatible readiness probe."""
    client = triton_client
    is_ready = bool(client and client.is_model_ready())
    return JSONResponse({"ready": is_ready}, status_code=200)


@app.get("/v1/metadata", dependencies=[auth_dependency])
async def metadata() -> JSONResponse:
    """Expose model metadata for discovery."""
    short_name = f"{settings.model_name}:{settings.model_version}"
    payload = {
        "id": settings.model_name,
        "name": settings.model_name,
        "version": settings.model_version,
        "modelInfo": [
            {
                "name": settings.model_name,
                "version": settings.model_version,
                "shortName": short_name,
                "maxBatchSize": settings.max_batch_size,
            }
        ],
    }
    return JSONResponse(payload, status_code=200)


@app.get("/metrics", dependencies=[auth_dependency])
@app.get("/v2/metrics", dependencies=[auth_dependency])
async def metrics() -> JSONResponse:
    """Render Prometheus metrics."""
    return render_metrics()


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
    client = triton_client
    is_ready = bool(client and client.is_model_ready())
    return JSONResponse({"ready": is_ready})


@app.post("/v1/infer", dependencies=[auth_dependency])
async def infer(request_body: InferRequest) -> JSONResponse:
    """HTTP inference endpoint returning bounding boxes."""
    client = _ensure_client()
    start_time = time.perf_counter()
    endpoint = "/v1/infer"
    if len(request_body.input) > settings.max_batch_size:
        record_request(endpoint, "http", "batch_limit", start_time)
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(request_body.input)} exceeds limit {settings.max_batch_size}",
        )

    try:
        images = await encode_request_images(
            request_body.input, settings.request_timeout_seconds
        )
    except InvalidImageError as exc:
        record_request(endpoint, "http", "invalid_payload", start_time)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InferenceError as exc:
        record_request(endpoint, "http", "error", start_time)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        with track_inflight("http", endpoint):
            predictions = await client.infer(images)
    except TritonInferenceError as exc:
        record_request(endpoint, "http", "error", start_time)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced to client
        logger.exception("Model inference failed.")
        record_request(endpoint, "http", "error", start_time)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response_items = format_http_predictions(predictions)
    record_request(endpoint, "http", "success", start_time)
    return JSONResponse({"data": response_items}, status_code=200)


@app.get("/")
async def root() -> JSONResponse:
    """Health message for the service root."""
    return JSONResponse({"message": f"{settings.model_name} HTTP"}, status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.http_port)
