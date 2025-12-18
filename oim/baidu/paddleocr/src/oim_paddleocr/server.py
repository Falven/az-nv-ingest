from __future__ import annotations

import logging
import time
from contextlib import nullcontext

from fastapi import FastAPI, HTTPException
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
from oim_common.metrics import record_request, track_inflight
from oim_common.triton import TritonHttpClient

from .inference import infer as run_inference
from .inference import init_inference, load_image, model_ready
from .models import InferRequest
from .settings import ServiceSettings

logger = logging.getLogger("paddleocr-nim")


settings = ServiceSettings()
configure_logging(settings.log_level)
tracer = configure_service_tracer(
    enabled=settings.enable_otel,
    model_id=settings.model_id,
    model_version=settings.model_version,
    otel_endpoint=settings.otel_endpoint,
)
init_inference(settings)
triton_health = TritonHttpClient(
    endpoint=settings.triton_http_url,
    model_name=settings.model_name,
    timeout=settings.request_timeout_seconds,
    verbose=bool(settings.triton_log_verbose),
)
auth_dependency = build_http_auth_dependency(settings)


async def _lifespan(app: FastAPI):
    start_background_metrics(settings)
    yield


app = FastAPI(
    title=settings.model_id, version=settings.model_version, lifespan=_lifespan
)

install_http_middleware(app, tracer)
add_health_routes(
    app,
    ready_check=lambda: triton_health.is_ready() and model_ready(),
    response_style="status",
)
add_metadata_routes(
    app,
    model_id=settings.model_id,
    model_version=settings.model_version,
    short_name=settings.short_name,
    auth_dependency=[auth_dependency],
    max_batch_size=settings.max_batch_size,
)
add_triton_routes(
    app,
    triton_model_name=settings.model_name,
    client=triton_health,
    auth_dependency=[auth_dependency],
)
add_metrics_route(app, auth_dependency=[auth_dependency])
add_root_route(app, message=settings.model_id)


@app.post("/v1/infer", dependencies=[auth_dependency])
async def infer_http(request: InferRequest) -> dict:
    if len(request.input) == 0:
        raise HTTPException(
            status_code=400, detail="input must contain at least one image_url item"
        )
    if len(request.input) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(request.input)} exceeds limit {settings.max_batch_size}.",
        )
    try:
        triton_ready = triton_health.is_ready()
    except Exception as exc:
        logger.exception("Triton model readiness check failed: %s", exc)
        raise HTTPException(status_code=503, detail="Model is not loaded") from exc
    if not triton_ready or not model_ready():
        raise HTTPException(status_code=503, detail="Model is not loaded")

    images = []
    for item in request.input:
        try:
            images.append(load_image(item.url, settings.request_timeout_seconds))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = time.perf_counter()
    status_label = "200"
    try:
        with track_inflight("http", "/v1/infer"):
            with (
                tracer.start_as_current_span("http_infer") if tracer else nullcontext()
            ):
                result = run_inference(images, settings)
    except Exception as exc:
        status_label = "500"
        logger.exception("Triton inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    parsed = [item.model_dump() for item in result]
    record_request("http", "/v1/infer", status_label, started)
    return {"data": parsed}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.http_port)
