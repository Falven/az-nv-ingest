from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import AsyncIterator, Awaitable, Callable, ContextManager

import numpy as np
import tritonclient.grpc as grpcclient
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from oim_common.auth import build_http_auth_dependency
from oim_common.logging import configure_logging
from oim_common.metrics import (
    metrics_response,
    observe_ocr_request,
    start_metrics_server,
)
from oim_common.telemetry import configure_tracer
from opentelemetry.trace import Status, StatusCode

from .inference import parsed_from_triton_output
from .models import InferRequest, InferResponse, InferResponseItem
from .settings import ServiceSettings

settings = ServiceSettings()
configure_logging(settings.log_level)
logger = logging.getLogger("nemoretriever-ocr-v1")
tracer = configure_tracer(
    enabled=getattr(settings, "enable_otel", False) or bool(settings.otel_endpoint),
    service_name=settings.otel_service_name or settings.model_name,
    otel_endpoint=settings.otel_endpoint,
)
triton_client = grpcclient.InferenceServerClient(
    url=settings.triton_grpc_url, verbose=False
)
auth_dependency = Depends(build_http_auth_dependency(settings))


def _maybe_start_span(name: str) -> ContextManager[object]:
    """
    Start an OTEL span when tracing is enabled.
    """
    return tracer.start_as_current_span(name) if tracer else nullcontext()


async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """
    Start background services while Triton remains managed by the entrypoint.
    """
    start_metrics_server(settings.metrics_port, settings.auth_required)
    yield


app = FastAPI(
    title=settings.model_name, version=settings.model_version, lifespan=_lifespan
)


@app.middleware("http")
async def _http_tracing(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Wrap HTTP requests with an OTEL span when tracing is configured.
    """
    if tracer is None:
        return await call_next(request)
    span_name = f"http {request.method.lower()} {request.url.path}"
    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.route", request.url.path)
        try:
            response = await call_next(request)
        except Exception as exc:  # pragma: no cover - surfaced to caller
            span.record_exception(exc)
            span.set_status(Status(status_code=StatusCode.ERROR))
            raise
        span.set_attribute("http.status_code", response.status_code)
        if response.status_code >= 400:
            span.set_status(Status(status_code=StatusCode.ERROR))
        return response


@app.get("/v1/health/live")
async def live() -> JSONResponse:
    """
    Simple liveness probe.
    """
    return JSONResponse({"status": "live"}, status_code=200)


@app.get("/v1/health/ready")
async def ready() -> JSONResponse:
    """
    Report readiness based on Triton model status.
    """
    try:
        is_ready = triton_client.is_model_ready(settings.model_name)
    except Exception as exc:
        logger.exception("Triton readiness check failed: %s", exc)
        return JSONResponse({"status": "unavailable"}, status_code=503)
    status_value = "ready" if is_ready else "unavailable"
    return JSONResponse({"status": status_value}, status_code=200 if is_ready else 503)


@app.get("/v1/metadata", dependencies=[auth_dependency])
async def metadata() -> JSONResponse:
    """
    Return model metadata in the nv-ingest discovery shape.
    """
    try:
        meta = triton_client.get_model_metadata(
            model_name=settings.model_name, as_json=True
        )
        max_batch = meta.get("max_batch_size", settings.max_batch_size)
    except Exception:
        max_batch = settings.max_batch_size
    model_info = {
        "id": "nvidia/nemotron-ocr-v1",
        "name": "nemotron-ocr-v1",
        "version": settings.model_version,
        "shortName": "nemoretriever-ocr-v1",
        "maxBatchSize": max_batch,
    }
    return JSONResponse({"modelInfo": [model_info]}, status_code=200)


@app.get("/metrics", dependencies=[auth_dependency])
@app.get("/v2/metrics", dependencies=[auth_dependency])
async def metrics() -> Response:
    """
    Render Prometheus metrics collected by the FastAPI shim.
    """
    return metrics_response()


@app.post("/v1/infer", dependencies=[auth_dependency], response_model=InferResponse)
async def infer(request: InferRequest) -> InferResponse:
    """
    Run OCR inference via Triton and return nv-ingest response shape.
    """
    if len(request.input) == 0:
        raise HTTPException(
            status_code=400, detail="input must contain at least one image_url item"
        )

    if len(request.input) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(request.input)} exceeds limit {settings.max_batch_size}.",
        )

    merge_levels = (
        request.merge_levels
        if request.merge_levels is not None
        else [settings.merge_level] * len(request.input)
    )
    if len(merge_levels) != len(request.input):
        raise HTTPException(
            status_code=400, detail="merge_levels length must match input length."
        )

    try:
        model_ready = triton_client.is_model_ready(settings.model_name)
    except Exception as exc:
        logger.exception("Triton readiness check failed: %s", exc)
        raise HTTPException(status_code=503, detail="Model is not loaded") from exc
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    batch_size = len(request.input)
    image_values = np.array(
        [item.url for item in request.input], dtype=np.object_
    ).reshape(batch_size, 1)
    merge_values = np.array(merge_levels, dtype=np.object_).reshape(batch_size, 1)

    input_images = grpcclient.InferInput(
        "INPUT_IMAGE_URLS", image_values.shape, "BYTES"
    )
    input_images.set_data_from_numpy(image_values)
    merge_input = grpcclient.InferInput("MERGE_LEVELS", merge_values.shape, "BYTES")
    merge_input.set_data_from_numpy(merge_values)
    output = grpcclient.InferRequestedOutput("OUTPUT")

    start_time = time.perf_counter()
    with _maybe_start_span("http_infer") as span:
        if tracer and span:
            span.set_attribute("ocr.batch_size", batch_size)
        try:
            result = triton_client.infer(
                model_name=settings.model_name,
                inputs=[input_images, merge_input],
                outputs=[output],
                client_timeout=settings.request_timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - surfaced to caller
            if tracer and span:
                span.record_exception(exc)
                span.set_status(Status(status_code=StatusCode.ERROR))
            logger.exception("Triton inference failed.")
            raise HTTPException(
                status_code=500, detail="Model inference failed"
            ) from exc

    duration = time.perf_counter() - start_time
    observe_ocr_request("http", batch_size, duration)

    try:
        parsed = parsed_from_triton_output(result.as_numpy("OUTPUT"))
    except Exception as exc:  # pragma: no cover - surfaced to caller
        logger.exception("Failed to parse Triton output: %s", exc)
        raise HTTPException(
            status_code=500, detail="Failed to parse inference output"
        ) from exc

    response_items = [
        InferResponseItem(text_detections=item.detections) for item in parsed
    ]
    return InferResponse(data=response_items)


@app.get("/v2/health/live")
async def triton_live() -> dict:
    """
    Triton-style liveness probe.
    """
    return {"live": True}


@app.get("/v2/health/ready")
async def triton_ready() -> dict:
    """
    Triton-style readiness probe.
    """
    try:
        return {"ready": triton_client.is_model_ready(settings.model_name)}
    except Exception:
        return {"ready": False}


@app.get("/v2/models", dependencies=[auth_dependency])
async def list_models() -> dict:
    """
    List the single model served by this NIM in Triton format.
    """
    try:
        ready = triton_client.is_model_ready(settings.model_name)
    except Exception:
        ready = False
    state = "READY" if ready else "UNAVAILABLE"
    return {
        "models": [
            {
                "name": settings.model_name,
                "version": settings.model_version,
                "state": state,
            }
        ]
    }


@app.get("/v2/models/{model_name}", dependencies=[auth_dependency])
async def model_metadata_http(model_name: str) -> dict:
    """
    Return Triton metadata for the configured model.
    """
    if model_name != settings.model_name:
        raise HTTPException(status_code=404, detail=f"Unknown model {model_name}")
    try:
        metadata = triton_client.get_model_metadata(model_name=model_name, as_json=True)
    except Exception as exc:
        logger.exception("Failed to fetch model metadata: %s", exc)
        raise HTTPException(
            status_code=500, detail="Model metadata unavailable"
        ) from exc
    return metadata


@app.get("/v2/models/{model_name}/ready")
async def model_ready_http(model_name: str) -> dict:
    """
    Report readiness for a specific model.
    """
    if model_name != settings.model_name:
        return {"ready": False}
    try:
        return {"ready": triton_client.is_model_ready(model_name)}
    except Exception:
        return {"ready": False}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.http_port)
