from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import AsyncIterator, ContextManager

import numpy as np
import tritonclient.grpc as grpcclient
from fastapi import Depends, FastAPI, HTTPException
from oim_common.auth import build_http_auth_dependency
from oim_common.fastapi import (
    add_health_routes,
    add_metadata_routes,
    add_metrics_route,
    configure_service_tracer,
    install_http_middleware,
    start_background_metrics,
)
from oim_common.logging import configure_logging
from oim_common.metrics import observe_ocr_request
from oim_common.triton import parse_max_batch_size
from opentelemetry.trace import Status, StatusCode

from .inference import parsed_from_triton_output
from .models import InferRequest, InferResponse, InferResponseItem
from .settings import ServiceSettings

MODEL_ID = "nvidia/nemotron-ocr-v1"
MODEL_SHORT_NAME = "nemoretriever-ocr-v1"

settings = ServiceSettings()
configure_logging(settings.log_level)
logger = logging.getLogger("nemoretriever-ocr-v1")
tracer = configure_service_tracer(
    enabled=getattr(settings, "enable_otel", False) or bool(settings.otel_endpoint),
    model_id=settings.otel_service_name or settings.model_name,
    model_version=settings.model_version,
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
    start_background_metrics(settings)
    yield


app = FastAPI(
    title=settings.model_name, version=settings.model_version, lifespan=_lifespan
)

install_http_middleware(app, tracer)


def _ready_check() -> bool:
    """
    Report readiness based on Triton model status.
    """
    try:
        return bool(triton_client.is_model_ready(settings.model_name))
    except Exception as exc:
        logger.exception("Triton readiness check failed: %s", exc)
        return False


def _metadata_max_batch_size() -> int:
    """
    Resolve maxBatchSize from Triton metadata with a local fallback.
    """
    try:
        metadata = triton_client.get_model_metadata(
            model_name=settings.model_name, as_json=True
        )
    except Exception:
        logger.debug("Failed to fetch Triton metadata for batch size.", exc_info=True)
        return settings.max_batch_size
    parsed = parse_max_batch_size(metadata) if isinstance(metadata, dict) else None
    return parsed if parsed is not None else settings.max_batch_size


add_health_routes(
    app,
    ready_check=_ready_check,
    response_style="status",
)
add_metadata_routes(
    app,
    model_id=MODEL_ID,
    model_version=settings.model_version,
    short_name=MODEL_SHORT_NAME,
    auth_dependency=[auth_dependency],
    max_batch_size=settings.max_batch_size,
    max_batch_size_resolver=_metadata_max_batch_size,
    include_models_endpoint=False,
)
add_metrics_route(
    app,
    auth_dependency=[auth_dependency],
    include_triton_route=True,
)


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
