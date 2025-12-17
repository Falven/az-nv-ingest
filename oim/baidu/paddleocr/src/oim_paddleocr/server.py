from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import tritonclient.http as triton_http
from oim_common.logging import configure_logging
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response

if TYPE_CHECKING:  # pragma: no cover
    from opentelemetry.trace import Tracer

from .auth import require_http_auth
from .metrics import (
    BATCH_SIZE,
    REQUEST_COUNTER,
    REQUEST_LATENCY,
    render_metrics,
    start_metrics_server,
)
from .models import InferRequest
from .settings import ServiceSettings

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except Exception:  # pragma: no cover - optional dependency
    trace = None
    OTLPSpanExporter = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None

logger = logging.getLogger("paddleocr-nim")


def _configure_tracer(settings: ServiceSettings) -> Optional["Tracer"]:
    endpoint = settings.otel_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not settings.enable_otel and not endpoint:
        return None
    if (
        trace is None
        or OTLPSpanExporter is None
        or TracerProvider is None
        or BatchSpanProcessor is None
    ):
        logger.warning(
            "OpenTelemetry requested but not available; skipping OTEL setup."
        )
        return None
    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(settings.otel_service_name)


settings = ServiceSettings()
configure_logging(settings.log_level)
tracer = _configure_tracer(settings)
TRITON_HTTP_URL = os.getenv("TRITON_HTTP_URL", "http://127.0.0.1:8500")
triton_client = triton_http.InferenceServerClient(url=TRITON_HTTP_URL, verbose=False)


async def _lifespan(app: FastAPI):
    start_metrics_server(settings)
    yield


app = FastAPI(
    title=settings.model_id, version=settings.model_version, lifespan=_lifespan
)


@app.get("/v1/health/live")
async def live() -> JSONResponse:
    return JSONResponse({"status": "live"}, status_code=200)


@app.get("/v1/health/ready")
async def ready() -> JSONResponse:
    try:
        server_ready = triton_client.is_server_ready()
        model_ready = triton_client.is_model_ready(settings.model_name)
    except Exception as exc:
        logger.exception("Triton readiness check failed: %s", exc)
        return JSONResponse({"status": "unavailable"}, status_code=503)
    is_ready = server_ready and model_ready
    return JSONResponse(
        {"status": "ready" if is_ready else "unavailable"},
        status_code=200 if is_ready else 503,
    )


@app.get("/v1/metadata", dependencies=[Depends(require_http_auth)])
async def metadata() -> JSONResponse:
    try:
        meta = triton_client.get_model_metadata(settings.model_name)
        max_batch = meta.get("max_batch_size", settings.max_batch_size)
    except Exception:
        max_batch = settings.max_batch_size
    model_info = {
        "id": settings.model_id,
        "name": settings.model_id,
        "version": settings.model_version,
        "shortName": settings.short_name,
        "maxBatchSize": max_batch,
    }
    return JSONResponse({"modelInfo": [model_info]}, status_code=200)


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"message": settings.model_id}, status_code=200)


@app.get("/metrics", dependencies=[Depends(require_http_auth)])
@app.get("/v2/metrics", dependencies=[Depends(require_http_auth)])
async def metrics() -> Response:
    return render_metrics()


@app.post("/v1/infer", dependencies=[Depends(require_http_auth)])
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
        ready = triton_client.is_model_ready(settings.model_name)
    except Exception as exc:
        logger.exception("Triton model readiness check failed: %s", exc)
        raise HTTPException(status_code=503, detail="Model is not loaded") from exc
    if not ready:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    urls: Sequence[str] = [item.url for item in request.input]
    triton_input = triton_http.InferInput("INPUT", [len(urls)], "BYTES")
    payload = np.array(urls, dtype=object)
    triton_input.set_data_from_numpy(payload)
    triton_output = triton_http.InferRequestedOutput("OUTPUT")

    REQUEST_COUNTER.labels("http").inc()
    try:
        with tracer.start_as_current_span("http_infer") if tracer else nullcontext():
            result = triton_client.infer(
                model_name=settings.model_name,
                inputs=[triton_input],
                outputs=[triton_output],
            )
    except Exception as exc:
        logger.exception("Triton inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    outputs = result.as_numpy("OUTPUT")
    if outputs is None:
        outputs = []
    BATCH_SIZE.labels("http").observe(float(len(urls)))
    REQUEST_LATENCY.labels("http").observe(0.0)
    try:
        parsed = [json.loads(item.decode("utf-8")) for item in outputs]
    except Exception as exc:
        logger.exception("Failed to parse Triton output: %s", exc)
        raise HTTPException(
            status_code=500, detail="Failed to parse inference output"
        ) from exc
    return {"data": parsed}


@app.get("/v2/health/live")
async def triton_live() -> dict:
    return {"live": True}


@app.get("/v2/health/ready")
async def triton_ready() -> dict:
    try:
        return {"ready": triton_client.is_model_ready(settings.model_name)}
    except Exception:
        return {"ready": False}


@app.get("/v2/models", dependencies=[Depends(require_http_auth)])
async def list_models() -> dict:
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


@app.get("/v2/models/{model_name}", dependencies=[Depends(require_http_auth)])
async def model_metadata_http(model_name: str) -> dict:
    if model_name != settings.model_name:
        raise HTTPException(status_code=404, detail=f"Unknown model {model_name}")
    try:
        return triton_client.get_model_metadata(model_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v2/models/{model_name}/config", dependencies=[Depends(require_http_auth)])
async def model_config_http(model_name: str) -> dict:
    if model_name != settings.model_name:
        raise HTTPException(status_code=404, detail=f"Unknown model {model_name}")
    try:
        return triton_client.get_model_config(model_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v2/models/{model_name}/ready")
async def model_ready_http(model_name: str) -> dict:
    try:
        is_ready = model_name == settings.model_name and triton_client.is_model_ready(
            model_name
        )
    except Exception:
        is_ready = False
    return {"ready": is_ready}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.http_port)
