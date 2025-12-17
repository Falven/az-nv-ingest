from __future__ import annotations

import json
import logging
import os
import time
from contextlib import nullcontext
from typing import Sequence

import numpy as np
import tritonclient.http as triton_http
from fastapi import Depends, FastAPI, HTTPException
from oim_common.auth import build_http_auth_dependency
from oim_common.fastapi_app import (
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
TRITON_HTTP_URL = os.getenv("TRITON_HTTP_URL", "http://127.0.0.1:8500")
triton_client = triton_http.InferenceServerClient(url=TRITON_HTTP_URL, verbose=False)
auth_dependency = Depends(build_http_auth_dependency(settings))


class _TritonClientProxy:
    def is_live(self) -> bool:
        try:
            return bool(triton_client.is_server_live())
        except Exception:
            return False

    def is_ready(self) -> bool:
        try:
            return bool(
                triton_client.is_server_ready()
                and triton_client.is_model_ready(settings.model_name)
            )
        except Exception:
            return False

    def repository_index(self):
        return triton_client.get_model_repository_index()

    def model_metadata(self):
        return triton_client.get_model_metadata(settings.model_name)

    def model_config(self):
        return triton_client.get_model_config(settings.model_name)


async def _lifespan(app: FastAPI):
    start_background_metrics(settings)
    yield


app = FastAPI(
    title=settings.model_id, version=settings.model_version, lifespan=_lifespan
)

install_http_middleware(app, tracer)
add_health_routes(app, ready_check=_TritonClientProxy().is_ready)
add_metadata_routes(
    app,
    model_id=settings.model_id,
    model_version=settings.model_version,
    auth_dependency=[auth_dependency],
    max_batch_size=settings.max_batch_size,
)
add_triton_routes(
    app,
    triton_model_name=settings.model_name,
    client=_TritonClientProxy(),
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

    started = time.perf_counter()
    status_label = "200"
    try:
        with track_inflight("http", "/v1/infer"):
            with (
                tracer.start_as_current_span("http_infer") if tracer else nullcontext()
            ):
                result = triton_client.infer(
                    model_name=settings.model_name,
                    inputs=[triton_input],
                    outputs=[triton_output],
                )
    except Exception as exc:
        status_label = "500"
        logger.exception("Triton inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    outputs = result.as_numpy("OUTPUT")
    if outputs is None:
        outputs = []
    try:
        parsed = [json.loads(item.decode("utf-8")) for item in outputs]
    except Exception as exc:
        status_label = "502"
        logger.exception("Failed to parse Triton output: %s", exc)
        raise HTTPException(
            status_code=500, detail="Failed to parse inference output"
        ) from exc
    record_request("http", "/v1/infer", status_label, started)
    return {"data": parsed}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.http_port)
