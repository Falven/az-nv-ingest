from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager, nullcontext
from typing import Any, Mapping

from oim_common.logging import configure_logging
from oim_common.metrics import (
    metrics_response,
    record_request,
    start_metrics_server,
    track_inflight,
)
from oim_common.rate_limit import AsyncRateLimiter
from oim_common.telemetry import configure_tracer
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from .auth import require_http_auth
from .clients.triton_client import TritonCaptionClient
from .inference import (
    generate_caption,
    prepare_request,
    stream_caption,
)
from .models import ChatChoice, ChatMessageResponse, ChatRequest, ChatResponse
from .settings import ServiceSettings
from .triton_server import TritonServer

settings = ServiceSettings()
configure_logging(settings.logging_level)
logger = logging.getLogger("nemotron-nano-12b-v2-vl")
tracer = configure_tracer(
    enabled=settings.enable_otel,
    service_name=settings.otel_service_name or settings.served_model_name,
    otel_endpoint=settings.otel_endpoint,
)
rate_limiter = AsyncRateLimiter(settings.rate_limit)
auth_dependency = require_http_auth(settings)
triton_server = TritonServer(settings)
triton_client: TritonCaptionClient | None = None
startup_error: str | None = None


def _ensure_triton_client() -> TritonCaptionClient:
    """
    Resolve the initialized Triton client or raise when unavailable.
    """
    if startup_error:
        raise HTTPException(
            status_code=503,
            detail=startup_error,
        )
    client = triton_client
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="triton client not initialized",
        )
    if not client.is_ready():
        raise HTTPException(
            status_code=503,
            detail="triton model not ready",
        )
    return client


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """
    Manage startup and shutdown tasks for the FastAPI application.
    """
    global triton_client, startup_error
    start_metrics_server(settings.metrics_port, settings.auth_required)
    try:
        await triton_server.start()
        client = TritonCaptionClient(settings)
        await client.wait_for_ready()
        triton_client = client
    except Exception as exc:  # pragma: no cover - surfaced to logs
        startup_error = str(exc)
        logger.exception("Failed to initialize Triton")
    yield
    if triton_client is not None:
        triton_client.close()
        triton_client = None
    try:
        await triton_server.stop()
    except Exception:
        logger.debug("Error during Triton shutdown", exc_info=True)


app = FastAPI(
    title=settings.served_model_name,
    version=settings.model_version,
    lifespan=_lifespan,
)


@app.get("/v1/health/live", dependencies=[auth_dependency])
async def live() -> Mapping[str, bool]:
    """
    Liveness probe.
    """
    return {"live": True}


@app.get("/v1/health/ready", dependencies=[auth_dependency])
async def ready() -> Mapping[str, Any]:
    """
    Readiness probe exposing Triton status.
    """
    client = triton_client
    if startup_error:
        return {"ready": False, "error": startup_error}
    if client is None:
        return {"ready": False, "error": "triton client not initialized"}
    try:
        is_ready = client.is_ready()
        return {
            "ready": is_ready,
            "error": None if is_ready else "triton model not ready",
        }
    except Exception as exc:  # pragma: no cover - best effort status
        return {"ready": False, "error": str(exc)}


@app.get("/v1/models", dependencies=[auth_dependency])
async def models() -> Mapping[str, Any]:
    """
    Enumerate the served model for compatibility with nv-ingest.
    """
    return {"data": [{"id": settings.served_model_name, "object": "model"}]}


@app.get("/v1/metadata", dependencies=[auth_dependency])
async def metadata() -> Mapping[str, Any]:
    """
    Report model metadata.
    """
    return {
        "id": settings.served_model_name,
        "model_id": settings.model_id,
        "version": settings.model_version,
        "max_batch_size": settings.max_batch_size,
        "max_output_tokens": settings.max_output_tokens,
    }


@app.get("/metrics", dependencies=[auth_dependency])
async def metrics_endpoint() -> Response:
    """
    Prometheus metrics endpoint.
    """
    return metrics_response()


@app.post("/v1/chat/completions", dependencies=[auth_dependency])
async def chat_completions(request: Request) -> Response:
    """
    Caption an image using the VLM model through Triton.
    """
    started = time.perf_counter()
    endpoint = "/v1/chat/completions"
    status_label = "200"
    with track_inflight("http", endpoint):
        try:
            payload = ChatRequest.model_validate(await request.json())
            client = _ensure_triton_client()
            parsed = prepare_request(payload, settings)
            completion_id = f"chatcmpl-{os.urandom(6).hex()}"
            created = int(time.time())

            rate_context = rate_limiter.limit()
            span_context = (
                tracer.start_as_current_span("chat_completions")
                if tracer
                else nullcontext()
            )
            async with rate_context:
                with span_context:
                    if parsed.stream:
                        generator = stream_caption(
                            client,
                            parsed,
                            completion_id,
                            created,
                            settings.served_model_name,
                        )
                        return StreamingResponse(
                            generator,
                            media_type="text/event-stream",
                        )
                    caption = await generate_caption(client, parsed)

            response = ChatResponse(
                id=completion_id,
                object="chat.completion",
                created=created,
                model=settings.served_model_name,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessageResponse(
                            role="assistant",
                            content=caption,
                        ),
                        finish_reason="stop",
                    )
                ],
            )
            return JSONResponse(content=response.model_dump())
        except HTTPException as exc:
            status_label = str(exc.status_code)
            raise
        except ValidationError as exc:
            status_label = "400"
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        except Exception as exc:  # pragma: no cover - surfaced to logs
            logger.exception("Unhandled error during chat completion")
            status_label = "500"
            raise HTTPException(
                status_code=500,
                detail="internal server error",
            ) from exc
        finally:
            record_request("http", endpoint, status_label, started)


@app.get("/", dependencies=[auth_dependency])
async def root() -> Mapping[str, str]:
    """
    Basic service descriptor.
    """
    return {"service": settings.served_model_name, "status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "oim_nemotron_nano_12b_v2_vl.server:app",
        host="0.0.0.0",
        port=settings.http_port,
        reload=False,
    )
