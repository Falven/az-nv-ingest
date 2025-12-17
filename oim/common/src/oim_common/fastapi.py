from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any, Awaitable, Callable, Iterable, Mapping

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from .metrics import (
    metrics_response,
    record_request,
    start_metrics_server,
    track_inflight,
)
from .telemetry import configure_tracer

HttpHandler = Callable[[Request], Awaitable[Response]]


def install_http_middleware(app: FastAPI, tracer: Any | None) -> None:
    """
    Attach middleware that records request metrics and optional tracing.
    """

    @app.middleware("http")
    async def _http_tracing(request: Request, call_next: HttpHandler) -> Response:
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


def add_health_routes(
    app: FastAPI,
    *,
    live_dependency: Iterable[Depends] | None = None,
    ready_dependency: Iterable[Depends] | None = None,
    ready_check: Callable[[], bool],
) -> None:
    """
    Add standardized live/ready endpoints to the application.
    """

    @app.get("/v1/health/live", dependencies=list(live_dependency or ()))
    async def live() -> Mapping[str, bool]:
        return {"live": True}

    @app.get("/v1/health/ready", dependencies=list(ready_dependency or ()))
    async def ready() -> Mapping[str, bool]:
        return {"ready": ready_check()}


def add_metadata_routes(
    app: FastAPI,
    *,
    model_id: str,
    model_version: str,
    auth_dependency: Iterable[Depends] | None = None,
    max_batch_size: int | None = None,
) -> None:
    """
    Add nv-ingest compatible metadata and models endpoints.
    """

    dependencies = list(auth_dependency or ())
    short_name = f"{model_id}:{model_version}"

    @app.get("/v1/models", dependencies=dependencies)
    async def list_models() -> Mapping[str, Any]:
        return {"object": "list", "data": [{"id": model_id}]}

    @app.get("/v1/metadata", dependencies=dependencies)
    async def metadata() -> Mapping[str, Any]:
        info: dict[str, Any] = {
            "name": model_id,
            "version": model_version,
            "shortName": short_name,
        }
        if max_batch_size is not None:
            info["maxBatchSize"] = max_batch_size
        return {
            "id": model_id,
            "name": model_id,
            "version": model_version,
            "modelInfo": [info],
        }


def add_triton_routes(
    app: FastAPI,
    *,
    triton_model_name: str,
    client: Any,
    auth_dependency: Iterable[Depends] | None = None,
) -> None:
    """
    Add Triton v2 compatibility endpoints.
    """

    dependencies = list(auth_dependency or ())

    @app.get("/v2/health/live", dependencies=dependencies)
    async def triton_live() -> Mapping[str, bool]:
        return {"live": client.is_live()}

    @app.get("/v2/health/ready", dependencies=dependencies)
    async def triton_ready() -> Mapping[str, bool]:
        return {"ready": client.is_ready()}

    @app.get("/v2/models", dependencies=dependencies)
    async def list_triton_models() -> Mapping[str, Any]:
        return {"models": client.repository_index()}

    @app.get("/v2/models/{model_name}", dependencies=dependencies)
    async def model_metadata_http(model_name: str) -> Mapping[str, Any]:
        _validate_model_name(triton_model_name, model_name)
        return client.model_metadata()

    @app.get("/v2/models/{model_name}/config", dependencies=dependencies)
    async def model_config_http(model_name: str) -> Mapping[str, Any]:
        _validate_model_name(triton_model_name, model_name)
        return client.model_config()

    @app.get("/v2/models/{model_name}/ready", dependencies=dependencies)
    async def model_ready_http(model_name: str) -> Mapping[str, bool]:
        _validate_model_name(triton_model_name, model_name)
        return {"ready": client.is_ready()}


def add_metrics_route(
    app: FastAPI, *, auth_dependency: Iterable[Depends] | None = None
) -> None:
    """
    Mount /metrics endpoint with optional auth dependency.
    """

    @app.get("/metrics", dependencies=list(auth_dependency or ()))
    async def metrics_endpoint() -> Response:
        return metrics_response()


def add_root_route(
    app: FastAPI, *, message: str, auth_dependency: Iterable[Depends] | None = None
) -> None:
    """
    Add a simple root route mirroring upstream behavior.
    """

    @app.get("/", dependencies=list(auth_dependency or ()))
    async def root() -> JSONResponse:
        return JSONResponse({"message": message}, status_code=200)


def configure_service_tracer(
    *,
    enabled: bool,
    model_id: str,
    model_version: str,
    otel_endpoint: str | None,
) -> Any | None:
    """
    Wrapper to configure OpenTelemetry with standard attributes.
    """

    return configure_tracer(
        enabled=enabled,
        service_name=model_id,
        otel_endpoint=otel_endpoint,
        resource_attributes={"service.version": model_version},
    )


def start_background_metrics(settings) -> None:
    """
    Start standalone metrics server if safe to expose.
    """
    start_metrics_server(settings.metrics_port, settings.auth_required)


def _validate_model_name(expected: str, provided: str) -> None:
    if provided != expected:
        raise HTTPException(status_code=404, detail=f"Unknown model '{provided}'")
