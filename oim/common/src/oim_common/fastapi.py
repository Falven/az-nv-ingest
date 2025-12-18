from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import Any, Awaitable, Callable, Iterable, Literal, Mapping

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from opentelemetry.trace import Status, StatusCode

from .metrics import (
    metrics_response,
    record_request,
    start_metrics_server,
    track_inflight,
)
from .telemetry import configure_tracer

logger = logging.getLogger(__name__)

HttpHandler = Callable[[Request], Awaitable[Response]]


def install_http_middleware(app: FastAPI, tracer: Any | None) -> None:
    """
    Attach middleware that records request metrics and optional tracing.
    """

    @app.middleware("http")
    async def _http_tracing(request: Request, call_next: HttpHandler) -> Response:
        endpoint = request.url.path
        status_label = "200"
        started = time.perf_counter()
        span = None
        ctx_manager = (
            tracer.start_as_current_span(f"http {request.method.lower()} {endpoint}")
            if tracer
            else nullcontext()
        )
        with track_inflight("http", endpoint):
            try:
                with ctx_manager as active_span:
                    span = active_span if tracer else None
                    if span is not None:
                        span.set_attribute("http.method", request.method)
                        span.set_attribute("http.route", endpoint)
                    response = await call_next(request)
                    status_label = str(response.status_code)
                    if span is not None:
                        span.set_attribute("http.status_code", response.status_code)
                        if response.status_code >= 400:
                            span.set_status(Status(status_code=StatusCode.ERROR))
                    return response
            except HTTPException as exc:
                status_label = str(exc.status_code)
                if span is not None:
                    span.record_exception(exc)
                    span.set_status(Status(status_code=StatusCode.ERROR))
                raise
            except Exception as exc:
                status_label = "500"
                if span is not None:
                    span.record_exception(exc)
                    span.set_status(Status(status_code=StatusCode.ERROR))
                raise
            finally:
                record_request("http", endpoint, status_label, started)


def add_health_routes(
    app: FastAPI,
    *,
    live_dependency: Iterable[Depends] | None = None,
    ready_dependency: Iterable[Depends] | None = None,
    ready_check: Callable[[], bool],
    response_style: Literal["boolean", "status"] = "boolean",
) -> None:
    """
    Add standardized live/ready endpoints to the application.
    """

    if response_style not in {"boolean", "status"}:
        raise ValueError("Unsupported health response style")

    @app.get(
        "/v1/health/live", dependencies=list(live_dependency or ()), response_model=None
    )
    async def live() -> JSONResponse | Mapping[str, bool]:
        if response_style == "status":
            return JSONResponse({"status": "live"}, status_code=200)
        return {"live": True}

    @app.get(
        "/v1/health/ready",
        dependencies=list(ready_dependency or ()),
        response_model=None,
    )
    async def ready() -> JSONResponse | Mapping[str, bool]:
        ready = bool(ready_check())
        if response_style == "status":
            status_text = "ready" if ready else "unavailable"
            status_code = 200 if ready else 503
            return JSONResponse({"status": status_text}, status_code=status_code)
        return {"ready": ready}


def add_metadata_routes(
    app: FastAPI,
    *,
    model_id: str,
    model_version: str,
    short_name: str | None = None,
    auth_dependency: Iterable[Depends] | None = None,
    max_batch_size: int | None = None,
    max_batch_size_resolver: Callable[[], int | None] | None = None,
    include_models_endpoint: bool = True,
) -> None:
    """
    Add nv-ingest compatible metadata endpoints and optional model listing.
    """

    dependencies = list(auth_dependency or ())
    resolved_short_name = short_name or f"{model_id}:{model_version}"

    def _resolved_max_batch_size() -> int | None:
        if max_batch_size_resolver is None:
            return max_batch_size
        try:
            resolved = max_batch_size_resolver()
        except Exception:
            logger.debug(
                "Failed to resolve dynamic max_batch_size; using fallback.",
                exc_info=True,
            )
            return max_batch_size
        return resolved if resolved is not None else max_batch_size

    if include_models_endpoint:

        @app.get("/v1/models", dependencies=dependencies)
        async def list_models() -> Mapping[str, Any]:
            return {"object": "list", "data": [{"id": model_id}]}

    @app.get("/v1/metadata", dependencies=dependencies)
    async def metadata() -> Mapping[str, Any]:
        info: dict[str, Any] = {
            "id": model_id,
            "name": model_id,
            "version": model_version,
            "shortName": resolved_short_name,
        }
        resolved_max_batch_size = _resolved_max_batch_size()
        if resolved_max_batch_size is not None:
            info["maxBatchSize"] = resolved_max_batch_size
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
    include_repository_index: bool = True,
    health_dependency: Iterable[Depends] | None = None,
    metadata_dependency: Iterable[Depends] | None = None,
    model_ready_dependency: Iterable[Depends] | None = None,
) -> None:
    """
    Add Triton v2 compatibility endpoints.

    The optional dependency parameters allow distinct auth requirements for
    health versus metadata endpoints while preserving the legacy behavior when
    omitted. Model readiness routes can also be configured independently.
    """

    dependencies = list(auth_dependency or ())
    health_dependencies = list(health_dependency or dependencies)
    metadata_dependencies = list(metadata_dependency or dependencies)
    ready_dependencies = list(model_ready_dependency or metadata_dependencies)

    @app.get("/v2/health/live", dependencies=health_dependencies)
    async def triton_live() -> Mapping[str, bool]:
        return {"live": client.is_live()}

    @app.get("/v2/health/ready", dependencies=health_dependencies)
    async def triton_ready() -> Mapping[str, bool]:
        return {"ready": client.is_ready()}

    if include_repository_index:

        @app.get("/v2/models", dependencies=metadata_dependencies)
        async def list_triton_models() -> Mapping[str, Any]:
            return {"models": client.repository_index()}

    @app.get("/v2/models/{model_name}", dependencies=metadata_dependencies)
    async def model_metadata_http(model_name: str) -> Mapping[str, Any]:
        _validate_model_name(triton_model_name, model_name)
        return client.model_metadata()

    @app.get("/v2/models/{model_name}/config", dependencies=metadata_dependencies)
    async def model_config_http(model_name: str) -> Mapping[str, Any]:
        _validate_model_name(triton_model_name, model_name)
        return client.model_config()

    @app.get("/v2/models/{model_name}/ready", dependencies=ready_dependencies)
    async def model_ready_http(model_name: str) -> Mapping[str, bool]:
        _validate_model_name(triton_model_name, model_name)
        return {"ready": client.is_ready()}


def add_metrics_route(
    app: FastAPI,
    *,
    auth_dependency: Iterable[Depends] | None = None,
    include_triton_route: bool = False,
) -> None:
    """
    Mount /metrics endpoint with optional auth dependency, and optionally expose the Triton alias.
    """

    dependencies = list(auth_dependency or ())

    @app.get("/metrics", dependencies=dependencies)
    async def metrics_endpoint() -> Response:
        return metrics_response()

    if include_triton_route:

        @app.get("/v2/metrics", dependencies=dependencies)
        async def triton_metrics_endpoint() -> Response:
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
