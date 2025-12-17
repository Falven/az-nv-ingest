from __future__ import annotations

import time
from typing import Tuple

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from google.protobuf.json_format import MessageToDict

from .grpc_server import (
    TritonControlService,
    triton_model_config_response,
    triton_model_metadata,
    triton_model_ready_response,
    triton_repository_index_response,
    triton_server_live_response,
    triton_server_ready_response,
)
from .metrics import RequestMetrics
from .models import TritonModelConfig
from .settings import ServiceSettings


def create_router(
    settings: ServiceSettings,
    metrics: RequestMetrics,
    auth_dependency: object,
    start_time: float,
) -> APIRouter:
    """
    Build the FastAPI router exposing health, metadata, metrics, and Triton shims.

    Args:
        settings: Service configuration.
        metrics: Metrics registry wrapper.
        auth_dependency: FastAPI dependency enforcing HTTP auth.
        start_time: Process start timestamp for uptime reporting.

    Returns:
        Configured APIRouter.
    """

    router = APIRouter(dependencies=[auth_dependency])

    @router.get("/")
    async def root() -> JSONResponse:
        """
        Basic root endpoint mirroring the legacy shim.
        """
        return JSONResponse({"status": "SERVING"})

    @router.get("/v1/health/live")
    async def live() -> JSONResponse:
        """
        Liveness check.
        """
        return JSONResponse({"status": "SERVING"})

    @router.get("/v1/health/ready")
    async def ready() -> JSONResponse:
        """
        Readiness check including uptime and model identifiers.
        """
        uptime_seconds = int(time.time() - start_time)
        return JSONResponse(
            {
                "status": "READY",
                "uptime_seconds": uptime_seconds,
                "model": settings.model_name,
                "version": settings.model_version,
            }
        )

    @router.get("/v1/metadata")
    async def metadata() -> JSONResponse:
        """
        Model metadata matching the nv-ingest control-plane expectations.
        """
        short_name = f"{settings.model_name}:{settings.model_version}"
        return JSONResponse(
            {
                "id": settings.model_name,
                "name": settings.model_name,
                "version": settings.model_version,
                "limits": {
                    "maxBatchSize": settings.max_batch_size,
                    "maxConcurrentRequests": settings.max_concurrent_requests,
                    "maxStreamingSessions": settings.max_streaming_sessions,
                },
                "features": {
                    "languageCode": settings.default_language_code,
                    "punctuation": True,
                    "punctuationDefault": settings.default_punctuation,
                    "diarization": True,
                    "diarizationDefault": settings.default_diarization,
                    "endpointing": {
                        "startHistoryMs": settings.endpoint_start_history_ms,
                        "startThreshold": settings.endpoint_start_threshold,
                        "stopHistoryMs": settings.endpoint_stop_history_ms,
                        "stopThreshold": settings.endpoint_stop_threshold,
                        "eouHistoryMs": settings.endpoint_eou_history_ms,
                        "eouThreshold": settings.endpoint_eou_threshold,
                        "vadFrameMs": settings.vad_frame_ms,
                    },
                },
                "modelInfo": [
                    {
                        "name": settings.model_name,
                        "version": settings.model_version,
                        "shortName": short_name,
                    }
                ],
            }
        )

    @router.get("/metrics")
    async def render_metrics() -> Response:
        """
        Prometheus metrics endpoint.
        """
        return metrics.render()

    def _triton_state(
        request: Request,
    ) -> Tuple[TritonControlService, TritonModelConfig]:
        service = getattr(request.app.state, "triton_service", None)
        config = getattr(request.app.state, "triton_config", None)
        if service is None or config is None:
            raise HTTPException(
                status_code=503, detail="triton control plane not initialized"
            )
        return service, config

    @router.get("/v2/health/live")
    async def triton_live(request: Request) -> JSONResponse:
        """
        Triton live endpoint shim.
        """
        payload = MessageToDict(
            triton_server_live_response(), preserving_proto_field_name=True
        )
        return JSONResponse(payload)

    @router.get("/v2/health/ready")
    async def triton_ready(request: Request) -> JSONResponse:
        """
        Triton ready endpoint shim.
        """
        triton_service, _ = _triton_state(request)
        payload = MessageToDict(
            triton_server_ready_response(triton_service.loaded),
            preserving_proto_field_name=True,
        )
        return JSONResponse(payload)

    @router.get("/v2/repository/index")
    async def triton_repository(request: Request) -> JSONResponse:
        """
        Repository index shim for Triton compatibility.
        """
        triton_service, triton_config = _triton_state(request)
        payload = MessageToDict(
            triton_repository_index_response(triton_config, triton_service.loaded),
            preserving_proto_field_name=True,
        )
        return JSONResponse(payload)

    @router.get("/v2/models/{model_name}")
    async def triton_model_metadata_http(
        model_name: str, request: Request
    ) -> JSONResponse:
        """
        Triton model metadata endpoint.
        """
        triton_service, triton_config = _triton_state(request)
        if model_name != triton_config.name:
            raise HTTPException(status_code=404, detail="model not found")
        payload = MessageToDict(
            triton_model_metadata(triton_config),
            preserving_proto_field_name=True,
        )
        return JSONResponse(payload)

    @router.get("/v2/models/{model_name}/config")
    async def triton_model_config_http(
        model_name: str, request: Request
    ) -> JSONResponse:
        """
        Triton model config endpoint.
        """
        _, triton_config = _triton_state(request)
        if model_name != triton_config.name:
            raise HTTPException(status_code=404, detail="model not found")
        payload = MessageToDict(
            triton_model_config_response(triton_config, settings),
            preserving_proto_field_name=True,
        )
        return JSONResponse(payload)

    @router.get("/v2/models/{model_name}/ready")
    async def triton_model_ready_http(
        model_name: str, request: Request
    ) -> JSONResponse:
        """
        Triton model readiness endpoint.
        """
        triton_service, triton_config = _triton_state(request)
        if model_name != triton_config.name:
            return JSONResponse(
                MessageToDict(
                    triton_model_ready_response(False), preserving_proto_field_name=True
                )
            )
        payload = MessageToDict(
            triton_model_ready_response(triton_service.loaded),
            preserving_proto_field_name=True,
        )
        return JSONResponse(payload)

    return router
