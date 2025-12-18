from __future__ import annotations

import grpc
from fastapi import Depends
from oim_common.auth import AuthValidator, build_http_auth_dependency

from .settings import ServiceSettings


def require_http_auth(settings: ServiceSettings) -> Depends:
    """FastAPI dependency enforcing bearer/NGC auth when configured."""

    return build_http_auth_dependency(settings)


def authorize_grpc(context: grpc.ServicerContext, settings: ServiceSettings) -> None:
    """Enforce bearer/NGC auth for gRPC calls using the same rules as HTTP endpoints."""

    validator = AuthValidator(settings)
    metadata = {md.key.lower(): md.value for md in context.invocation_metadata()}
    try:
        validator.validate_metadata(metadata)
    except PermissionError as exc:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, str(exc))
