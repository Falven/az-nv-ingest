from __future__ import annotations

from typing import Awaitable, Callable

import grpc
from common.auth import ensure_authorized, extract_bearer_token
from fastapi import Request

from .settings import ServiceSettings


def require_http_auth(
    settings: ServiceSettings,
) -> Callable[[Request], Awaitable[None]]:
    """
    FastAPI dependency enforcing bearer auth when configured.

    Args:
        settings: Service settings containing auth requirements and tokens.

    Returns:
        A dependency callable suitable for FastAPI routes.
    """

    async def _dependency(request: Request) -> None:
        """
        Ensure the incoming request is authorized.

        Args:
            request: Incoming HTTP request.

        Raises:
            HTTPException: When authorization fails according to configured rules.
        """
        ensure_authorized(request, settings.auth_tokens, settings.auth_required)

    return _dependency


def authorize_grpc(context: grpc.ServicerContext, settings: ServiceSettings) -> None:
    """
    Enforce bearer auth for gRPC calls using the same rules as HTTP endpoints.

    Args:
        context: gRPC ServicerContext for the active call.
        settings: Service settings containing auth requirements and tokens.

    Raises:
        grpc.RpcError: When authorization fails; surfaced via ``context.abort``.
    """
    metadata = {md.key: md.value for md in context.invocation_metadata()}
    token = extract_bearer_token(metadata)
    has_tokens = bool(settings.auth_tokens)

    if not settings.auth_required:
        if token is None or not has_tokens:
            return

    if not has_tokens:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "authorization token missing")
        return
    if token is None:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "bearer token required")
        return
    if token not in settings.auth_tokens:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid authorization token")
