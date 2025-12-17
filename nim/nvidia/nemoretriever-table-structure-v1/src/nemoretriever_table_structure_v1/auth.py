from __future__ import annotations

from typing import Awaitable, Callable

import grpc
from common.auth import extract_token_with_fallback
from fastapi import HTTPException, Request, status

from .settings import ServiceSettings


def require_http_auth(
    settings: ServiceSettings,
) -> Callable[[Request], Awaitable[None]]:
    """FastAPI dependency enforcing bearer/NGC auth when configured."""

    async def _dependency(request: Request) -> None:
        token = extract_token_with_fallback(request.headers)
        allowed_tokens = settings.auth_tokens
        has_tokens = bool(allowed_tokens)
        if not settings.auth_required:
            if token is None or not has_tokens:
                return
        if not has_tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="authorization token missing",
            )
        if token is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="bearer token required",
            )
        if token not in allowed_tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid authorization token",
            )

    return _dependency


def authorize_grpc(context: grpc.ServicerContext, settings: ServiceSettings) -> None:
    """Enforce bearer/NGC auth for gRPC calls using the same rules as HTTP endpoints."""

    metadata = {md.key.lower(): md.value for md in context.invocation_metadata()}
    token = extract_token_with_fallback(metadata)
    allowed_tokens = settings.auth_tokens
    has_tokens = bool(allowed_tokens)

    if not settings.auth_required:
        if token is None or not has_tokens:
            return

    if not has_tokens:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "authorization token missing")
        return
    if token is None:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "bearer token required")
        return
    if token not in allowed_tokens:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid authorization token")
