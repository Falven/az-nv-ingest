from __future__ import annotations

from typing import Mapping

import grpc
from common.auth import ensure_authorized, extract_bearer_token
from fastapi import Depends, Request

from .settings import ServiceSettings


def _allowed_tokens(settings: ServiceSettings):
    tokens = settings.allowed_tokens()
    return tokens


async def require_http_auth(
    request: Request, settings: ServiceSettings = Depends()
) -> None:
    ensure_authorized(request, _allowed_tokens(settings), settings.require_auth)


def ensure_grpc_authorized(
    context: grpc.ServicerContext, settings: ServiceSettings
) -> None:
    tokens = _allowed_tokens(settings)
    if not settings.require_auth or not tokens:
        return
    metadata: Mapping[str, str] = {
        md.key.lower(): md.value for md in context.invocation_metadata()
    }
    token = extract_bearer_token(metadata)
    if token is None or token not in tokens:
        context.abort(
            grpc.StatusCode.UNAUTHENTICATED, "invalid or missing bearer token"
        )
