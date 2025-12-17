from __future__ import annotations

from typing import Awaitable, Callable, Mapping

from oim_common.auth import extract_token_with_fallback
from fastapi import HTTPException, Request, status

from .settings import ServiceSettings


def extract_token(headers: Mapping[str, str]) -> str | None:
    """
    Extract a bearer or raw token from HTTP-style headers.
    """
    return extract_token_with_fallback(headers)


def _is_authorized(token: str | None, settings: ServiceSettings) -> bool:
    if not settings.auth_required:
        return True
    if token is None:
        return False
    if settings.auth_tokens:
        return token in settings.auth_tokens
    return True


def require_http_auth(
    settings: ServiceSettings,
) -> Callable[[Request], Awaitable[None]]:
    """
    Build a FastAPI dependency enforcing bearer authentication.
    """

    async def _dependency(request: Request) -> None:
        token = extract_token(request.headers)
        if _is_authorized(token, settings):
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )

    return _dependency
