from __future__ import annotations

from oim_common.auth import extract_token_with_fallback
from fastapi import Depends, HTTPException, Request, status

from .settings import ServiceSettings


def require_http_auth(settings: ServiceSettings) -> Depends:
    """
    FastAPI dependency enforcing bearer authentication with legacy compatibility.

    Args:
        settings: Service configuration containing auth flags and tokens.

    Returns:
        A ``Depends`` instance that validates incoming requests.

    Raises:
        HTTPException: When authentication is required but missing or invalid.
    """

    async def _dependency(request: Request) -> None:
        token = extract_token_with_fallback(request.headers)
        tokens = settings.auth_tokens
        has_tokens = bool(tokens)

        if not settings.require_auth and not has_tokens:
            return

        if has_tokens:
            if token is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="bearer token required",
                )
            if token not in tokens:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="invalid authorization token",
                )
            return

        if settings.require_auth and token is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="bearer token required",
            )

    return Depends(_dependency)
