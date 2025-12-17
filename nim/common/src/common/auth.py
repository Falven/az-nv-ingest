from __future__ import annotations

from typing import Iterable, Mapping, Set

from fastapi import HTTPException, Request, status


def extract_bearer_token(headers: Mapping[str, str]) -> str | None:
    """
    Extract a bearer token from HTTP-style headers.

    Args:
        headers: Mapping of header names to values (case-insensitive).

    Returns:
        The bearer token string if present; otherwise ``None``.
    """
    auth_header = headers.get("authorization") or headers.get("Authorization")
    if auth_header is None:
        return None
    if not auth_header.lower().startswith("bearer "):
        return None
    token = auth_header[7:].strip()
    return token or None


def ensure_authorized(
    request: Request,
    allowed_tokens: Iterable[str],
    require_auth: bool,
) -> None:
    """
    Validate a request against configured bearer tokens.

    Args:
        request: Incoming FastAPI request to inspect.
        allowed_tokens: Iterable of accepted bearer tokens.
        require_auth: Whether authentication is mandatory even when no tokens
            are configured.

    Raises:
        HTTPException: When authentication is required and the request is missing
            or presents an invalid bearer token.
    """
    token_set: Set[str] = set(allowed_tokens)
    has_tokens = bool(token_set)
    token = extract_bearer_token(request.headers)
    if not require_auth:
        if token is None or not has_tokens:
            return
    if not has_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authorization token missing",
        )
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="bearer token required"
        )
    if token not in token_set:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid authorization token",
        )


def extract_token_with_fallback(headers: Mapping[str, str]) -> str | None:
    """
    Extract a bearer token from Authorization, NGC, or API key headers, tolerating raw tokens.

    Args:
        headers: Mapping of header names to values (case-insensitive).

    Returns:
        The normalized token string if present; otherwise ``None``.
    """
    normalized = {key.lower(): value for key, value in headers.items()}
    auth_header = normalized.get("authorization")
    if auth_header:
        bearer = extract_bearer_token({"authorization": auth_header})
        if bearer:
            return bearer
        token = auth_header.strip()
        if token:
            return token
    ngc_header = normalized.get("ngc-api-key")
    if ngc_header:
        token = ngc_header.strip()
        if token:
            return token
    api_key_header = normalized.get("x-api-key")
    if api_key_header:
        token = api_key_header.strip()
        if token:
            return token
    return None
