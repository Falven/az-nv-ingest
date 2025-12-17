from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Optional, Protocol, Set

try:  # pragma: no cover - optional dependency
    import grpc
except Exception:  # pragma: no cover - optional dependency
    grpc = None  # type: ignore[assignment]

from fastapi import Depends, HTTPException, Request, status


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


class AuthSettings(Protocol):
    """
    Minimal auth configuration required to build HTTP/gRPC dependencies.
    """

    auth_tokens: Set[str]
    auth_required: bool


def build_http_auth_dependency(
    settings: AuthSettings,
    *,
    allow_unauthenticated_paths: Optional[Set[str]] = None,
) -> Depends:
    """
    Construct a FastAPI dependency enforcing bearer authentication.

    Args:
        settings: Settings object exposing ``auth_tokens`` and ``auth_required``.
        allow_unauthenticated_paths: Set of request paths that may skip auth
            when authentication is disabled.

    Returns:
        FastAPI ``Depends`` marker for route dependencies.
    """

    allowed_paths = allow_unauthenticated_paths or set()
    tokens = set(settings.auth_tokens)
    require_auth = bool(settings.auth_required)

    async def _dependency(request: Request) -> None:
        allowlisted = request.url.path in allowed_paths
        if allowlisted and not require_auth:
            return
        ensure_authorized(request, tokens, require_auth)

    return Depends(_dependency)


class AuthValidator:
    """
    Validates bearer and NGC tokens for gRPC metadata or HTTP headers.
    """

    def __init__(self, settings: AuthSettings):
        self.allowed_tokens = set(settings.auth_tokens)
        self.enabled = bool(settings.auth_required)

    def validate_metadata(
        self, metadata: Mapping[str, str], allow_unauthenticated: bool = False
    ) -> None:
        if not self.enabled or allow_unauthenticated:
            return
        token = extract_token_with_fallback(metadata)
        if token and (not self.allowed_tokens or token in self.allowed_tokens):
            return
        raise PermissionError("authentication failed")

    def validate_headers(
        self, headers: Mapping[str, str], allow_unauthenticated: bool = False
    ) -> None:
        if not self.enabled or allow_unauthenticated:
            return
        token = extract_token_with_fallback(headers)
        if token and (not self.allowed_tokens or token in self.allowed_tokens):
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
        )


class AuthInterceptor:  # pragma: no cover - thin gRPC wrapper
    """
    gRPC interceptor enforcing metadata authentication.
    """

    def __init__(self, validator: AuthValidator):
        if grpc is None:
            raise RuntimeError("grpcio is required for AuthInterceptor")
        self.validator = validator

    def intercept_service(self, continuation, handler_call_details):  # type: ignore[override]
        handler = continuation(handler_call_details)
        if handler is None or not self.validator.enabled:
            return handler

        def metadata_from_context(context) -> MutableMapping[str, str]:
            return {
                key.lower(): value
                for key, value in (context.invocation_metadata() or [])
            }

        def enforce_auth(context) -> None:
            try:
                self.validator.validate_metadata(metadata_from_context(context))
            except PermissionError as exc:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, str(exc))

        if handler.unary_unary:

            def unary_unary(request, context):  # type: ignore[no-untyped-def]
                enforce_auth(context)
                return handler.unary_unary(request, context)

            return grpc.unary_unary_rpc_method_handler(
                unary_unary,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.stream_unary:

            def stream_unary(request_iter, context):  # type: ignore[no-untyped-def]
                enforce_auth(context)
                return handler.stream_unary(request_iter, context)

            return grpc.stream_unary_rpc_method_handler(
                stream_unary,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.unary_stream:

            def unary_stream(request, context):  # type: ignore[no-untyped-def]
                enforce_auth(context)
                return handler.unary_stream(request, context)

            return grpc.unary_stream_rpc_method_handler(
                unary_stream,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.stream_stream:

            def stream_stream(request_iter, context):  # type: ignore[no-untyped-def]
                enforce_auth(context)
                return handler.stream_stream(request_iter, context)

            return grpc.stream_stream_rpc_method_handler(
                stream_stream,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        return handler


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
