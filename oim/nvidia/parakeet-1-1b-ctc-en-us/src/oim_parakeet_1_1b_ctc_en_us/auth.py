from __future__ import annotations

from typing import Mapping, MutableMapping, Optional, Set

import grpc
from oim_common.auth import extract_token_with_fallback
from fastapi import Depends, HTTPException, Request, status

from .settings import ServiceSettings


class AuthValidator:
    """
    Validates bearer and NGC tokens for both gRPC metadata and HTTP headers.
    """

    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.allowed_tokens = settings.auth_tokens
        self.enabled = settings.auth_required

    def validate_metadata(
        self, metadata: Mapping[str, str], allow_unauthenticated: bool = False
    ) -> None:
        """
        Validate gRPC invocation metadata.
        """
        if not self.enabled or allow_unauthenticated:
            return
        token = extract_token_with_fallback(metadata)
        if token and (not self.allowed_tokens or token in self.allowed_tokens):
            return
        raise PermissionError("authentication failed")

    def validate_headers(
        self, headers: Mapping[str, str], allow_unauthenticated: bool = False
    ) -> None:
        """
        Validate HTTP headers for bearer/NGC tokens.
        """
        if not self.enabled or allow_unauthenticated:
            return
        token = extract_token_with_fallback(headers)
        if token and (not self.allowed_tokens or token in self.allowed_tokens):
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
        )


class AuthInterceptor(grpc.ServerInterceptor):
    """
    Enforces token validation for all gRPC calls.
    """

    def __init__(self, validator: AuthValidator):
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


def build_http_auth_dependency(
    validator: AuthValidator,
    allow_unauthenticated_health: bool,
    public_paths: Optional[Set[str]] = None,
) -> Depends:
    """
    Construct a FastAPI dependency enforcing HTTP authentication.
    """
    allowed = public_paths or set()

    async def dependency(request: Request) -> None:
        allow = allow_unauthenticated_health and request.url.path in allowed
        validator.validate_headers(request.headers, allow_unauthenticated=allow)

    return Depends(dependency)
