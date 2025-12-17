from __future__ import annotations

import logging
from typing import Mapping

import pytest
from oim_common.auth import (
    ensure_authorized,
    extract_bearer_token,
    extract_token_with_fallback,
)
from oim_common.logging import configure_logging, get_logger
from oim_common.settings import CommonSettings
from fastapi import HTTPException, Request


def _make_request(headers: Mapping[str, str]) -> Request:
    """
    Build a Starlette/FastAPI Request with the provided headers.
    """
    scope = {
        "type": "http",
        "headers": [
            (key.lower().encode(), value.encode()) for key, value in headers.items()
        ],
    }
    return Request(scope)


def test_common_settings_collects_env_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NGC_API_KEY", "token-a")
    monkeypatch.setenv("NIM_NGC_API_KEY", "token-b")
    monkeypatch.setenv("NVIDIA_API_KEY", "token-c")

    settings = CommonSettings()

    assert settings.require_auth is True
    assert settings.log_level == "INFO"
    assert settings.resolved_auth_tokens() == {"token-a", "token-b", "token-c"}


def test_extract_bearer_token_parses_authorization_header() -> None:
    assert extract_bearer_token({"Authorization": "Bearer primary"}) == "primary"
    assert extract_bearer_token({"authorization": "bearer second"}) == "second"
    assert extract_bearer_token({"Authorization": "Token nope"}) is None
    assert extract_bearer_token({}) is None


def test_extract_token_with_fallback_prefers_authorization_then_api_keys() -> None:
    assert (
        extract_token_with_fallback(
            {
                "Authorization": "Bearer preferred",
                "ngc-api-key": "secondary",
                "x-api-key": "tertiary",
            }
        )
        == "preferred"
    )
    assert extract_token_with_fallback({"authorization": "raw-token"}) == "raw-token"
    assert extract_token_with_fallback({"NGC-API-KEY": "ngc-token"}) == "ngc-token"
    assert (
        extract_token_with_fallback({"X-API-Key": "fallback-token"}) == "fallback-token"
    )
    assert extract_token_with_fallback({}) is None


def test_ensure_authorized_enforces_and_allows_optional_auth() -> None:
    allowed = {"valid-token"}
    ensure_authorized(
        _make_request({"Authorization": "Bearer valid-token"}),
        allowed,
        require_auth=True,
    )

    with pytest.raises(HTTPException) as missing:
        ensure_authorized(_make_request({}), allowed, require_auth=True)
    assert missing.value.status_code == 401
    assert missing.value.detail == "bearer token required"

    with pytest.raises(HTTPException) as invalid:
        ensure_authorized(
            _make_request({"Authorization": "Bearer other"}),
            allowed,
            require_auth=False,
        )
    assert invalid.value.status_code == 401
    assert invalid.value.detail == "invalid authorization token"

    ensure_authorized(_make_request({}), set(), require_auth=False)


def test_configure_logging_is_idempotent_and_uppercases_levels() -> None:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    try:
        configured = configure_logging("debug")
        assert configured is root_logger
        assert len(root_logger.handlers) == 1
        assert root_logger.level == logging.DEBUG

        configure_logging("warning")
        assert len(root_logger.handlers) == 1
        assert root_logger.level == logging.WARNING

        module_logger = get_logger("oim_common.tests", level="error")
        assert module_logger.level == logging.ERROR
    finally:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)
