from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from functools import partial
from typing import Dict, Iterable, Mapping, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError

logger = logging.getLogger(__name__)

# Azure OpenAI GA API version that supports text-embedding-3 models.
DEFAULT_API_VERSION = "2024-06-01"

ENABLE_ENV_FLAG = "AZURE_OPENAI_EMBEDDINGS_ENABLED"
ENDPOINT_ENV = "AZURE_OPENAI_ENDPOINT"
EMBEDDING_ENDPOINT_ENV = "AZURE_OPENAI_EMBEDDING_ENDPOINT"
DEPLOYMENT_ENV = "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
MODEL_ENV = "AZURE_OPENAI_EMBEDDING_MODEL"
API_KEY_ENV = "AZURE_OPENAI_API_KEY"
API_VERSION_ENV = "AZURE_OPENAI_API_VERSION"
TIMEOUT_ENV = "AZURE_OPENAI_EMBEDDINGS_TIMEOUT_SECONDS"
MAX_RETRIES_ENV = "AZURE_OPENAI_EMBEDDINGS_MAX_RETRIES"
BACKOFF_ENV = "AZURE_OPENAI_EMBEDDINGS_BACKOFF_SECONDS"
MAX_BACKOFF_ENV = "AZURE_OPENAI_EMBEDDINGS_MAX_BACKOFF_SECONDS"
DIMENSIONS_ENV = "AZURE_OPENAI_EMBEDDING_DIMENSIONS"


class AzureOpenAIEmbeddingConfig(BaseModel):
    """Validated configuration for Azure OpenAI embeddings requests."""

    model_config = ConfigDict(extra="forbid")

    endpoint: HttpUrl
    deployment: str
    api_key: str
    api_version: str = Field(default=DEFAULT_API_VERSION)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    backoff_seconds: float = Field(default=1.0, gt=0)
    max_backoff_seconds: float = Field(default=15.0, gt=0)
    dimensions: Optional[int] = Field(default=None, gt=0)


def _to_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _parse_float(env_value: Optional[str], default: float) -> float:
    if env_value is None:
        return default
    try:
        return float(env_value)
    except ValueError:
        logger.warning("Invalid float value for Azure OpenAI embeddings config; using default %.2f", default)
        return default


def _parse_int(env_value: Optional[str], default: int) -> int:
    if env_value is None:
        return default
    try:
        return int(env_value)
    except ValueError:
        logger.warning("Invalid integer value for Azure OpenAI embeddings config; using default %d", default)
        return default


def _parse_optional_int(env_value: Optional[str]) -> Optional[int]:
    if env_value is None or env_value.strip() == "":
        return None
    try:
        return int(env_value)
    except ValueError:
        logger.warning("Invalid optional integer for Azure OpenAI embeddings config; ignoring value.")
        return None


def load_azure_embedding_config(env: Mapping[str, str] = os.environ) -> Optional[AzureOpenAIEmbeddingConfig]:
    """
    Build a validated Azure OpenAI embedding configuration from environment variables.

    Returns None when the feature flag is off or required values are missing, allowing callers
    to fall back to the default embedding provider without raising.
    """
    if not _to_bool(env.get(ENABLE_ENV_FLAG)):
        return None

    endpoint = env.get(ENDPOINT_ENV) or env.get(EMBEDDING_ENDPOINT_ENV)
    deployment = env.get(DEPLOYMENT_ENV) or env.get(MODEL_ENV)
    api_key = env.get(API_KEY_ENV)

    api_version = env.get(API_VERSION_ENV) or DEFAULT_API_VERSION
    timeout_seconds = _parse_float(env.get(TIMEOUT_ENV), default=30.0)
    max_retries = _parse_int(env.get(MAX_RETRIES_ENV), default=3)
    backoff_seconds = _parse_float(env.get(BACKOFF_ENV), default=1.0)
    max_backoff_seconds = _parse_float(env.get(MAX_BACKOFF_ENV), default=15.0)
    dimensions = _parse_optional_int(env.get(DIMENSIONS_ENV))

    if endpoint is None or deployment is None or api_key is None:
        logger.warning(
            "Azure OpenAI embeddings enabled but endpoint, deployment, or API key missing; "
            "falling back to default provider."
        )
        return None

    try:
        return AzureOpenAIEmbeddingConfig(
            endpoint=endpoint,
            deployment=deployment,
            api_key=api_key,
            api_version=api_version,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
            dimensions=dimensions,
        )
    except ValidationError as exc:
        logger.warning("Invalid Azure OpenAI embeddings configuration (%s); using default provider.", exc)
        return None


def _truncate_error(message: str) -> str:
    if len(message) <= 500:
        return message
    return f"{message[:200]}... [truncated] ...{message[-100:]}"


def _build_embeddings_url(endpoint: str, deployment: str) -> str:
    normalized_endpoint = endpoint.rstrip("/")
    return f"{normalized_endpoint}/openai/deployments/{deployment}/embeddings"


def azure_make_async_request(
    prompts: Iterable[str],
    api_key: str,
    embedding_nim_endpoint: str,
    embedding_model: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
    filter_errors: bool,
    modalities: Optional[list[str]] = None,
    *,
    api_version: str,
    timeout_seconds: float,
    max_retries: int,
    backoff_seconds: float,
    max_backoff_seconds: float,
    dimensions: Optional[int],
    http_client: Optional[httpx.Client] = None,
) -> Dict[str, object]:
    """
    Request embeddings from Azure OpenAI and mirror nv_ingest_api._make_async_request shape.

    The parameters `input_type`, `truncate`, `filter_errors`, and `modalities` are accepted
    for signature compatibility; Azure does not currently consume them.
    """
    _ = (input_type, truncate, filter_errors, modalities)

    url = _build_embeddings_url(embedding_nim_endpoint, embedding_model)
    request_body: Dict[str, object] = {"input": list(prompts)}
    if encoding_format is not None:
        request_body["encoding_format"] = encoding_format
    if dimensions is not None:
        request_body["dimensions"] = dimensions

    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=timeout_seconds)
    headers = {"api-key": api_key}

    last_error: Optional[str] = None

    try:
        for attempt in range(max_retries + 1):
            try:
                response = client.post(
                    url,
                    params={"api-version": api_version},
                    headers=headers,
                    json=request_body,
                    timeout=timeout_seconds,
                )
            except httpx.TimeoutException as err:
                last_error = f"Request timed out after {timeout_seconds:.2f}s: {err}"
                if attempt < max_retries:
                    time.sleep(min(max_backoff_seconds, backoff_seconds * (2**attempt)))
                    continue
                raise RuntimeError(_truncate_error(last_error)) from err
            except httpx.HTTPError as err:
                last_error = f"HTTP error contacting Azure OpenAI embeddings: {err}"
                raise RuntimeError(_truncate_error(last_error)) from err

            if response.status_code == httpx.codes.OK:
                try:
                    body = response.json()
                except ValueError as err:
                    raise RuntimeError("Azure OpenAI embeddings returned non-JSON response") from err

                data = body.get("data", [])
                embeddings = [item.get("embedding") for item in data]
                return {"embedding": embeddings, "info_msg": None}

            if response.status_code == httpx.codes.TOO_MANY_REQUESTS or response.status_code >= 500:
                last_error = (
                    f"Azure OpenAI embeddings request failed with status {response.status_code}: "
                    f"{response.text}"
                )
                if attempt < max_retries:
                    time.sleep(min(max_backoff_seconds, backoff_seconds * (2**attempt)))
                    continue
                raise RuntimeError(_truncate_error(last_error))

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as err:
                raise RuntimeError(_truncate_error(str(err))) from err

        raise RuntimeError(_truncate_error(last_error or "Azure OpenAI embeddings request failed."))
    finally:
        if owns_client:
            client.close()


@contextmanager
def azure_embeddings_provider(config: AzureOpenAIEmbeddingConfig):
    """
    Temporarily route nv_ingest_api embedding requests through Azure OpenAI.
    """
    from nv_ingest_api.internal.transform import embed_text as embed_text_module

    original = embed_text_module._make_async_request
    azure_request = partial(
        azure_make_async_request,
        api_version=config.api_version,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
        backoff_seconds=config.backoff_seconds,
        max_backoff_seconds=config.max_backoff_seconds,
        dimensions=config.dimensions,
    )
    embed_text_module._make_async_request = azure_request  # type: ignore[attr-defined]
    try:
        yield
    finally:
        embed_text_module._make_async_request = original  # type: ignore[attr-defined]

