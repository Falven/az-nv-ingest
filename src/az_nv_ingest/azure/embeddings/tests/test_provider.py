import json
import sys
import types

import httpx
import pytest

from az_nv_ingest.azure.embeddings.provider import (
    AzureOpenAIEmbeddingConfig,
    azure_embeddings_provider,
    azure_make_async_request,
    load_azure_embedding_config,
)


def test_load_azure_embedding_config_returns_none_when_flag_disabled():
    assert load_azure_embedding_config({}) is None


def test_azure_make_async_request_sends_expected_payload():
    captured_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 2.0, 3.0]}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    result = azure_make_async_request(
        prompts=["hello world"],
        api_key="test-key",
        embedding_nim_endpoint="https://example.openai.azure.com",
        embedding_model="embed-deploy",
        encoding_format="float",
        input_type="passage",
        truncate="END",
        filter_errors=False,
        modalities=None,
        api_version="2024-06-01",
        timeout_seconds=5.0,
        max_retries=0,
        backoff_seconds=0.0,
        max_backoff_seconds=0.0,
        dimensions=None,
        http_client=client,
    )

    assert result["embedding"] == [[1.0, 2.0, 3.0]]
    assert result["info_msg"] is None

    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.method == "POST"
    assert (
        str(request.url)
        == "https://example.openai.azure.com/openai/deployments/embed-deploy/embeddings?api-version=2024-06-01"
    )
    assert request.headers["api-key"] == "test-key"
    payload = json.loads(request.content.decode("utf-8"))
    assert payload["input"] == ["hello world"]
    assert payload["encoding_format"] == "float"


def test_azure_make_async_request_retries_on_429_then_succeeds():
    call_count = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["value"] += 1
        if call_count["value"] == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json={"data": [{"embedding": [9.0, 9.0]}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    result = azure_make_async_request(
        prompts=["retry me"],
        api_key="retry-key",
        embedding_nim_endpoint="https://example.openai.azure.com",
        embedding_model="embed-deploy",
        encoding_format="float",
        input_type="passage",
        truncate="END",
        filter_errors=False,
        modalities=None,
        api_version="2024-06-01",
        timeout_seconds=5.0,
        max_retries=1,
        backoff_seconds=0.0,
        max_backoff_seconds=0.0,
        dimensions=None,
        http_client=client,
    )

    assert call_count["value"] == 2
    assert result["embedding"] == [[9.0, 9.0]]


def test_azure_make_async_request_raises_after_timeouts():
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated timeout")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(RuntimeError) as excinfo:
        azure_make_async_request(
            prompts=["timeout"],
            api_key="timeout-key",
            embedding_nim_endpoint="https://example.openai.azure.com",
            embedding_model="embed-deploy",
            encoding_format="float",
            input_type="passage",
            truncate="END",
            filter_errors=False,
            modalities=None,
            api_version="2024-06-01",
            timeout_seconds=0.1,
            max_retries=1,
            backoff_seconds=0.0,
            max_backoff_seconds=0.0,
            dimensions=None,
            http_client=client,
        )

    assert "timed out" in str(excinfo.value)


def test_azure_embeddings_provider_patches_and_restores():
    module_path = "nv_ingest_api.internal.transform.embed_text"
    def original_callable(*args, **kwargs):
        return "original"
    stub_module = types.SimpleNamespace(_make_async_request=original_callable)
    sys.modules[module_path] = stub_module

    config = AzureOpenAIEmbeddingConfig(
        endpoint="https://example.openai.azure.com",
        deployment="embed-deploy",
        api_key="stub-key",
        max_retries=0,
        backoff_seconds=0.0,
        max_backoff_seconds=0.0,
    )

    with azure_embeddings_provider(config):
        assert stub_module._make_async_request is not original_callable

    assert stub_module._make_async_request is original_callable
    sys.modules.pop(module_path, None)
