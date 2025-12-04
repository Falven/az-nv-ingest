# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pandas as pd

from az_nv_ingest.azure.ai_search import AzureAISearchConfig
from nv_ingest.framework.orchestration.ray.stages.storage import store_embeddings


def _build_df() -> pd.DataFrame:
    metadata = {
        "content": "chunk",
        "embedding": [0.5, 0.6],
        "source_metadata": {
            "source_id": "doc-1",
            "source_location": "https://example/doc-1",
            "source_name": "doc-1",
            "source_type": "pdf",
        },
        "content_metadata": {"type": "text", "page_number": 1, "hierarchy": {}},
    }
    return pd.DataFrame([
        {"document_type": "text", "metadata": metadata, "uuid": "chunk-1"},
    ])


def test_maybe_upload_to_azure_search_uses_vector_store(monkeypatch):
    df = _build_df()
    original_metadata = copy.deepcopy(df.at[0, "metadata"])
    calls = {}

    class _DummyStore:
        def __init__(self, config):
            calls["config"] = config

        def upsert_documents(self, documents):
            calls["documents"] = documents

    def _fake_build_documents(df_store, cfg):
        return [{cfg.id_field: "chunk-1", cfg.embedding_field: [0.5, 0.6], cfg.content_field: "chunk"}]

    monkeypatch.setattr(store_embeddings, "AzureAISearchVectorStore", _DummyStore)
    monkeypatch.setattr(store_embeddings, "build_documents_from_dataframe", _fake_build_documents)

    task_config = {
        "params": {
            "azure_search_endpoint": "https://example.search.windows.net",
            "azure_search_index": "test-index",
            "azure_search_api_key": "key",
        }
    }

    result_df = store_embeddings._maybe_upload_to_azure_search(df, task_config)

    assert calls["documents"][0]["id"] == "chunk-1"
    assert result_df.at[0, "metadata"]["embedding_metadata"]["uploaded_embedding_index"] == "test-index"
    # Ensure original frame was not mutated
    assert "embedding_metadata" not in original_metadata


def test_maybe_upload_to_azure_search_prefers_task_params(monkeypatch):
    df = _build_df()
    calls: dict[str, str] = {}

    def _fake_env_config(_env):
        return AzureAISearchConfig(
            endpoint="https://env.search.windows.net",
            index_name="env-index",
            api_key="env-key",
        )

    def _fake_param_config(params):
        calls["params"] = params
        return AzureAISearchConfig(
            endpoint="https://param.search.windows.net",
            index_name="param-index",
            api_key="param-key",
        )

    class _DummyStore:
        def __init__(self, config):
            self.config = config

        def upsert_documents(self, _):
            calls["used_index"] = self.config.index_name

    monkeypatch.setattr(store_embeddings, "get_azure_search_config", _fake_param_config)
    monkeypatch.setattr(store_embeddings, "get_azure_search_config_from_env", _fake_env_config)
    monkeypatch.setattr(store_embeddings, "AzureAISearchVectorStore", _DummyStore)
    monkeypatch.setattr(store_embeddings, "build_documents_from_dataframe", lambda df_store, cfg: [])

    task_config = {"params": {"azure_search_endpoint": "https://param.search.windows.net"}}
    store_embeddings._maybe_upload_to_azure_search(df, task_config)

    assert calls["used_index"] == "param-index"


def test_maybe_upload_to_azure_search_falls_back_to_env(monkeypatch):
    df = _build_df()
    calls: dict[str, str] = {}

    def _fake_env_config(_env):
        return AzureAISearchConfig(
            endpoint="https://env.search.windows.net",
            index_name="env-index",
            api_key="env-key",
        )

    class _DummyStore:
        def __init__(self, config):
            self.config = config

        def upsert_documents(self, _):
            calls["used_index"] = self.config.index_name

    monkeypatch.setattr(store_embeddings, "get_azure_search_config", lambda params: None)
    monkeypatch.setattr(store_embeddings, "get_azure_search_config_from_env", _fake_env_config)
    monkeypatch.setattr(store_embeddings, "AzureAISearchVectorStore", _DummyStore)
    monkeypatch.setattr(store_embeddings, "build_documents_from_dataframe", lambda df_store, cfg: [])

    result = store_embeddings._maybe_upload_to_azure_search(df, task_config={})

    assert calls["used_index"] == "env-index"
    assert result is not None
