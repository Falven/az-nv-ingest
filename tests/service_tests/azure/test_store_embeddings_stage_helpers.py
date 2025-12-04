# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pandas as pd

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
