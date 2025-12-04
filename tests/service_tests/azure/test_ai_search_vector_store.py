# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from az_nv_ingest.azure.ai_search import (
    AzureAISearchConfig,
    AzureAISearchUpsertError,
    AzureAISearchVectorStore,
    build_documents_from_dataframe,
    get_azure_search_config,
)


def _build_sample_df(content: str = "hello world", document_type: str = "text") -> pd.DataFrame:
    metadata = {
        "content": content,
        "embedding": [0.1, 0.2],
        "source_metadata": {
            "source_id": "doc-1",
            "source_name": "doc-1.pdf",
            "source_type": "pdf",
            "source_location": "s3://bucket/doc-1.pdf",
            "collection_id": "collection-a",
            "partition_id": 7,
        },
        "content_metadata": {
            "type": document_type,
            "subtype": "",
            "page_number": 1,
            "hierarchy": {},
        },
    }

    return pd.DataFrame([
        {
            "document_type": document_type,
            "metadata": metadata,
            "uuid": "chunk-1",
        }
    ])


def test_get_azure_search_config_returns_none_when_incomplete():
    assert get_azure_search_config({"azure_search_endpoint": "https://example"}) is None


def test_build_documents_from_dataframe_maps_fields():
    df = _build_sample_df()
    config = AzureAISearchConfig(endpoint="https://example.search.windows.net", index_name="idx", api_key="key")

    documents = build_documents_from_dataframe(df, config)

    assert documents[0][config.id_field] == "chunk-1"
    assert documents[0][config.content_field] == "hello world"
    assert documents[0][config.embedding_field] == [0.1, 0.2]
    assert documents[0][config.filterable_metadata_field]["source_id"] == "doc-1"


def test_build_documents_from_dataframe_uses_source_location_for_structured():
    df = _build_sample_df(content="large-payload", document_type="structured")
    config = AzureAISearchConfig(endpoint="https://example.search.windows.net", index_name="idx", api_key="key")

    documents = build_documents_from_dataframe(df, config)

    assert documents[0][config.content_field] == "s3://bucket/doc-1.pdf"


def test_upsert_documents_batches_and_succeeds():
    class _Result:
        def __init__(self):
            self.succeeded = True

    class _Client:
        def __init__(self):
            self.calls = []

        def merge_or_upload_documents(self, documents):
            self.calls.append(documents)
            return [_Result() for _ in documents]

    config = AzureAISearchConfig(
        endpoint="https://example.search.windows.net",
        index_name="idx",
        api_key="key",
        batch_size=1,
    )
    store = AzureAISearchVectorStore(config=config, client=_Client())

    documents = [
        {config.id_field: "a", config.embedding_field: [0.1], config.content_field: "one"},
        {config.id_field: "b", config.embedding_field: [0.2], config.content_field: "two"},
    ]

    store.upsert_documents(documents)

    assert len(store._client.calls) == 2
    assert store._client.calls[0][0][config.id_field] == "a"


def test_upsert_documents_raises_on_failure():
    class _Result:
        def __init__(self, succeeded: bool, error_message: str = ""):
            self.succeeded = succeeded
            self.error_message = error_message

    class _Client:
        def merge_or_upload_documents(self, documents):
            return [_Result(False, "boom") for _ in documents]

    config = AzureAISearchConfig(endpoint="https://example.search.windows.net", index_name="idx", api_key="key")
    store = AzureAISearchVectorStore(config=config, client=_Client())

    with pytest.raises(AzureAISearchUpsertError):
        store.upsert_documents([{config.id_field: "a", config.embedding_field: [0.1], config.content_field: "one"}])

