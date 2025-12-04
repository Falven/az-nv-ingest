# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from az_nv_ingest.azure.ai_search import AzureAISearchConfig
from az_nv_ingest.azure.ai_search import index as ai_index


def test_ensure_index_exists_builds_index(monkeypatch):
    calls = {}

    def fake_build_search_credential(config, env=None):
        calls["env"] = env
        return "cred"

    class FakeIndexClient:
        def __init__(self, endpoint, credential):
            self.endpoint = endpoint
            self.credential = credential
            calls["endpoint"] = endpoint
            calls["credential"] = credential

        def create_or_update_index(self, index):
            calls["index"] = index
            return index

    monkeypatch.setattr(ai_index, "build_search_credential", fake_build_search_credential)
    monkeypatch.setattr(ai_index, "SearchIndexClient", FakeIndexClient)

    config = AzureAISearchConfig(endpoint="https://example.search.windows.net", index_name="idx", api_key="key")
    index = ai_index.ensure_index_exists(config, vector_dimensions=1536, env={"AZURE_CLIENT_ID": "client-id"})

    assert calls["endpoint"] == "https://example.search.windows.net"
    assert calls["credential"] == "cred"
    assert calls["index"].name == "idx"
    vector_fields = [field for field in index.fields if field.name == config.embedding_field]
    assert vector_fields[0].vector_search_dimensions == 1536


def test_get_vector_dimensions_parses_int():
    assert ai_index._get_vector_dimensions({"AZURE_COGNITIVE_SEARCH_VECTOR_DIMENSIONS": "10"}) == 10
    assert ai_index._get_vector_dimensions({"AZURE_COGNITIVE_SEARCH_VECTOR_DIMENSIONS": "not-int"}) is None
