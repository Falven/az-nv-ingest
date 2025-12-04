"""Helpers to provision Azure Cognitive Search indexes for az-nv-ingest embeddings."""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Optional

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from az_nv_ingest.azure.ai_search import (
    AzureAISearchConfig,
    build_search_credential,
    get_azure_search_config_from_env,
)


def _build_fields(config: AzureAISearchConfig, vector_dimensions: int) -> Iterable[SearchField]:
    vector_field = SearchField(
        name=config.embedding_field,
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=vector_dimensions,
        vector_search_profile_name="aznv-hnsw-profile",
    )

    filterable_metadata = SearchField(
        name=config.filterable_metadata_field,
        type=SearchFieldDataType.ComplexType,
        fields=[
            SimpleField(name="source_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="source_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="collection_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="partition_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="content_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="content_subtype", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SimpleField(name="document_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        ],
    )

    return [
        SimpleField(name=config.id_field, type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name=config.content_field, type=SearchFieldDataType.String),
        vector_field,
        filterable_metadata,
        SimpleField(name="document_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="content_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="content_subtype", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name="source_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="source_name", type=SearchFieldDataType.String, filterable=False, searchable=True),
        SimpleField(name="source_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="collection_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="partition_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
    ]


def build_index_schema(config: AzureAISearchConfig, vector_dimensions: int) -> SearchIndex:
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="aznv-hnsw-config",
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="aznv-hnsw-profile",
                algorithm_configuration_name="aznv-hnsw-config",
            )
        ],
    )

    return SearchIndex(
        name=config.index_name,
        fields=list(_build_fields(config, vector_dimensions)),
        vector_search=vector_search,
    )


def ensure_index_exists(
    config: AzureAISearchConfig,
    vector_dimensions: int,
    *,
    env: Mapping[str, str] | None = None,
    client: Optional[SearchIndexClient] = None,
) -> SearchIndex:
    credential = build_search_credential(config, env=env)
    index_client = client or SearchIndexClient(endpoint=config.endpoint, credential=credential)
    index_schema = build_index_schema(config, vector_dimensions)
    return index_client.create_or_update_index(index=index_schema)


def _to_bool(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _get_vector_dimensions(env: Mapping[str, str]) -> Optional[int]:
    raw_value = env.get("AZURE_COGNITIVE_SEARCH_VECTOR_DIMENSIONS")
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except ValueError:
        return None


def main() -> None:  # pragma: no cover - convenience CLI
    env = os.environ
    if not _to_bool(env.get("AZURE_COGNITIVE_SEARCH_CREATE_INDEX")):
        print("AZURE_COGNITIVE_SEARCH_CREATE_INDEX is not enabled; skipping index creation.")
        return

    config = get_azure_search_config_from_env(env)
    if config is None:
        raise SystemExit("Azure Cognitive Search environment variables are incomplete; cannot create index.")

    dimensions = _get_vector_dimensions(env)
    if dimensions is None or dimensions <= 0:
        raise SystemExit("Set AZURE_COGNITIVE_SEARCH_VECTOR_DIMENSIONS to a positive integer.")

    ensure_index_exists(config, dimensions, env=env)
    print(f"Ensured Azure Cognitive Search index '{config.index_name}' exists at {config.endpoint}.")


if __name__ == "__main__":  # pragma: no cover
    main()
