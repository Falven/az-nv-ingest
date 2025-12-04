"""Azure Cognitive Search vector store adapter for nv-ingest chunks."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar, Literal

import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from az_nv_ingest.azure.key_vault import build_key_vault_credential
from nv_ingest_api.internal.enums.common import ContentTypeEnum

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AzureAISearchUpsertError(RuntimeError):
    """Raised when Azure Cognitive Search upsert requests fail."""


class AzureAISearchConfig(BaseModel):
    endpoint: str
    index_name: str
    auth_mode: Literal["apiKey", "managedIdentity"] = "apiKey"
    api_key: Optional[str] = None
    managed_identity_client_id: Optional[str] = None
    embedding_field: str = Field(default="vector")
    content_field: str = Field(default="content")
    id_field: str = Field(default="id")
    filterable_metadata_field: str = Field(default="filterable_metadata")
    raw_metadata_field: str = Field(default="metadata")
    include_raw_metadata: bool = Field(default=True)
    batch_size: int = Field(default=64, ge=1)

    model_config = ConfigDict(extra="forbid")

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        if not value:
            raise ValueError("Azure Search endpoint is required")
        return value.rstrip("/")

    @field_validator("index_name", "api_key")
    @classmethod
    def validate_non_empty(cls, value: str, info: Any) -> str:
        if not value:
            raise ValueError(f"{info.field_name.replace('_', ' ')} is required")
        return value

    @model_validator(mode="after")
    def validate_auth(self) -> "AzureAISearchConfig":
        if self.auth_mode == "apiKey" and not self.api_key:
            raise ValueError("api_key is required when auth_mode='apiKey'")
        return self


def _normalize_auth_mode(value: str | None) -> str:
    if value is None:
        return "apiKey"
    normalized = value.strip().lower()
    if normalized in {"apikey", "api_key"}:
        return "apiKey"
    if normalized in {"managedidentity", "managed_identity"}:
        return "managedIdentity"
    return value


def _chunked(items: Sequence[T], size: int) -> Iterable[List[T]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _select_content_text(
    metadata: Dict[str, Any],
    document_type: Optional[str],
    content_metadata: Dict[str, Any],
) -> str:
    content_type = content_metadata.get("type") or document_type
    if content_type in {ContentTypeEnum.IMAGE.value, ContentTypeEnum.STRUCTURED.value}:
        source_metadata = metadata.get("source_metadata") or {}
        return source_metadata.get("source_location", "") or ""

    return metadata.get("content") or ""


def _build_filter_metadata(
    source_metadata: Dict[str, Any], content_metadata: Dict[str, Any], document_type: Optional[str]
) -> Dict[str, Any]:
    return {
        "source_id": source_metadata.get("source_id"),
        "source_type": source_metadata.get("source_type"),
        "collection_id": source_metadata.get("collection_id"),
        "partition_id": source_metadata.get("partition_id"),
        "content_type": content_metadata.get("type"),
        "content_subtype": content_metadata.get("subtype"),
        "page_number": content_metadata.get("page_number"),
        "document_type": document_type,
    }


def build_documents_from_dataframe(df: pd.DataFrame, config: AzureAISearchConfig) -> List[Dict[str, Any]]:
    """Map an nv-ingest chunk DataFrame to Azure Search documents."""

    if df.empty:
        return []

    documents: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Row metadata must be a dictionary")

        embedding = metadata.get("embedding")
        if embedding is None:
            continue

        source_metadata = metadata.get("source_metadata") or {}
        content_metadata = metadata.get("content_metadata") or {}

        doc_id = row.get("uuid") or source_metadata.get("source_id")
        if not doc_id:
            raise ValueError(f"Row {idx} missing uuid and source_id; cannot build Azure Search document")

        filter_metadata = _build_filter_metadata(source_metadata, content_metadata, row.get("document_type"))
        content_text = _select_content_text(metadata, row.get("document_type"), content_metadata)

        document: Dict[str, Any] = {
            config.id_field: str(doc_id),
            config.content_field: content_text,
            config.embedding_field: embedding,
            config.filterable_metadata_field: filter_metadata,
            "document_type": row.get("document_type"),
            "content_type": content_metadata.get("type"),
            "content_subtype": content_metadata.get("subtype"),
            "page_number": content_metadata.get("page_number"),
            "source_id": source_metadata.get("source_id"),
            "source_name": source_metadata.get("source_name"),
            "source_type": source_metadata.get("source_type"),
            "collection_id": source_metadata.get("collection_id"),
            "partition_id": source_metadata.get("partition_id"),
        }

        if config.include_raw_metadata:
            document[config.raw_metadata_field] = metadata

        documents.append(document)

    return documents


def get_azure_search_config(params: Optional[Dict[str, Any]]) -> Optional[AzureAISearchConfig]:
    """Create an AzureAISearchConfig from task params if all required fields are present."""

    if not params:
        return None

    endpoint = params.get("azure_search_endpoint") or params.get("azure_ai_search_endpoint")
    index_name = params.get("azure_search_index") or params.get("azure_ai_search_index_name")
    api_key = params.get("azure_search_api_key") or params.get("azure_ai_search_api_key")
    auth_mode = _normalize_auth_mode(params.get("azure_search_auth_mode"))
    managed_identity_client_id = params.get("azure_search_managed_identity_client_id")

    if not endpoint or not index_name:
        return None

    if (auth_mode or "").lower() == "apikey" and not api_key:
        return None

    return AzureAISearchConfig(
        endpoint=endpoint,
        index_name=index_name,
        api_key=api_key,
        auth_mode=auth_mode,
        managed_identity_client_id=managed_identity_client_id,
        embedding_field=params.get("azure_search_embedding_field", "vector"),
        content_field=params.get("azure_search_content_field", "content"),
        id_field=params.get("azure_search_id_field", "id"),
        filterable_metadata_field=params.get("azure_search_filter_field", "filterable_metadata"),
        raw_metadata_field=params.get("azure_search_raw_metadata_field", "metadata"),
        include_raw_metadata=params.get("azure_search_include_raw_metadata", True),
        batch_size=params.get("azure_search_batch_size", 64),
    )


def get_azure_search_config_from_env(env: Mapping[str, str]) -> Optional[AzureAISearchConfig]:
    """Create an AzureAISearchConfig from environment variables."""

    endpoint = env.get("AZURE_COGNITIVE_SEARCH_ENDPOINT")
    index_name = env.get("AZURE_COGNITIVE_SEARCH_INDEX")
    auth_mode = _normalize_auth_mode(env.get("AZURE_COGNITIVE_SEARCH_AUTH_MODE") or "apiKey")
    api_key = env.get("AZURE_COGNITIVE_SEARCH_API_KEY")
    managed_identity_client_id = env.get("AZURE_CLIENT_ID")

    if not endpoint or not index_name:
        return None

    if auth_mode.lower() == "apikey" and not api_key:
        return None

    return AzureAISearchConfig(
        endpoint=endpoint,
        index_name=index_name,
        auth_mode=auth_mode,
        api_key=api_key,
        managed_identity_client_id=managed_identity_client_id,
    )


def build_search_credential(
    config: AzureAISearchConfig,
    env: Mapping[str, str] | None = None,
):
    if config.auth_mode == "apiKey":
        if not config.api_key:
            raise AzureAISearchUpsertError("api_key is required when auth_mode='apiKey'")
        return AzureKeyCredential(config.api_key)

    env_with_client = dict(env or os.environ)
    if config.managed_identity_client_id and not env_with_client.get("AZURE_CLIENT_ID"):
        env_with_client["AZURE_CLIENT_ID"] = config.managed_identity_client_id

    return build_key_vault_credential(env_with_client)


class AzureAISearchVectorStore:
    """Adapter for idempotent Azure Cognitive Search upserts."""

    def __init__(
        self,
        config: AzureAISearchConfig,
        client: Optional[SearchClient] = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self._env = env or os.environ
        self._client = client or SearchClient(
            endpoint=config.endpoint,
            index_name=config.index_name,
            credential=self._build_credential(),
        )

    def upsert_documents(self, documents: Sequence[Dict[str, Any]]) -> None:
        if not documents:
            logger.info("No embeddings to upsert into Azure Search index '%s'", self.config.index_name)
            return

        for batch in _chunked(documents, self.config.batch_size):
            logger.info(
                "Upserting %s embeddings into Azure Search index '%s'", len(batch), self.config.index_name
            )
            results = self._client.merge_or_upload_documents(documents=batch)
            self._raise_on_failure(batch, results)

    def upsert_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        documents = build_documents_from_dataframe(df, self.config)
        self.upsert_documents(documents)
        return documents

    def _raise_on_failure(
        self, batch: Sequence[Dict[str, Any]], results: Optional[Sequence[Any]]
    ) -> None:
        if results is None:
            raise AzureAISearchUpsertError("Azure Search returned no results for merge_or_upload_documents")

        if len(results) != len(batch):
            raise AzureAISearchUpsertError(
                f"Result length mismatch: got {len(results)} results for {len(batch)} documents"
            )

        for doc, result in zip(batch, results):
            if not getattr(result, "succeeded", False):
                error_message = getattr(result, "error_message", "unknown error")
                raise AzureAISearchUpsertError(
                    f"Azure Search upsert failed for id '{doc.get(self.config.id_field)}': {error_message}"
            )

    def _build_credential(self):
        return build_search_credential(self.config, env=self._env)
