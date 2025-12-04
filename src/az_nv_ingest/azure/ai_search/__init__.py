"""Azure Cognitive Search vector store integration."""

from az_nv_ingest.azure.ai_search.vector_store import (
    AzureAISearchConfig,
    AzureAISearchUpsertError,
    AzureAISearchVectorStore,
    build_search_credential,
    build_documents_from_dataframe,
    get_azure_search_config,
    get_azure_search_config_from_env,
)

__all__ = [
    "AzureAISearchConfig",
    "AzureAISearchUpsertError",
    "AzureAISearchVectorStore",
    "build_search_credential",
    "build_documents_from_dataframe",
    "get_azure_search_config",
    "get_azure_search_config_from_env",
]
