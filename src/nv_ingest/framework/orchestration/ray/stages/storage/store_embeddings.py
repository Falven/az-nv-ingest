# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any, Optional

import pandas as pd
import ray

from az_nv_ingest.azure.ai_search import (
    AzureAISearchVectorStore,
    build_documents_from_dataframe,
    get_azure_search_config,
    get_azure_search_config_from_env,
)
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.store.store_embedding_schema import EmbeddingStorageSchema
from nv_ingest_api.internal.store.embed_text_upload import store_text_embeddings_internal
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)

logger = logging.getLogger(__name__)


@ray.remote
class EmbeddingStorageStage(RayActorStage):
    """
    A Ray actor stage that stores text embeddings in the configured backend.

    It expects an IngestControlMessage containing a DataFrame with embedding data. It then:
      1. Removes the "store_embedding" task from the message.
      2. Attempts an Azure Cognitive Search upsert when the task supplies Azure config.
      3. Falls back to the default MinIO/Milvus upload path when Azure config is absent.
      4. Updates the message payload with the stored embeddings DataFrame.
    """

    def __init__(self, config: EmbeddingStorageSchema) -> None:
        super().__init__(config)
        try:
            self.validated_config = config
            logger.info("EmbeddingStorageStage configuration validated successfully.")
        except Exception as e:
            logger.exception(f"Error validating Embedding Storage config: {e}")
            raise

    @traceable("embedding_storage")
    @filter_by_task(required_tasks=["store_embedding"])
    @nv_ingest_node_failure_try_except(annotation_id="embedding_storage", raise_on_failure=False)
    def on_data(self, control_message: IngestControlMessage) -> IngestControlMessage:
        """
        Process the control message by storing embeddings.

        Parameters
        ----------
        control_message : IngestControlMessage
            The message containing a DataFrame payload with embedding data.

        Returns
        -------
        IngestControlMessage
            The updated message with embeddings stored in MinIO.
        """
        logger.info("EmbeddingStorageStage.on_data: Starting embedding storage process.")

        # Extract the DataFrame payload.
        df_ledger = control_message.payload()
        logger.debug("Extracted payload with %d rows.", len(df_ledger))

        # Remove the "store_embedding" task from the message to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "store_embedding")
        logger.debug("Extracted task config: %s", task_config)

        azure_upload_df = _maybe_upload_to_azure_search(df_ledger, task_config)
        if azure_upload_df is not None:
            control_message.payload(azure_upload_df)
            return control_message

        # Perform embedding storage.
        new_df = store_text_embeddings_internal(
            df_store_ledger=df_ledger,
            task_config=task_config,
            store_config=self.validated_config,
            execution_trace_log=None,
        )
        logger.info("Embedding storage completed. Resulting DataFrame has %d rows.", len(new_df))

        # Update the message payload with the stored embeddings DataFrame.
        control_message.payload(new_df)

        return control_message


def _maybe_upload_to_azure_search(df_store_ledger: pd.DataFrame, task_config: Any) -> Optional[pd.DataFrame]:
    if hasattr(task_config, "model_dump"):
        task_config = task_config.model_dump()

    params = (task_config or {}).get("params")
    azure_config = get_azure_search_config(params) or get_azure_search_config_from_env(os.environ)
    if not azure_config:
        return None

    logger.info("Using Azure Cognitive Search vector store: index=%s", azure_config.index_name)
    vector_store = AzureAISearchVectorStore(azure_config)
    documents = build_documents_from_dataframe(df_store_ledger, azure_config)
    vector_store.upsert_documents(documents)
    return _annotate_embedding_metadata(df_store_ledger, azure_config.index_name)


def _annotate_embedding_metadata(df_store_ledger: pd.DataFrame, index_name: str) -> pd.DataFrame:
    updated_df = df_store_ledger.copy()
    for idx, row in updated_df.iterrows():
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("embedding") is None:
            continue

        updated_metadata = metadata.copy()
        embedding_metadata = updated_metadata.get("embedding_metadata") or {}
        if not isinstance(embedding_metadata, dict):
            embedding_metadata = {}

        embedding_metadata["uploaded_embedding_index"] = index_name
        updated_metadata["embedding_metadata"] = embedding_metadata
        updated_df.at[idx, "metadata"] = updated_metadata

    return updated_df
