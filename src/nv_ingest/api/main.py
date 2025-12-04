# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from fastapi import FastAPI

from az_nv_ingest.azure.key_vault import load_key_vault_secrets
from az_nv_ingest.azure.observability import configure_tracer

from .v1.health import router as HealthApiRouter
from .v1.ingest import router as IngestApiRouter
from .v1.metrics import router as MetricsApiRouter

logger = logging.getLogger(__name__)

# nv-ingest FastAPI app declaration
app = FastAPI(
    title="NV-Ingest Microservice",
    description="Service for ingesting heterogenous datatypes",
    version="25.9.0",
    contact={
        "name": "NVIDIA Corporation",
        "url": "https://nvidia.com",
    },
    docs_url="/docs",
)

app.include_router(IngestApiRouter, prefix="/v1")
app.include_router(HealthApiRouter, prefix="/v1/health")
app.include_router(MetricsApiRouter, prefix="/v1")

# Optionally hydrate secrets from Key Vault before wiring telemetry
load_key_vault_secrets()

# Configure tracing: defaults to OTLP; switches to App Insights when a
# connection string is present (env or Key Vault)
tracer = configure_tracer(service_name=os.getenv("OTEL_SERVICE_NAME", "nv-ingest"))
