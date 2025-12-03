# infra/ — AGENTS.md (Bicep IaC)

## Scope

Bicep code to deploy infrastructure prerequisites for az-nv-ingest on AKS.

## Hard constraints

- **DO NOT deploy Azure OpenAI resources**. Only accept existing AOAI endpoint info as parameters.
- Use modular Bicep. Prefer Azure Verified Modules (AVM) wherever possible.
- Keep modules composable: resource group scope + subscription scope separation.

## What infra MUST deploy/configure

- Resource Group(s) structure for platform/app
- AKS cluster with:
  - a **GPU node pool** (NVIDIA GPUs)
  - Managed Identity
  - OIDC issuer/workload identity enabled (preferred for Key Vault access)
- ACR (optional but recommended) for your built images
- Azure Monitor / Application Insights (optional toggle) for OTEL ingestion
- Azure Key Vault (optional toggle) + RBAC assignments for AKS workload identity
- Azure Cognitive Search (optional toggle) + identity-based access

## Directory structure (recommended)

- `infra/main.bicep` (orchestration only; minimal logic)
- `infra/modules/*` (one module per major resource)
- `infra/params/*.bicepparam` (environments: dev/test/prod)
- `infra/tests/*` (what-if scripts, validation)
- `infra/README.md` (how to deploy)

## Outputs (must be emitted)

- AKS name, RG, kubelet identity / workload identity details
- ACR login server
- Key Vault name + URI
- App Insights connection string or resource id
- Cognitive Search endpoint + resource id

## Deployment guidance

- Provide CLI examples using `az deployment sub create` / `group create`.
- Provide “what-if” commands.
- Provide a minimal dev profile that works without private networking first (then add private options).
