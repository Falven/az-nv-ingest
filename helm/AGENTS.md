# helm/ — AGENTS.md (Kubernetes deployment)

## Scope

Helm charts for deploying az-nv-ingest on AKS (GPU).

## az-nv-ingest requirements via Helm values

Must support:

- AKS + NVIDIA GPU nodes
- Optional Azure Cognitive Search integration
- Optional Azure Key Vault integration (usually via CSI driver)
- Optional Application Insights export (OTEL exporter)
- External embeddings via Azure OpenAI (no in-cluster embedding service)

## Rules

- Keep values.yaml backwards compatible when feasible.
- Prefer feature toggles like:
  - `azureCognitiveSearch.enabled`
  - `azureKeyVault.enabled`
  - `azureOpenAI.enabled`
  - `applicationInsights.enabled`

## GPU-specific requirements

- Ensure pods request GPU resources (`nvidia.com/gpu: 1`) where needed.
- Provide clear nodeSelector / tolerations guidance (AKS GPU node pool).
- Do not hardcode GPU model assumptions.

## Deliverables

- Document all new values with examples.
- Provide a minimal “open” profile (no NIM, no NGC key).
