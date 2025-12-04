# Azure compat matrix (scratch)

Quick mapping of Helm values → env vars → runtime readers for Azure features.

| Helm value | Env var(s) emitted | Runtime consumer | Notes |
| --- | --- | --- | --- |
| `azureOpenAI.enabled` | `AZURE_OPENAI_EMBEDDINGS_ENABLED`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION` | `az_nv_ingest.azure.embeddings.load_azure_embedding_config` | Flag now set by chart; no extra manual env required. |
| `azureCognitiveSearch.enabled` + `endpoint/index/authMode` | `AZURE_COGNITIVE_SEARCH_ENDPOINT`, `AZURE_COGNITIVE_SEARCH_INDEX`, `AZURE_COGNITIVE_SEARCH_AUTH_MODE`, `AZURE_COGNITIVE_SEARCH_API_KEY` (or `AZURE_CLIENT_ID` for MI) | `get_azure_search_config_from_env` → `_maybe_upload_to_azure_search` | Task params still take precedence; env provides default upload target. |
| `azureKeyVault.runtime.*` | `AZURE_KEY_VAULT_URI`, `AZURE_KEY_VAULT_SECRET_MAPPINGS` | `load_key_vault_secrets` | Runtime hydration path; CSI mounts remain separate/optional. |
| `applicationInsights.roleName` | `APPLICATIONINSIGHTS_ROLE_NAME` | `configure_tracer` | Sets `service.namespace` resource attribute; idempotent tracer config. |
| `workloadIdentity.*` | ServiceAccount label `azure.workload.identity/use`, annotation `azure.workload.identity/client-id` | Azure workload identity (Key Vault / Search / App Insights) | Requires matching federated credential against AKS OIDC issuer. |
