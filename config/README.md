# Observability and secret toggles

- **App Insights (traces):** set `APPLICATIONINSIGHTS_CONNECTION_STRING` (or `APPINSIGHTS_CONNECTION_STRING`).
  Falls back to `OTEL_EXPORTER_OTLP_ENDPOINT` when absent. Headers can be added with
  `OTEL_EXPORTER_OTLP_HEADERS` (comma-separated `key=value`).
- **Key Vault (opt-in):** set `AZURE_KEY_VAULT_URI` and `AZURE_KEY_VAULT_SECRET_MAPPINGS`
  (comma-separated `ENV_VAR=secret-name`). Values are fetched with workload identity when
  `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, and `AZURE_FEDERATED_TOKEN_FILE` are present; otherwise
  the DefaultAzureCredential chain is used. Existing environment values are left untouched.
- Keep secrets out of configs; map Key Vault secrets to the env vars above when deploying via Helm/IaC.
