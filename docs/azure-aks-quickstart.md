# Azure AKS quickstart (aoai + cognitive search + kv)

Minimal end-to-end path to stand up az-nv-ingest on AKS with Azure OpenAI embeddings, Azure Cognitive Search storage, and Key Vault runtime hydration.

## Prereqs
- Azure CLI, kubectl, helm
- An existing Azure OpenAI endpoint + embedding deployment
- Optional: App Insights instance/connection string

## 1) Deploy infra
```bash
cd infra
az deployment sub create \
  --name az-nv-ingest-dev \
  --location eastus2 \
  --template-file main.bicep \
  --parameters @params/dev.bicepparam \
  --parameters deployCognitiveSearch=true createWorkloadIdentity=true
```

Record outputs: `searchEndpoint`, `workloadIdentityClientId`, `oidcIssuerUrl`, `keyVaultUri` (if enabled).

## 2) Bind workload identity to your Kubernetes service account
```bash
az identity federated-credential create \
  --name az-nv-ingest \
  --identity-name <workload-identity-name> \
  --resource-group <platform-rg> \
  --issuer "<oidcIssuerUrl>" \
  --subject "system:serviceaccount:<namespace>:<service-account-name>" \
  --audiences "api://AzureADTokenExchange"
```

## 3) Provision the Cognitive Search index (one-time)
```bash
export AZURE_COGNITIVE_SEARCH_ENDPOINT=https://<search-name>.search.windows.net
export AZURE_COGNITIVE_SEARCH_INDEX=nv-ingest
export AZURE_COGNITIVE_SEARCH_AUTH_MODE=managedIdentity   # or apiKey + AZURE_COGNITIVE_SEARCH_API_KEY
export AZURE_CLIENT_ID=<workloadIdentityClientId>
export AZURE_COGNITIVE_SEARCH_VECTOR_DIMENSIONS=1536
export AZURE_COGNITIVE_SEARCH_CREATE_INDEX=1
python -m az_nv_ingest.azure.ai_search.index
```

## 4) Helm install with Azure values
Create `values-azure-dev.yaml`:
```yaml
azureOpenAI:
  enabled: true
  endpoint: https://<aoai>.openai.azure.com
  embeddingDeployment: text-embedding-3-large
  apiKey:
    createSecret: true
    value: "<aoai-api-key>"

azureCognitiveSearch:
  enabled: true
  endpoint: https://<search-name>.search.windows.net
  indexName: nv-ingest
  authMode: managedIdentity
  managedIdentityClientId: "<workloadIdentityClientId>"

azureKeyVault:
  runtime:
    enabled: true
    vaultUri: "<keyVaultUri>"
    secretMappings: "AZURE_OPENAI_API_KEY=aoai-api-key,AZURE_COGNITIVE_SEARCH_API_KEY=search-admin-key"

workloadIdentity:
  enabled: true
  clientId: "<workloadIdentityClientId>"
```

Install:
```bash
helm upgrade --install nv-ingest ./helm -f values-azure-dev.yaml -n <namespace> --create-namespace
```

## 5) Smoke ingest
Submit a small job against the service (port-forward or use service DNS):
```bash
curl -X POST http://<svc-host>:7670/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"ingest": {"ingest_id": "smoke-1", "document": {"uri": "https://raw.githubusercontent.com/NVIDIA/nv-ingest/main/data/multimodal_test.pdf"}}, "tasks": [{"task_type":"extract"},{"task_type":"store_embedding"}]}'
```

Verify:
- Embeddings show `embedding_metadata.uploaded_embedding_index` in the response.
- Search index `nv-ingest` contains new documents (e.g., via Azure Portal/Search Explorer).
- Application Insights shows traces (if enabled).
