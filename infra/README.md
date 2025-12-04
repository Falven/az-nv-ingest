# az-nv-ingest infra (AKS + dependencies)

Azure Bicep templates to stand up the az-nv-ingest runtime on AKS with optional platform services. **Azure OpenAI is not deployed by this stack.**

## Layout
- `main.bicep` – subscription-scope orchestration, resource groups + optional services
- `modules/` – discrete resource modules (AKS, ACR, Key Vault, App Insights/Log Analytics, Cognitive Search, identities, role assignment)
- `params/dev.bicepparam` – minimal dev defaults
- `tests/what-if-dev.sh` – helper to run `az deployment sub what-if` against dev params

## What this deploys
- Resource groups for AKS (`rg-aznvingest-aks-<env>`) and platform services (`rg-aznvingest-platform-<env>`)
- AKS cluster with system pool + GPU user pool, managed identity, OIDC/workload identity, optional Key Vault CSI add-on
- Optional ACR (system-assigned MI) and `AcrPull` role assignment to AKS kubelet identity
- Optional Key Vault (RBAC) with `Key Vault Secrets User` for the workload identity
- Optional Log Analytics + Application Insights
- Optional Azure Cognitive Search (system-assigned MI)

## Quickstart (subscription-scope)
1. `az login` and `az account set -s <subscription-id>`
2. Inspect/override params in `infra/params/dev.bicepparam` (e.g., region, GPU SKU, toggles)
3. What-if:
   ```bash
   cd infra
   ./tests/what-if-dev.sh
   ```
4. Deploy:
   ```bash
   az deployment sub create \
     --name az-nv-ingest-dev \
     --location eastus2 \
     --template-file main.bicep \
     --parameters @params/dev.bicepparam
   ```

## Key parameters (main.bicep)
- `environment` (default `dev`): suffix for names/tags
- `deployAcr|deployKeyVault|deployAppInsights|deployCognitiveSearch` (bool toggles)
- `systemNodeVmSize`, `gpuNodePoolVmSize`, `gpuNodePoolCount`: AKS sizing (GPU pool is tainted `sku=gpu:NoSchedule`)
- `attachAcrToAks` (default true): creates `AcrPull` role assignment to kubelet identity when ACR is deployed
- `createWorkloadIdentity` (default true): user-assigned identity for workloads; granted `Key Vault Secrets User` when KV is deployed
- `keyVaultPublicNetworkAccess`, `enableKeyVaultPurgeProtection`: KV security knobs
- `searchSku`, `searchReplicaCount`, `searchPartitionCount`: Cognitive Search sizing

Name defaults use `uniqueString(subscription().id, environment)` to keep global uniqueness for ACR/KV/Search.

## Outputs
- AKS: `aksName`, `aksResourceGroup`, `aksId`, `kubeletIdentity*`, `oidcIssuerUrl`
- ACR: `acrLoginServer`, `acrId` (empty when `deployAcr=false`)
- Key Vault: `keyVaultNameOut`, `keyVaultUri` (empty when `deployKeyVault=false`)
- App Insights: `appInsightsConnectionString`, `appInsightsId` (empty when `deployAppInsights=false`)
- Cognitive Search: `searchEndpoint`, `searchResourceId` (empty when `deployCognitiveSearch=false`)
- Workload identity: `workloadIdentityClientId`, `workloadIdentityPrincipalId` (empty when `createWorkloadIdentity=false`)

## Notes
- AKS OIDC issuer and workload identity are enabled by default. Use the emitted `oidcIssuerUrl` + `workloadIdentityClientId` when creating Kubernetes service accounts with Azure Workload Identity.
- Default GPU size `Standard_NC12ads_A10_v4` is available in many regions; adjust if unsupported in your target region.
- Key Vault and ACR names are alphanumeric to satisfy global naming rules; override via parameters if you need deterministic names.
