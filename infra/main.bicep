// Orchestrates az-nv-ingest infra for AKS + optional dependencies. Azure OpenAI is intentionally not deployed.
targetScope = 'subscription'

@description('Azure region for all resources.')
param location string
@description('Environment name used in resource naming (e.g., dev, test, prod).')
param environment string = 'dev'
@description('Common tags applied to all resources.')
param tags object = {}

@description('Resource group for AKS.')
param aksResourceGroupName string = ''
@description('Resource group for platform/shared services (ACR, KV, Insights, Search).')
param platformResourceGroupName string = ''

@description('AKS cluster name.')
param clusterName string = ''
@description('Optional Kubernetes version; leave empty to let Azure pick default.')
param kubernetesVersion string = ''
@description('DNS prefix for API server; leave empty to derive from cluster name/environment.')
param dnsPrefix string = ''
@description('Network plugin for AKS.')
@allowed([
  'azure'
  'kubenet'
])
param networkPlugin string = 'azure'
@allowed([
  'loadBalancer'
  'userDefinedRouting'
])
param outboundType string = 'loadBalancer'
param apiServerAuthorizedIpRanges array = []
param enableKeyVaultCsi bool = true

// System pool
param systemNodeVmSize string = 'Standard_D4s_v5'
@minValue(1)
param systemNodeCount int = 1
@minValue(30)
param systemNodeOsDiskSizeGB int = 120

// GPU pool
param gpuNodePoolName string = 'gpu'
param gpuNodePoolVmSize string = 'Standard_NC12ads_A10_v4'
@minValue(0)
param gpuNodePoolCount int = 1
@minValue(30)
param gpuNodePoolOsDiskSizeGB int = 256

// ACR
param deployAcr bool = true
param acrName string = ''
@allowed([
  'Basic'
  'Standard'
  'Premium'
])
param acrSku string = 'Standard'
param acrAdminUserEnabled bool = false
param attachAcrToAks bool = true

// Key Vault
param deployKeyVault bool = true
param keyVaultName string = ''
param enableKeyVaultPurgeProtection bool = false
@allowed([
  'Enabled'
  'Disabled'
])
param keyVaultPublicNetworkAccess string = 'Enabled'

// App Insights / Log Analytics
param deployAppInsights bool = true
param appInsightsName string = ''
param workspaceName string = ''
@minValue(7)
param logsRetentionInDays int = 30

// Cognitive Search
param deployCognitiveSearch bool = false
param searchServiceName string = ''
@allowed([
  'basic'
  'standard'
  'standard2'
  'standard3'
  'storage_optimized_l1'
  'storage_optimized_l2'
])
param searchSku string = 'standard'
@minValue(1)
@maxValue(12)
param searchReplicaCount int = 1
@minValue(1)
@maxValue(12)
param searchPartitionCount int = 1
@allowed([
  'enabled'
  'disabled'
])
param searchPublicNetworkAccess string = 'enabled'

// Workload identity used by apps (for KV etc.)
param createWorkloadIdentity bool = true
param workloadIdentityName string = ''

var nameSeed = toLower(uniqueString(subscription().id, environment))
var commonTags = union(tags, {
  environment: environment
})

var resolvedAksRg = aksResourceGroupName != '' ? aksResourceGroupName : format('rg-aznvingest-aks-{0}', environment)
var resolvedPlatformRg = platformResourceGroupName != '' ? platformResourceGroupName : format('rg-aznvingest-platform-{0}', environment)
var resolvedClusterName = clusterName != '' ? clusterName : format('aznv-{0}', environment)
var resolvedDnsPrefix = dnsPrefix != '' ? dnsPrefix : substring(format('{0}-{1}', toLower(replace(resolvedClusterName, '_', '-')), nameSeed), 0, 45)
var resolvedAcrName = acrName != '' ? acrName : substring(toLower(format('aznvingest{0}', nameSeed)), 0, 50)
var resolvedKeyVaultName = keyVaultName != '' ? keyVaultName : substring(toLower(format('kvaznvingest{0}', nameSeed)), 0, 24)
var resolvedWorkspaceName = workspaceName != '' ? workspaceName : format('log-aznvingest-{0}', environment)
var resolvedAppInsightsName = appInsightsName != '' ? appInsightsName : format('appi-aznvingest-{0}', environment)
var resolvedSearchName = searchServiceName != '' ? searchServiceName : substring(toLower(format('aznvingest{0}', nameSeed)), 0, 60)
var resolvedWorkloadIdentityName = workloadIdentityName != '' ? workloadIdentityName : format('id-aznvingest-{0}', environment)
var acrResourceId = resourceId(resolvedPlatformRg, 'Microsoft.ContainerRegistry/registries', resolvedAcrName)
var keyVaultResourceId = resourceId(resolvedPlatformRg, 'Microsoft.KeyVault/vaults', resolvedKeyVaultName)

var acrPullRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
var kvSecretsUserRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')

// Resource groups
resource aksRg 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: resolvedAksRg
  location: location
  tags: commonTags
}

resource platformRg 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: resolvedPlatformRg
  location: location
  tags: commonTags
}

// Optional workload identity for apps
module workloadIdentity 'modules/userAssignedIdentity.bicep' = if (createWorkloadIdentity) {
  name: 'workload-identity'
  scope: platformRg
  params: {
    name: resolvedWorkloadIdentityName
    location: location
    tags: commonTags
  }
}

// ACR (optional)
module acr 'modules/acr.bicep' = if (deployAcr) {
  name: 'acr'
  scope: platformRg
  params: {
    name: resolvedAcrName
    location: location
    tags: commonTags
    sku: acrSku
    adminUserEnabled: acrAdminUserEnabled
  }
}

// Log Analytics (only when App Insights is deployed)
module logAnalytics 'modules/logAnalytics.bicep' = if (deployAppInsights) {
  name: 'log'
  scope: platformRg
  params: {
    name: resolvedWorkspaceName
    location: location
    tags: commonTags
    retentionInDays: logsRetentionInDays
  }
}

// Application Insights (optional)
module appInsights 'modules/appInsights.bicep' = if (deployAppInsights) {
  name: 'appi'
  scope: platformRg
  params: {
    name: resolvedAppInsightsName
    location: location
    tags: commonTags
    workspaceResourceId: resourceId(resolvedPlatformRg, 'Microsoft.OperationalInsights/workspaces', resolvedWorkspaceName)
  }
}

// Key Vault (optional)
module keyVault 'modules/keyVault.bicep' = if (deployKeyVault) {
  name: 'kv'
  scope: platformRg
  params: {
    name: resolvedKeyVaultName
    location: location
    tags: commonTags
    tenantId: subscription().tenantId
    enablePurgeProtection: enableKeyVaultPurgeProtection
    publicNetworkAccess: keyVaultPublicNetworkAccess
  }
}

// Cognitive Search (optional)
module search 'modules/cognitiveSearch.bicep' = if (deployCognitiveSearch) {
  name: 'search'
  scope: platformRg
  params: {
    name: resolvedSearchName
    location: location
    tags: commonTags
    sku: searchSku
    replicaCount: searchReplicaCount
    partitionCount: searchPartitionCount
    publicNetworkAccess: searchPublicNetworkAccess
  }
}

// AKS cluster with GPU pool
module aks 'modules/aks.bicep' = {
  name: 'aks'
  scope: aksRg
  params: {
    name: resolvedClusterName
    location: location
    tags: commonTags
    kubernetesVersion: kubernetesVersion
    dnsPrefix: resolvedDnsPrefix
    networkPlugin: networkPlugin
    outboundType: outboundType
    apiServerAuthorizedIpRanges: apiServerAuthorizedIpRanges
    enableKeyVaultCsi: enableKeyVaultCsi
    systemNodeVmSize: systemNodeVmSize
    systemNodeCount: systemNodeCount
    systemNodeOsDiskSizeGB: systemNodeOsDiskSizeGB
    gpuNodePoolName: gpuNodePoolName
    gpuNodePoolVmSize: gpuNodePoolVmSize
    gpuNodePoolCount: gpuNodePoolCount
    gpuNodePoolOsDiskSizeGB: gpuNodePoolOsDiskSizeGB
  }
}

// Attach ACR pull role to kubelet identity when both are present
module acrPullAssignment 'modules/acrRoleAssignment.bicep' = if (deployAcr && attachAcrToAks) {
  name: 'acr-pull'
  scope: platformRg
  params: {
    roleAssignmentName: guid(acrResourceId, acrPullRoleId, 'kubelet')
    registryName: resolvedAcrName
    principalId: aks.outputs.kubeletObjectId
    roleDefinitionId: acrPullRoleId
    principalType: 'ServicePrincipal'
  }
}

// Grant Key Vault access to workload identity when both exist
module kvSecretsAssignment 'modules/keyVaultRoleAssignment.bicep' = if (deployKeyVault && createWorkloadIdentity) {
  name: 'kv-secrets'
  scope: platformRg
  params: {
    roleAssignmentName: guid(keyVaultResourceId, kvSecretsUserRoleId, resolvedWorkloadIdentityName)
    keyVaultName: resolvedKeyVaultName
    principalId: workloadIdentity.outputs.principalId
    roleDefinitionId: kvSecretsUserRoleId
    principalType: 'ServicePrincipal'
  }
}

// Outputs
output aksName string = aks.outputs.name
output aksResourceGroup string = aksRg.name
output aksId string = aks.outputs.id
output kubeletIdentityObjectId string = aks.outputs.kubeletObjectId
output kubeletIdentityClientId string = aks.outputs.kubeletClientId
output kubeletIdentityResourceId string = aks.outputs.kubeletResourceId
output oidcIssuerUrl string = aks.outputs.oidcIssuerUrl
output acrLoginServer string = deployAcr ? acr.outputs.loginServer : ''
output acrId string = deployAcr ? acr.outputs.id : ''
output keyVaultNameOut string = deployKeyVault ? keyVault.outputs.name : ''
output keyVaultUri string = deployKeyVault ? keyVault.outputs.vaultUri : ''
output appInsightsConnectionString string = deployAppInsights ? appInsights.outputs.connectionString : ''
output appInsightsId string = deployAppInsights ? appInsights.outputs.id : ''
output searchEndpoint string = deployCognitiveSearch ? search.outputs.endpoint : ''
output searchResourceId string = deployCognitiveSearch ? search.outputs.id : ''
output workloadIdentityClientId string = createWorkloadIdentity ? workloadIdentity.outputs.clientId : ''
output workloadIdentityPrincipalId string = createWorkloadIdentity ? workloadIdentity.outputs.principalId : ''
