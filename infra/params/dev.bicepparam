using '../main.bicep'

param location = 'eastus2'
param environment = 'dev'
param deployAcr = true
param deployKeyVault = true
param deployAppInsights = true
param deployCognitiveSearch = false
param systemNodeCount = 1
param systemNodeVmSize = 'Standard_D4s_v5'
param gpuNodePoolCount = 1
param gpuNodePoolVmSize = 'Standard_NC12ads_A10_v4'
param logsRetentionInDays = 30
param attachAcrToAks = true
param enableKeyVaultPurgeProtection = false
