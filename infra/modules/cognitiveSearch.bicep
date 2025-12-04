targetScope = 'resourceGroup'

param name string
param location string
param tags object = {}
@allowed([
  'basic'
  'standard'
  'standard2'
  'standard3'
  'storage_optimized_l1'
  'storage_optimized_l2'
])
param sku string = 'standard'
@minValue(1)
@maxValue(12)
param replicaCount int = 1
@minValue(1)
@maxValue(12)
param partitionCount int = 1
@allowed([
  'enabled'
  'disabled'
])
param publicNetworkAccess string = 'enabled'

resource search 'Microsoft.Search/searchServices@2023-11-01' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: sku
  }
  properties: {
    partitionCount: partitionCount
    replicaCount: replicaCount
    publicNetworkAccess: publicNetworkAccess
    hostingMode: 'default'
  }
  identity: {
    type: 'SystemAssigned'
  }
}

output id string = search.id
output name string = search.name
output endpoint string = format('https://{0}.search.windows.net', search.name)
output principalId string = search.identity.principalId
