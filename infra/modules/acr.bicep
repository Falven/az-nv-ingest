targetScope = 'resourceGroup'

param name string
param location string
param tags object = {}
@allowed([
  'Basic'
  'Standard'
  'Premium'
])
param sku string = 'Standard'
@description('Enable the admin user; recommended to keep false when using managed identities.')
param adminUserEnabled bool = false

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: name
  location: location
  sku: {
    name: sku
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    adminUserEnabled: adminUserEnabled
    publicNetworkAccess: 'Enabled'
  }
  tags: tags
}

output loginServer string = registry.properties.loginServer
output id string = registry.id
output name string = registry.name
