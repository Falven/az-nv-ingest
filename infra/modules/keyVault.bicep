targetScope = 'resourceGroup'

param name string
param location string
param tags object = {}
param tenantId string
@allowed([
  'standard'
  'premium'
])
param skuName string = 'standard'
@description('Enable purge protection; recommended for production.')
param enablePurgeProtection bool = false
@description('Set public network access state.')
@allowed([
  'Enabled'
  'Disabled'
])
param publicNetworkAccess string = 'Enabled'

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    tenantId: tenantId
    sku: {
      family: 'A'
      name: skuName
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    enablePurgeProtection: enablePurgeProtection
    publicNetworkAccess: publicNetworkAccess
  }
}

output id string = vault.id
output name string = vault.name
output vaultUri string = vault.properties.vaultUri
