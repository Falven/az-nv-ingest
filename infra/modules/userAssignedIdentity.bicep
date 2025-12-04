targetScope = 'resourceGroup'

param name string
param location string
@description('Resource tags applied to the identity.')
param tags object = {}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: name
  location: location
  tags: tags
}

output resourceId string = identity.id
output clientId string = identity.properties.clientId
output principalId string = identity.properties.principalId
