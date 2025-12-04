targetScope = 'resourceGroup'

param roleAssignmentName string
param keyVaultName string
param principalId string
param roleDefinitionId string
@allowed([
  'ServicePrincipal'
  'User'
  'Group'
  'Application'
])
param principalType string = 'ServicePrincipal'

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

resource assignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: roleAssignmentName
  scope: vault
  properties: {
    principalId: principalId
    roleDefinitionId: roleDefinitionId
    principalType: principalType
  }
}

output id string = assignment.id
