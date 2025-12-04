targetScope = 'resourceGroup'

param roleAssignmentName string
param registryName string
param principalId string
param roleDefinitionId string
@allowed([
  'ServicePrincipal'
  'User'
  'Group'
  'Application'
])
param principalType string = 'ServicePrincipal'

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: registryName
}

resource assignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: roleAssignmentName
  scope: registry
  properties: {
    principalId: principalId
    roleDefinitionId: roleDefinitionId
    principalType: principalType
  }
}

output id string = assignment.id
