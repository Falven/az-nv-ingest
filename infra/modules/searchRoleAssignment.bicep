targetScope = 'resourceGroup'

param roleAssignmentName string
param searchServiceName string
param principalId string
param roleDefinitionId string
@allowed([
  'ServicePrincipal'
  'User'
  'Group'
  'Application'
])
param principalType string = 'ServicePrincipal'

resource search 'Microsoft.Search/searchServices@2023-11-01' existing = {
  name: searchServiceName
}

resource assignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: roleAssignmentName
  scope: search
  properties: {
    principalId: principalId
    roleDefinitionId: roleDefinitionId
    principalType: principalType
  }
}

output id string = assignment.id
