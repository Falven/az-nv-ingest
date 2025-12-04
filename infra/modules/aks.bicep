targetScope = 'resourceGroup'

param name string
param location string
param tags object = {}
@description('Optional Kubernetes version; leave empty to use the default for the region.')
param kubernetesVersion string = ''
param dnsPrefix string
@allowed([
  'azure'
  'kubenet'
])
param networkPlugin string = 'azure'
@description('Outbound type for the cluster load balancer.')
@allowed([
  'loadBalancer'
  'userDefinedRouting'
])
param outboundType string = 'loadBalancer'

@description('System node pool VM size.')
param systemNodeVmSize string = 'Standard_D4s_v5'
@minValue(1)
param systemNodeCount int = 1
@minValue(30)
param systemNodeOsDiskSizeGB int = 120

@description('GPU node pool name (<=12 chars).')
param gpuNodePoolName string = 'gpu'
@description('GPU node pool VM size.')
param gpuNodePoolVmSize string = 'Standard_NC12ads_A10_v4'
@minValue(0)
param gpuNodePoolCount int = 1
@minValue(30)
param gpuNodePoolOsDiskSizeGB int = 256

param apiServerAuthorizedIpRanges array = []
param workloadIdentityEnabled bool = true
param oidcIssuerEnabled bool = true
param enableKeyVaultCsi bool = true

var apiServerProfile = length(apiServerAuthorizedIpRanges) == 0 ? null : {
  authorizedIpRanges: apiServerAuthorizedIpRanges
}

var addonProfiles = enableKeyVaultCsi ? {
  azureKeyvaultSecretsProvider: {
    enabled: true
    config: {
      useManagedIdentity: 'true'
    }
  }
} : {}

var baseProperties = {
  dnsPrefix: dnsPrefix
  enableRBAC: true
  agentPoolProfiles: [
    {
      name: 'system'
      vmSize: systemNodeVmSize
      count: systemNodeCount
      osDiskSizeGB: systemNodeOsDiskSizeGB
      osType: 'Linux'
      mode: 'System'
      type: 'VirtualMachineScaleSets'
      enableNodePublicIP: false
      upgradeSettings: {
        maxSurge: '33%'
      }
    }
    {
      name: gpuNodePoolName
      vmSize: gpuNodePoolVmSize
      count: gpuNodePoolCount
      osDiskSizeGB: gpuNodePoolOsDiskSizeGB
      osType: 'Linux'
      mode: 'User'
      type: 'VirtualMachineScaleSets'
      enableNodePublicIP: false
      nodeLabels: {
        accelerator: 'nvidia'
      }
      nodeTaints: [
        'sku=gpu:NoSchedule'
      ]
      upgradeSettings: {
        maxSurge: '33%'
      }
    }
  ]
  networkProfile: {
    networkPlugin: networkPlugin
    loadBalancerSku: 'standard'
    outboundType: outboundType
  }
  workloadIdentityProfile: {
    enabled: workloadIdentityEnabled
  }
  oidcIssuerProfile: {
    enabled: oidcIssuerEnabled
  }
  addonProfiles: addonProfiles
}

resource aks 'Microsoft.ContainerService/managedClusters@2024-02-01' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: union(
    baseProperties,
    kubernetesVersion == '' ? {} : {
      kubernetesVersion: kubernetesVersion
    },
    apiServerProfile == null ? {} : {
      apiServerAccessProfile: apiServerProfile
    }
  )
}

output id string = aks.id
output name string = aks.name
output principalId string = aks.identity.principalId
output kubeletObjectId string = aks.properties.identityProfile.kubeletidentity.objectId
output kubeletClientId string = aks.properties.identityProfile.kubeletidentity.clientId
output kubeletResourceId string = aks.properties.identityProfile.kubeletidentity.resourceId
output nodeResourceGroup string = aks.properties.nodeResourceGroup
output oidcIssuerUrl string = aks.properties.oidcIssuerProfile.issuerURL
