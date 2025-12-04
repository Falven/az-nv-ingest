{{/*
Expand the name of the chart.
*/}}
{{- define "nv-ingest.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "nv-ingest.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "nv-ingest.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "nv-ingest.labels" -}}
helm.sh/chart: {{ include "nv-ingest.chart" . }}
{{ include "nv-ingest.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "nv-ingest.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nv-ingest.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "nv-ingest.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "nv-ingest.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create secret to access docker registry
*/}}
{{- define "nv-ingest.ngcImagePullSecret" }}
{{- printf "{\"auths\": {\"%s\": {\"auth\": \"%s\"}}}" .Values.ngcImagePullSecret.registry (printf "%s:%s" .Values.ngcImagePullSecret.username .Values.ngcImagePullSecret.password | b64enc) | b64enc }}
{{- end }}

{{/*
  Create secret to access NGC Api
  */}}
  {{- define "nv-ingest.ngcApiSecret" }}
  {{- printf "%s" .Values.ngcApiSecret.password  }}
  {{- end }}

{{/*
Resolve the secret name used for Azure Cognitive Search API keys.
*/}}
{{- define "nv-ingest.azureCognitiveSearch.secretName" -}}
{{- $default := printf "%s-azcognitivesearch" (include "nv-ingest.fullname" .) -}}
{{- coalesce .Values.azureCognitiveSearch.apiKey.existingSecret .Values.azureCognitiveSearch.apiKey.secretName $default -}}
{{- end }}

{{/*
Resolve the secret name used for Application Insights connection strings.
*/}}
{{- define "nv-ingest.applicationInsights.secretName" -}}
{{- $default := printf "%s-appinsights" (include "nv-ingest.fullname" .) -}}
{{- coalesce .Values.applicationInsights.existingSecret .Values.applicationInsights.secretName $default -}}
{{- end }}

{{/*
Resolve the secret name used for Azure OpenAI API keys.
*/}}
{{- define "nv-ingest.azureOpenAI.secretName" -}}
{{- $default := printf "%s-azure-openai" (include "nv-ingest.fullname" .) -}}
{{- coalesce .Values.azureOpenAI.apiKey.existingSecret .Values.azureOpenAI.apiKey.secretName $default -}}
{{- end }}

{{/*
Resolve the SecretProviderClass name for Azure Key Vault CSI mounts.
*/}}
{{- define "nv-ingest.azureKeyVault.secretProviderClassName" -}}
{{- $default := printf "%s-akv" (include "nv-ingest.fullname" .) -}}
{{- coalesce .Values.azureKeyVault.secretProviderClass.existingName .Values.azureKeyVault.secretProviderClass.name $default -}}
{{- end }}

{{/*
Render the Azure Key Vault objects list into the format expected by the CSI driver.
*/}}
{{- define "nv-ingest.azureKeyVault.objects" -}}
{{- if .Values.azureKeyVault.secretProviderClass.objects }}
array:
{{- range $obj := .Values.azureKeyVault.secretProviderClass.objects }}
  - |
    objectName: {{ required "azureKeyVault.secretProviderClass.objects[].objectName is required" $obj.objectName }}
    objectType: {{ default "secret" $obj.objectType }}
    {{- if $obj.objectVersion }}
    objectVersion: {{ $obj.objectVersion }}
    {{- end }}
    {{- if $obj.objectAlias }}
    objectAlias: {{ $obj.objectAlias }}
    {{- end }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Build the Azure OpenAI base URL (adds deployment path when both endpoint and deployment are supplied).
*/}}
{{- define "nv-ingest.azureOpenAI.baseURL" -}}
{{- $endpoint := trimSuffix "/" .Values.azureOpenAI.endpoint -}}
{{- if and $endpoint .Values.azureOpenAI.embeddingDeployment }}
{{- printf "%s/openai/deployments/%s" $endpoint .Values.azureOpenAI.embeddingDeployment -}}
{{- else }}
{{- $endpoint -}}
{{- end }}
{{- end }}
