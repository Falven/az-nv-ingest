#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

deployment_name="az-nv-ingest-dev-whatif"
location="eastus2"

echo "Running what-if for dev params in ${location}..."
az deployment sub what-if \
  --name "${deployment_name}" \
  --location "${location}" \
  --template-file "${ROOT_DIR}/main.bicep" \
  --parameters @"${ROOT_DIR}/params/dev.bicepparam"
