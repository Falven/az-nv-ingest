#!/usr/bin/env bash
set -euo pipefail

export OIM_APP_MODULE="oim_nemoretriever_ocr_v1.server:app"
export OIM_WAIT_FOR_READY=0
export TRITON_HTTP_PORT_DEFAULT=8003
export TRITON_GRPC_PORT_DEFAULT=8001
export TRITON_METRICS_PORT_DEFAULT=8002
export TRITON_MODEL_REPO_DEFAULT="${TRITON_MODEL_REPOSITORY:-/models}"

exec /app/common-entrypoint.sh
