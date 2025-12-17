#!/usr/bin/env bash
set -euo pipefail

export OIM_APP_MODULE="oim_llama_3_2_nv_rerankqa_1b_v2.server:app"
export OIM_EXPORT_TRITON_URLS=0
export TRITON_HTTP_PORT_DEFAULT=8003
export TRITON_GRPC_PORT_DEFAULT=8004
export TRITON_METRICS_PORT_DEFAULT=8005
export TRITON_MODEL_REPO_DEFAULT="/app/triton/model_repository"
export TRITON_READY_TIMEOUT_DEFAULT=60

exec /app/common-entrypoint.sh
