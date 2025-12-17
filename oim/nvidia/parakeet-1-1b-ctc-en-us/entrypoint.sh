#!/usr/bin/env bash
set -euo pipefail

export OIM_APP_MODULE="oim_parakeet_1_1b_ctc_en_us.server:app"
export OIM_HTTP_PORT_DEFAULT=9000
export OIM_EXPORT_TRITON_URLS=0
export TRITON_HTTP_PORT_DEFAULT=8003
export TRITON_GRPC_PORT_DEFAULT=8004
export TRITON_METRICS_PORT_DEFAULT=8005
export TRITON_MODEL_REPO_DEFAULT="/app/triton/model_repository"
export TRITON_READY_TIMEOUT_DEFAULT=60

exec /app/common-entrypoint.sh
