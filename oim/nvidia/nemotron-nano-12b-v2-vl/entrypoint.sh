#!/usr/bin/env bash
set -euo pipefail

export OIM_APP_MODULE="oim_nemotron_nano_12b_v2_vl.server:app"
export OIM_ADD_VENV_SITE_PACKAGES=1
export OIM_EXTRA_PYTHONPATH="/app/src"
export TRITON_SERVER_BIN="${TRITON_SERVER_BIN:-/opt/tritonserver/bin/tritonserver}"
export TRITON_HTTP_PORT_DEFAULT=8003
export TRITON_GRPC_PORT_DEFAULT=8004
export TRITON_METRICS_PORT_DEFAULT=8005
export TRITON_MODEL_REPO_DEFAULT="${TRITON_MODEL_REPO:-${TRITON_MODEL_REPOSITORY:-/app/triton/model_repository}}"
export TRITON_LOG_VERBOSE_DEFAULT=0
export TRITON_READY_TIMEOUT_DEFAULT=120

exec /app/common-entrypoint.sh
