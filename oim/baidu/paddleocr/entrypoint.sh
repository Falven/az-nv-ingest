#!/usr/bin/env bash
set -euo pipefail

export OIM_APP_MODULE="oim_paddleocr.server:app"
export OIM_WAIT_FOR_READY=0
export OIM_EXPORT_TRITON_URLS=0
export OIM_HTTP_PORT_DEFAULT=8000
export TRITON_HTTP_PORT_DEFAULT=8500
export TRITON_GRPC_PORT_DEFAULT=8501
export TRITON_METRICS_PORT_DEFAULT=8502
export TRITON_MODEL_REPO_DEFAULT=/models
export TRITON_READY_TIMEOUT_DEFAULT=30

exec /app/common-entrypoint.sh
