#!/usr/bin/env bash
set -euo pipefail

TRITON_HTTP_PORT=${TRITON_HTTP_PORT:-8500}
TRITON_GRPC_PORT=${TRITON_GRPC_PORT:-8501}
TRITON_METRICS_PORT=${TRITON_METRICS_PORT:-8502}

tritonserver --model-repository=/models \
	--http-port="${TRITON_HTTP_PORT}" \
	--grpc-port="${TRITON_GRPC_PORT}" \
	--metrics-port="${TRITON_METRICS_PORT}" &

TRITON_PID=$!

trap "kill ${TRITON_PID}" TERM INT

uvicorn oim_paddleocr.server:app --host 0.0.0.0 --port 8000
