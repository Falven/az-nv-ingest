#!/usr/bin/env bash
set -euo pipefail

TRITON_HTTP_PORT=${TRITON_HTTP_PORT:-8003}
TRITON_GRPC_PORT=${TRITON_GRPC_PORT:-${NIM_TRITON_GRPC_PORT:-8001}}
TRITON_METRICS_PORT=${TRITON_METRICS_PORT:-${NIM_TRITON_METRICS_PORT:-8002}}
TRITON_MODEL_REPOSITORY=${TRITON_MODEL_REPOSITORY:-/models}
TRITON_EXTRA_ARGS=${TRITON_EXTRA_ARGS:-}

tritonserver \
  --model-repository="${TRITON_MODEL_REPOSITORY}" \
  --http-port="${TRITON_HTTP_PORT}" \
  --grpc-port="${TRITON_GRPC_PORT}" \
  --metrics-port="${TRITON_METRICS_PORT}" \
  ${TRITON_EXTRA_ARGS} &

TRITON_PID=$!

cleanup() {
  local pids=("${TRITON_PID}")
  if [[ -n "${UVICORN_PID:-}" ]]; then
    pids+=("${UVICORN_PID}")
  fi
  kill "${pids[@]}" 2>/dev/null || true
}

trap cleanup TERM INT

export TRITON_GRPC_URL=${TRITON_GRPC_URL:-"localhost:${TRITON_GRPC_PORT}"}
export TRITON_HTTP_URL=${TRITON_HTTP_URL:-"http://127.0.0.1:${TRITON_HTTP_PORT}"}

uvicorn nemoretriever_ocr_v1.server:app --host 0.0.0.0 --port "${NIM_HTTP_API_PORT:-8000}" &
UVICORN_PID=$!

set +e
wait -n "${TRITON_PID}" "${UVICORN_PID}"
STATUS=$?
cleanup
wait "${TRITON_PID}" 2>/dev/null || true
wait "${UVICORN_PID}" 2>/dev/null || true
exit "${STATUS}"
