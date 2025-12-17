#!/usr/bin/env bash
set -euo pipefail

TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-${NIM_TRITON_HTTP_PORT:-8100}}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-${NIM_GRPC_API_PORT:-8001}}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-${NIM_TRITON_METRICS_PORT:-8003}}"
TRITON_MODEL_REPO="${TRITON_MODEL_REPO:-${TRITON_MODEL_REPOSITORY:-/app/triton/model_repository}}"
TRITON_SERVER_BIN="${TRITON_SERVER_BIN:-/opt/tritonserver/bin/tritonserver}"
LOG_VERBOSE="${NIM_TRITON_LOG_VERBOSE:-0}"
HTTP_PORT="${NIM_HTTP_API_PORT:-8000}"

"${TRITON_SERVER_BIN}" \
  --model-repository="${TRITON_MODEL_REPO}" \
  --http-port="${TRITON_HTTP_PORT}" \
  --grpc-port="${TRITON_GRPC_PORT}" \
  --metrics-port="${TRITON_METRICS_PORT}" \
  --model-control-mode=none \
  --log-verbose="${LOG_VERBOSE}" &
TRITON_PID=$!

cleanup() {
  if kill -0 "${TRITON_PID}" >/dev/null 2>&1; then
    kill "${TRITON_PID}"
  fi
}
trap cleanup EXIT TERM INT

READY_URL="http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready"
ready=0
for _ in $(seq 1 120); do
  if curl -sf "${READY_URL}" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${TRITON_PID}" >/dev/null 2>&1; then
    echo "Triton server exited before readiness" >&2
    exit 1
  fi
  sleep 1
done

if [ "${ready}" -ne 1 ]; then
  echo "Triton failed to become ready" >&2
  exit 1
fi

exec uvicorn oim_nemoretriever_table_structure_v1.server:app --host 0.0.0.0 --port "${HTTP_PORT}"
