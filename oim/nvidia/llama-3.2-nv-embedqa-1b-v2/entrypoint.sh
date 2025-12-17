#!/usr/bin/env bash
set -euo pipefail

TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-8003}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-8004}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-8005}"
TRITON_MODEL_REPO="${TRITON_MODEL_REPO:-/app/triton/model_repository}"

tritonserver \
	--model-repository="${TRITON_MODEL_REPO}" \
	--http-port="${TRITON_HTTP_PORT}" \
	--grpc-port="${TRITON_GRPC_PORT}" \
	--metrics-port="${TRITON_METRICS_PORT}" \
	--model-control-mode=none &
TRITON_PID=$!

cleanup() {
	if kill -0 "${TRITON_PID}" >/dev/null 2>&1; then
		kill "${TRITON_PID}"
	fi
}
trap cleanup EXIT TERM INT

READY_URL="http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready"
ready=0
for _ in $(seq 1 60); do
	if curl -sf "${READY_URL}" >/dev/null 2>&1; then
		ready=1
		break
	fi
	sleep 1
done
if [ "${ready}" -ne 1 ]; then
	echo "Triton failed to become ready" >&2
	exit 1
fi

exec uvicorn llama_3_2_nv_embedqa_1b_v2.server:app --host 0.0.0.0 --port "${NIM_HTTP_API_PORT:-8000}"
