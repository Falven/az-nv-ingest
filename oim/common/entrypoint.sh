#!/usr/bin/env bash
set -euo pipefail

# Shared entrypoint for OIM services. Configurable via env so per-model wrappers
# can set defaults without duplicating logic.

require_var() {
	local name="$1"
	local value="$2"
	if [[ -z "${value}" ]]; then
		echo "Missing required env: ${name}" >&2
		exit 1
	fi
}

prepend_path() {
	local value="$1"
	local target="$2"
	if [[ -z "${value}" ]]; then
		return
	fi
	if [[ -n "${!target:-}" ]]; then
		export "${target}"="${value}:${!target}"
		return
	fi
	export "${target}"="${value}"
}

OIM_APP_MODULE="${OIM_APP_MODULE:-}"
require_var "OIM_APP_MODULE" "${OIM_APP_MODULE}"

OIM_HTTP_PORT_DEFAULT="${OIM_HTTP_PORT_DEFAULT:-8000}"
OIM_HTTP_PORT="${OIM_HTTP_PORT:-${NIM_HTTP_API_PORT:-${HTTP_PORT:-${OIM_HTTP_PORT_DEFAULT}}}}"

TRITON_HTTP_PORT_DEFAULT="${TRITON_HTTP_PORT_DEFAULT:-8003}"
TRITON_GRPC_PORT_DEFAULT="${TRITON_GRPC_PORT_DEFAULT:-8004}"
TRITON_METRICS_PORT_DEFAULT="${TRITON_METRICS_PORT_DEFAULT:-8005}"

TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-${NIM_TRITON_HTTP_PORT:-${TRITON_HTTP_PORT_DEFAULT}}}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-${NIM_TRITON_GRPC_PORT:-${NIM_GRPC_API_PORT:-${TRITON_GRPC_PORT_DEFAULT}}}}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-${NIM_TRITON_METRICS_PORT:-${TRITON_METRICS_PORT_DEFAULT}}}"

TRITON_MODEL_REPO_DEFAULT="${TRITON_MODEL_REPO_DEFAULT:-/app/triton/model_repository}"
TRITON_MODEL_REPO="${TRITON_MODEL_REPO:-${TRITON_MODEL_REPOSITORY:-${TRITON_MODEL_REPO_DEFAULT}}}"

TRITON_SERVER_BIN="${TRITON_SERVER_BIN:-tritonserver}"
TRITON_MODEL_CONTROL_MODE="${TRITON_MODEL_CONTROL_MODE:-none}"
TRITON_LOG_VERBOSE="${TRITON_LOG_VERBOSE:-${NIM_TRITON_LOG_VERBOSE:-${TRITON_LOG_VERBOSE_DEFAULT:-}}}"
TRITON_EXTRA_ARGS="${TRITON_EXTRA_ARGS:-}"

TRITON_READY_TIMEOUT="${TRITON_READY_TIMEOUT:-${TRITON_READY_TIMEOUT_DEFAULT:-120}}"
TRITON_READY_INTERVAL="${TRITON_READY_INTERVAL:-1}"
TRITON_READY_URL="${TRITON_READY_URL:-http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready}"
OIM_WAIT_FOR_READY="${OIM_WAIT_FOR_READY:-1}"
OIM_EXPORT_TRITON_URLS="${OIM_EXPORT_TRITON_URLS:-1}"

OIM_ADD_VENV_SITE_PACKAGES="${OIM_ADD_VENV_SITE_PACKAGES:-0}"
OIM_EXTRA_PYTHONPATH="${OIM_EXTRA_PYTHONPATH:-}"

UVICORN_BIN="${UVICORN_BIN:-uvicorn}"
UVICORN_HOST="${UVICORN_HOST:-0.0.0.0}"
UVICORN_ARGS="${UVICORN_ARGS:-}"

if [[ "${OIM_ADD_VENV_SITE_PACKAGES}" == "1" ]]; then
	python_version="$(python3 - <<'PY'
import sys
print(f"python{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
	venv_site="/opt/venv/lib/${python_version}/site-packages"
	if [[ -d "${venv_site}" ]]; then
		prepend_path "${venv_site}" PYTHONPATH
	fi
fi

if [[ -n "${OIM_EXTRA_PYTHONPATH}" ]]; then
	prepend_path "${OIM_EXTRA_PYTHONPATH}" PYTHONPATH
fi

triton_args=(
	"--model-repository=${TRITON_MODEL_REPO}"
	"--http-port=${TRITON_HTTP_PORT}"
	"--grpc-port=${TRITON_GRPC_PORT}"
	"--metrics-port=${TRITON_METRICS_PORT}"
	"--model-control-mode=${TRITON_MODEL_CONTROL_MODE}"
)

if [[ -n "${TRITON_LOG_VERBOSE}" ]]; then
	triton_args+=("--log-verbose=${TRITON_LOG_VERBOSE}")
fi

if [[ -n "${TRITON_EXTRA_ARGS}" ]]; then
	read -r -a extra_args <<< "${TRITON_EXTRA_ARGS}"
	triton_args+=("${extra_args[@]}")
fi

"${TRITON_SERVER_BIN}" "${triton_args[@]}" &
TRITON_PID=$!

cleanup() {
	local pids=()
	if [[ -n "${TRITON_PID:-}" ]]; then
		pids+=("${TRITON_PID}")
	fi
	if [[ -n "${UVICORN_PID:-}" ]]; then
		pids+=("${UVICORN_PID}")
	fi
	if [[ ${#pids[@]} -gt 0 ]]; then
		kill "${pids[@]}" 2>/dev/null || true
	fi
}

trap cleanup EXIT TERM INT

if [[ "${OIM_WAIT_FOR_READY}" == "1" ]]; then
	ready=0
	for ((i = 1; i <= TRITON_READY_TIMEOUT; i++)); do
		if curl -sf "${TRITON_READY_URL}" >/dev/null 2>&1; then
			ready=1
			break
		fi
		if ! kill -0 "${TRITON_PID}" >/dev/null 2>&1; then
			echo "Triton server exited before readiness" >&2
			exit 1
		fi
		sleep "${TRITON_READY_INTERVAL}"
	done
	if [[ "${ready}" -ne 1 ]]; then
		echo "Triton failed to become ready after ${TRITON_READY_TIMEOUT}s" >&2
		exit 1
	fi
fi

if [[ "${OIM_EXPORT_TRITON_URLS}" == "1" ]]; then
	export TRITON_GRPC_URL="${TRITON_GRPC_URL:-127.0.0.1:${TRITON_GRPC_PORT}}"
	export TRITON_HTTP_URL="${TRITON_HTTP_URL:-http://127.0.0.1:${TRITON_HTTP_PORT}}"
fi

uvicorn_cmd=("${UVICORN_BIN}" "${OIM_APP_MODULE}" --host "${UVICORN_HOST}" --port "${OIM_HTTP_PORT}")
if [[ -n "${UVICORN_ARGS}" ]]; then
	read -r -a uvicorn_extra <<< "${UVICORN_ARGS}"
	uvicorn_cmd+=("${uvicorn_extra[@]}")
fi

"${uvicorn_cmd[@]}" &
UVICORN_PID=$!

set +e
wait -n "${TRITON_PID}" "${UVICORN_PID}"
status=$?
cleanup
wait "${TRITON_PID}" 2>/dev/null || true
wait "${UVICORN_PID}" 2>/dev/null || true
exit "${status}"
