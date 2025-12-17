# llama-3.2-nv-rerankqa-1b-v2 (open replacement)

Reverse-engineered drop-in for the private `nvcr.io/nim/nvidia/llama-3.2-nv-rerankqa-1b-v2` reranker. Lives under `nim/nvidia/llama-3.2-nv-rerankqa-1b-v2` to keep upstream sources untouched.

## What nv-ingest expects

- HTTP endpoint on `${NIM_HTTP_API_PORT:-8000}` with `POST /v1/ranking` (plus compatibility alias `/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking`). Payload: `{"model":"nvidia/llama-3.2-nv-rerankqa-1b-v2","query":{"text":"<query>"},"passages":[{"text":"..."},...],"truncate":"END"}`. `model` defaults to the server model; `truncate` supports `END`, `START`, or `NONE`.
- Response shape: `{"rankings":[{"index":0,"logit":6.8},...]}` sorted by descending `logit`.
- Health: `GET /v1/health/live` and `/v1/health/ready` return 200. Metadata: `GET /v1/models` lists the model id; `GET /v1/metadata` exposes `modelInfo[0].shortName` and `maxBatchSize` when available. Triton-compatible readiness endpoints live under `/v2/health/*`.
- Batch limit enforced by `max_batch_size` in the Triton model config (default `64`, further clamped by `NIM_TRITON_MAX_BATCH_SIZE` when set). Requests exceeding the limit are rejected with `400`. The FastAPI surface delegates inference to Triton via `tritonclient`.
- Auth: Bearer token required for app endpoints when `NIM_REQUIRE_AUTH=true` (tokens from `NGC_API_KEY`/`NVIDIA_API_KEY`/`NIM_NGC_API_KEY`). Health endpoints stay open.
- Metrics: Prometheus on `${NIM_TRITON_METRICS_PORT:-8002}` (FastAPI metrics when auth is disabled). OTEL tracing is optional (`NIM_ENABLE_OTEL` + `NIM_OTEL_*`). Triton metrics are exposed on `${TRITON_METRICS_PORT:-8005}`.

## Model source

- Uses the open weights from https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2.
- Applies the official prompt template (`question:{query} \n \n passage:{passage}`) and returns raw logits per passage.

## Running locally

Prereqs: `docker` (preferred) or Python 3.10+ with `git-lfs`. The service ships with a Triton model repository under `triton/model_repository/llama_3_2_nv_rerankqa_1b_v2`.

### Docker (recommended)

```bash
cd nim/nvidia/llama-3.2-nv-rerankqa-1b-v2
docker build -t llama-rerank-triton .
docker run --gpus all -p 8000:8000 -p 8003:8003 llama-rerank-triton
```

### Manual start (API + Triton sidecar)

```bash
cd nim/nvidia/llama-3.2-nv-rerankqa-1b-v2
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# start tritonserver in a separate shell
tritonserver --model-repository=./triton/model_repository --http-port=8003 --grpc-port=8004 --metrics-port=8005 &
TRITON_HTTP_ENDPOINT=http://127.0.0.1:8003 uvicorn llama_3_2_nv_rerankqa_1b_v2.server:app --host 0.0.0.0 --port ${NIM_HTTP_API_PORT:-8000}
```

For development without downloading model weights, enable deterministic mock scoring for the Triton Python backend:

```bash
export ENABLE_MOCK_INFERENCE=1
tritonserver --model-repository=./triton/model_repository --http-port=8003 --grpc-port=8004 --metrics-port=8005 &
TRITON_HTTP_ENDPOINT=http://127.0.0.1:8003 uvicorn llama_3_2_nv_rerankqa_1b_v2.server:app --host 0.0.0.0 --port ${NIM_HTTP_API_PORT:-8000}
```

Smoke test:

```bash
curl -X POST http://127.0.0.1:${NIM_HTTP_API_PORT:-8000}/v1/ranking \
  -H "content-type: application/json" \
  -d '{"model":"nvidia/llama-3.2-nv-rerankqa-1b-v2","query":{"text":"which way should i go?"},"passages":[{"text":"two roads diverged in a yellow wood..."},{"text":"then took the other, as just as fair..."}],"truncate":"END"}'
```

## Compose/K8s alignment

- Exposes port `8000` (HTTP API), Triton HTTP on `8003`, Triton gRPC on `8004`, and optional metrics on `8002` (FastAPI) and `8005` (Triton). Compatible with the `docker-compose.yaml` `reranker` service and Helm values (`llama_3_2_nv_rerankqa_1b_v2`).
- Mirrors key envs: `NIM_HTTP_API_PORT`, `TRITON_HTTP_ENDPOINT`, `NIM_TRITON_MAX_BATCH_SIZE`, `NIM_TRITON_RATE_LIMIT`, `NIM_TRITON_LOG_VERBOSE`, `MODEL_VERSION`, `LOG_LEVEL`, optional `NIM_ENABLE_OTEL` and `NIM_OTEL_*`.

## Known deltas vs. closed NIM

- Lightweight mock mode is available via `ENABLE_MOCK_INFERENCE=1` for environments without GPU weights.
- Only the reranking surface (`/v1/ranking` and the compatibility path) plus health/metadata is implemented; no async job APIs.
