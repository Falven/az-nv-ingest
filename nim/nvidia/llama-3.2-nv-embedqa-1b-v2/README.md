# llama-3.2-nv-embedqa-1b-v2 (open replacement)

Reverse-engineered drop-in for the private `nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2` text embedding NIM. Lives under `nim/nvidia/llama-3.2-nv-embedqa-1b-v2` to keep upstream sources untouched.

## What nv-ingest expects

- HTTP endpoint on `${NIM_HTTP_API_PORT:-8000}` with `POST /v1/embeddings`. Payload: `{"model":"nvidia/llama-3.2-nv-embedqa-1b-v2","input":["..."],"encoding_format":"float","input_type":"passage","truncate":"END","dimensions":2048}`. `model` defaults to the server model; `input` may be a single string or list. `input_type` controls prefixing (`query:` vs `passage:`); `truncate` supports `END`, `START`, or `NONE`. `dimensions` is optional Matryoshka down-projection.
- Response shape: `{"object":"list","model":"...","data":[{"object":"embedding","index":0,"embedding":[...float...]}],"usage":{"prompt_tokens":N,"total_tokens":N}}`. `encoding_format="base64"` returns base64-encoded float32 bytes in `embedding`.
- Health: `GET /v1/health/live` and `/v1/health/ready` return 200. Metadata: `GET /v1/models` lists the model id; `GET /v1/metadata` exposes `modelInfo[0].shortName` for `get_model_name()` compatibility. Triton-compatible readiness endpoints live under `/v2/health/*`.
- Batch limit enforced by `NIM_TRITON_MAX_BATCH_SIZE` (default `30`, Helm defaults to `3` for ONNX profile). Requests exceeding the limit are rejected with `400`. The FastAPI surface delegates inference to Triton via `tritonclient`.

## Model source

- Uses the open weights from https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2.
- Applies the official prefixing (`query:` / `passage:`) and mean-pooling with L2 normalization. `dimensions` slices the leading components and renormalizes (Matryoshka embeddings).

## Running locally

Prereqs: `docker` (preferred) or Python 3.10+ with `git-lfs`. The service now ships with a Triton model repository under `triton/model_repository/llama_3_2_nv_embedqa_1b_v2`.

### Docker (recommended)

```bash
cd nim/nvidia/llama-3.2-nv-embedqa-1b-v2
docker build -t llama-embedqa-triton .
docker run --gpus all -p 8000:8000 -p 8003:8003 llama-embedqa-triton
```

### Manual start (API + Triton sidecar)

```bash
cd nim/nvidia/llama-3.2-nv-embedqa-1b-v2
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# start tritonserver in a separate shell
tritonserver --model-repository=./triton/model_repository --http-port=8003 --grpc-port=8004 &
TRITON_HTTP_ENDPOINT=http://127.0.0.1:8003 uvicorn llama_3_2_nv_embedqa_1b_v2.server:app --host 0.0.0.0 --port ${NIM_HTTP_API_PORT:-8000}
```

For development without downloading the model weights, enable deterministic mock embeddings for the Triton Python backend:

```bash
export ENABLE_MOCK_INFERENCE=1
tritonserver --model-repository=./triton/model_repository --http-port=8003 --grpc-port=8004 &
TRITON_HTTP_ENDPOINT=http://127.0.0.1:8003 uvicorn llama_3_2_nv_embedqa_1b_v2.server:app --host 0.0.0.0 --port ${NIM_HTTP_API_PORT:-8000}
```

Smoke test:

```bash
curl -X POST http://127.0.0.1:${NIM_HTTP_API_PORT:-8000}/v1/embeddings \
  -H "content-type: application/json" \
  -d '{"model":"nvidia/llama-3.2-nv-embedqa-1b-v2","input":["example passage text"],"input_type":"passage","truncate":"END"}'
```

## Compose/K8s alignment

- Exposes port `8000` (HTTP API), Triton HTTP on `8003`, and optional metrics on `8002` when auth is disabled. Compatible with the `docker-compose.yaml` `embedding` service and Helm values (`embedqa` section).
- Mirrors key envs: `NIM_HTTP_API_PORT`, `TRITON_HTTP_ENDPOINT`, `NIM_TRITON_MAX_BATCH_SIZE`, `MODEL_VERSION` (for `modelInfo.shortName`), `LOG_LEVEL`.

## Known deltas vs. closed NIM

- A lightweight mock mode is available via `ENABLE_MOCK_INFERENCE=1` for environments without GPU weights.
- Only the embeddings surface (`/v1/embeddings`) plus health/metadata is implemented; no async job APIs.
