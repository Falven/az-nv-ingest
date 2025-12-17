# nemoretriever-ocr-v1 (open replacement)

Reverse-engineered drop-in for the private `nvcr.io/nim/nvidia/nemoretriever-ocr-v1` NIM using the open-source `nvidia/nemotron-ocr-v1` model.

## What nv-ingest expects

- Endpoints (see `docker-compose.yaml`, `helm/values.yaml`, and `default_pipeline_impl.py`):
  - HTTP: `OCR_HTTP_ENDPOINT` → defaults to `http://ocr:8000/v1/infer`
  - gRPC: `OCR_GRPC_ENDPOINT` → defaults to `ocr:8001` (can be left empty to force HTTP)
  - Protocol toggle: `OCR_INFER_PROTOCOL` → defaults to `grpc`; set to `http` for this server.
- Request payload (HTTP): `{"input":[{"type":"image_url","url":"data:image/png;base64,<...>"}], "merge_levels":["paragraph", ...]}` where `merge_levels` is optional (defaults to `paragraph`).
- Response payload (HTTP): `{"data":[{"text_detections":[{"bounding_box":{"points":[{"x":..,"y":..},...]},"text_prediction":{"text":"...", "confidence": <float>}}, ...]}]}`.
  - Bounding boxes are normalized coordinates (0–1) with four points per detection.
- Health: `/v1/health/live` and `/v1/health/ready` must return HTTP 200 for readiness checks.
- Metadata: `/v1/metadata` should expose `modelInfo[0].shortName` so discovery works if queried (we return `nemoretriever-ocr-v1`).

## Open implementation in this directory

- `triton/model_repository/scene_text_wrapper`: Triton Python backend that loads `NemotronOCR` (or a deterministic mock) and returns OCR detections as JSON tensors.
- `server.py`: FastAPI shim that validates HTTP requests and forwards them to Triton via the official `tritonclient` gRPC client, returning the legacy `/v1/infer` shape.
- `entrypoint.sh`: Starts Triton (`--grpc-port NIM_TRITON_GRPC_PORT`, `--http-port 8003`) and uvicorn on `NIM_HTTP_API_PORT` (defaults: 8001 gRPC, 8003 Triton HTTP, 8000 FastAPI, 8002 Triton metrics when auth is disabled).
- Honors env vars: `NIM_HTTP_API_PORT`, `NIM_TRITON_GRPC_PORT`, `NIM_TRITON_MAX_BATCH_SIZE`, `MERGE_LEVEL`, `MODEL_DIR` (HF checkpoints), `ENABLE_MOCK_INFERENCE`, `TRITON_MODEL_REPOSITORY` (defaults to `/models`), and bearer tokens via `NGC_API_KEY`/`NVIDIA_API_KEY`.
- Dependencies are declared in `pyproject.toml`; install the HF package separately (`nemotron-ocr` from the repo).

## Setup

1. Fetch the model repo + weights (requires git-lfs, CUDA-capable system):

```
git lfs install
git clone https://huggingface.co/nvidia/nemotron-ocr-v1
cd nemotron-ocr-v1/nemotron-ocr
pip install -v .
```

2. Install the server deps (uv or pip):

```
cd /Users/fran/Source/open-nv-ingest/nim/nvidia/nemoretriever-ocr-v1
uv sync
```

3. Run Triton + the HTTP shim locally (assumes checkpoints at `/opt/models`):

```
NIM_HTTP_API_PORT=8000 \
NIM_TRITON_GRPC_PORT=8001 \
TRITON_HTTP_PORT=8003 \
TRITON_MODEL_REPOSITORY=/Users/fran/Source/open-nv-ingest/nim/nvidia/nemoretriever-ocr-v1/triton/model_repository \
MODEL_DIR=/opt/models \
ENABLE_MOCK_INFERENCE=false \
uv run ./entrypoint.sh
```

4. Wire nv-ingest to it:

- Set `OCR_INFER_PROTOCOL=grpc` (preferred) and point `OCR_GRPC_ENDPOINT` to the Triton gRPC port (default `ocr:8001`).
- For HTTP fallback, set `OCR_INFER_PROTOCOL=http` and point `OCR_HTTP_ENDPOINT=http://<host>:8000/v1/infer`.
- If using Helm, override `nimOperator.nemoretriever_ocr_v1.expose.service.port` to 8000 and set envs above.

## Request/response example

```
curl -X POST http://localhost:8000/v1/infer \
  -H "content-type: application/json" \
  -d '{"input":[{"type":"image_url","url":"data:image/png;base64,<base64_png>"}],"merge_levels":["word"]}'
```

Response:

```
{
  "data": [
    {
      "text_detections": [
        {
          "bounding_box": {
            "points": [
              {"x":0.12,"y":0.18},
              {"x":0.42,"y":0.18},
              {"x":0.42,"y":0.24},
              {"x":0.12,"y":0.24}
            ],
            "type": "quadrilateral"
          },
          "text_prediction": {"text":"Example", "confidence":0.97}
        }
      ]
    }
  ]
}
```

## Known gaps vs. closed NIM

- The HF pipeline runs per-image; batches are processed sequentially up to `NIM_TRITON_MAX_BATCH_SIZE`.

## Mock / dry-run mode

Set `ENABLE_MOCK_INFERENCE=true` to skip loading the Nemotron OCR weights and return deterministic sample detections. This is useful for local contract testing when GPU resources or model assets are unavailable.
