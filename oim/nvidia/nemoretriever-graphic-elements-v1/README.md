Nemotron Graphic Elements v1 (open)
====================================

Reverse-engineered replacement for `nvcr.io/nim/nvidia/nemoretriever-graphic-elements-v1` using the open-source `nvidia/nemotron-graphic-elements-v1` model.

What the upstream NIM exposes
-----------------------------
- Endpoints used by nv-ingest (see `docker-compose.yaml`, `helm/values.yaml`, and `src/nv_ingest/pipeline/default_pipeline_impl.py`):
  - gRPC: `YOLOX_GRAPHIC_ELEMENTS_GRPC_ENDPOINT` (defaults to `graphic-elements:8001`)
  - HTTP: `YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT` (defaults to `http://graphic-elements:8000/v1/infer`)
  - Protocol toggle: `YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL` (defaults to `grpc`; libmode defaults to `http`).
- Model contract (from `chart_extractor.py` and `yolox.py`):
  - Model name: `yolox_ensemble`
  - gRPC inputs: `INPUT_IMAGES` (BYTES, shape `[batch]`, base64 PNG) and `THRESHOLDS` (FP32, shape `[batch,2]`, values `[0.01,0.25]`).
  - gRPC output: one tensor of bytes/JSON per image containing a dict keyed by class label with normalized `[x_min,y_min,x_max,y_max,score]`.
  - HTTP request payload: `{"input": [{"type": "image_url", "url": "data:image/png;base64,<...>"} , ...]}`.
  - HTTP response payload: `{"data": [{"bounding_boxes": {"chart_title": [{"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ..., "confidence": ...}], ...}}]}`.
  - Class labels: `["chart_title","x_title","y_title","xlabel","ylabel","other","legend_label","legend_title","mark_label","value_label"]`.
  - Max batch size is probed from Triton; nv-ingest requests up to 8 for chart extraction. Postprocessing expects **normalized** coords; nv-ingest scales them to pixel space later.

Open implementation in this directory
-------------------------------------
- Triton model repository at `triton/model_repository/yolox_ensemble` (Python backend). It decodes base64 PNG bytes, applies `nemotron_graphic_elements_v1` with the official postprocessing, and returns BYTES tensors containing JSON per image. Threshold defaults follow upstream (`conf_thresh=0.01`, `iou_thresh=0.25`, score cutoff `THRESHOLD`, default `0.1`).
- `server.py`: FastAPI shim that starts `tritonserver` with the bundled model repo and delegates inference to Triton via the official `tritonclient` gRPC client. It exposes `/v1/infer`, `/v1/health/*`, `/v1/metadata`, `/metrics` (and `/v2/metrics`), plus `/v2/models/*` metadata/config shims for compatibility.
- Auth: Bearer/NGC auth via `Authorization: Bearer <token>` or `ngc-api-key: <token>` on all endpoints except health; metrics are guarded when auth is enabled.
- Metrics: Prometheus on `/metrics`; optional standalone listener when `NIM_METRICS_PORT` is set and auth is disabled.
- Ports: HTTP shim on `${NIM_HTTP_API_PORT:-8000}`, Triton gRPC on `${NIM_GRPC_API_PORT:-8001}`, Triton HTTP on `${NIM_TRITON_HTTP_PORT:-8100}`, and Triton metrics on `${NIM_TRITON_METRICS_PORT:-8003}`.

Setup
-----
1) Ensure Git LFS is available (needed for the model dependency):
```
git lfs install
```

2) Install deps (uv recommended; pulls the HF model via pip):
```
cd /Users/fran/Source/open-nv-ingest
uv sync --no-dev --directory nim/nvidia/nemoretriever-graphic-elements-v1
```

3) Run the service (starts Triton + HTTP shim; set `DEVICE` for GPU selection):
```
NIM_HTTP_API_PORT=8000 \
NIM_GRPC_API_PORT=8001 \
NIM_TRITON_HTTP_PORT=8100 \
NIM_TRITON_METRICS_PORT=8003 \
NIM_TRITON_MAX_BATCH_SIZE=8 \
NGC_API_KEY=your_token_here \
uv run --directory nim/nvidia/nemoretriever-graphic-elements-v1 \
  uvicorn nemoretriever_graphic_elements_v1.server:app --host 0.0.0.0 --port ${NIM_HTTP_API_PORT:-8000}
```
- For API-only smoke testing without weights, set `ENABLE_MOCK_INFERENCE=1`; Triton still starts but serves mock predictions.

4) Wire nv-ingest:
- gRPC (parity): set `YOLOX_GRAPHIC_ELEMENTS_GRPC_ENDPOINT=<host>:8001`, `YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT=http://<host>:8000/v1/infer`, `YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL=grpc`, and provide `NGC_API_KEY` (or `NIM_NGC_API_KEY`/`NVIDIA_API_KEY`).
- HTTP-only (legacy): set `YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT=http://<host>:8000/v1/infer`, leave gRPC endpoint empty, and set `YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL=http`.

Container build (optional)
--------------------------
`docker-compose.yaml` now builds `nemoretriever-graphic-elements-v1-open` from this directory by default. To build/push manually:
```
docker build -t nemoretriever-graphic-elements-v1-open:latest nim/nvidia/nemoretriever-graphic-elements-v1
# optional: docker tag nemoretriever-graphic-elements-v1-open:latest <your-registry>/nemoretriever-graphic-elements-v1-open:latest
# optional: docker push <your-registry>/nemoretriever-graphic-elements-v1-open:latest
```
- The image bundles the Triton model repo and launches `tritonserver` (gRPC on 8001, HTTP on 8100, metrics on 8003) alongside the FastAPI shim on 8000.

Request/response example
------------------------
```
curl -X POST http://localhost:8000/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"input":[{"type":"image_url","url":"data:image/png;base64,<base64_png>"}]}'
```
Response:
```
{
  "data": [
    {
      "bounding_boxes": {
        "chart_title": [{"x_min":0.02,"y_min":0.05,"x_max":0.94,"y_max":0.12,"confidence":0.91}],
        "x_title": [],
        "y_title": [],
        "xlabel": [...],
        "ylabel": [...],
        "other": [...]
      }
    }
  ]
}
```

Notes
-----
- gRPC clients should point directly at the Triton gRPC port (`${NIM_GRPC_API_PORT:-8001}`) with model name `yolox_ensemble`; the HTTP shim remains for `/v1/infer` compatibility.
- Max batch size is enforced at the HTTP shim (`NIM_TRITON_MAX_BATCH_SIZE`, default 8) and in the Triton model config.
