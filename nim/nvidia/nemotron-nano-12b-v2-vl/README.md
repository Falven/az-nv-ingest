# Nemotron Nano 12B v2 VL (open)

Reverse-engineered notes for `nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl` (VLM captioning) using the open `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` release. The service now ships with a Triton Python backend under `triton/model_repository/nemotron_nano_12b_v2_vl`; the FastAPI shim delegates all inference to Triton via the official `tritonclient` HTTP client.

## What nv-ingest expects

- Endpoint: HTTP only; defaults to `VLM_CAPTION_ENDPOINT=http://vlm:8000/v1/chat/completions`. No gRPC path is exercised in the code.
- Model name header: `model` field set from `VLM_CAPTION_MODEL_NAME` (default `nvidia/nemotron-nano-12b-v2-vl`). A Bearer token is sent if `NGC_API_KEY`/`NVIDIA_API_KEY` is set.
- Prompt wiring: system prompt defaults to `/no_think`; user prompt defaults to `Caption the content of this image:`. Both are sent as OpenAI-style chat messages.
- Input shape (per request): one user message per image containing `{type:"text",text:"<prompt>"}` and `{type:"image_url",image_url:{url:"data:image/png;base64,<...>"}}`. Images are base64 PNG data URLs; the client pre-scales them to ~180k chars max.
- Output contract: JSON with `choices` array; each choice must include `message.content` (string caption). Only `choices[*].message.content` is read.
- Batching: with the HTTP path, the client falls back to batch size 1; expect one image per request.
- Triton surface: FastAPI proxies to Triton running locally on `NIM_TRITON_HTTP_PORT` (default `8003`) using the bundled model repository.

## Request/response example

```
POST /v1/chat/completions
Authorization: Bearer $NGC_API_KEY
Content-Type: application/json

{
  "model": "nvidia/nemotron-nano-12b-v2-vl",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "/no_think"}]},
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Caption the content of this image:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64_png>"}}
      ]
    }
  ],
  "max_tokens": 512,
  "temperature": 1.0,
  "top_p": 1.0,
  "stream": false
}
```

Response shape (anything with `choices[*].message.content` works):

```
{
  "id": "chatcmpl-xyz",
  "object": "chat.completion",
  "created": 0000000000,
  "model": "nvidia/nemotron-nano-12b-v2-vl",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "A caption for the image."},
      "finish_reason": "stop"
    }
  ]
}
```

## Local FastAPI server (mock friendly)

- Run the packaged FastAPI app with Triton locally; set `ENABLE_MOCK_INFERENCE=1` to skip model downloads for contract testing. You need `tritonserver` available on `PATH` (or set `TRITON_SERVER_BIN`):

```
ENABLE_MOCK_INFERENCE=1 NIM_REQUIRE_AUTH=0 ./entrypoint.sh
```

- Sample request against the local server:

```
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "model": "nvidia/nemotron-nano-12b-v2-vl",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Caption the content of this image:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}
      ]
    }
  ],
  "stream": false
}
EOF
```

- Replace the empty data URL with a real base64 PNG payload. The mock response returns deterministic captions while preserving the OpenAI-style contract.

## Notes and parity gaps

- The current client does not stream and expects one caption per request; multi-image batching would need a gRPC path that nv-ingest does not use.
- Images arrive as data URLs already resized to the ~180k base64 budget; keep PNG support on the server side.
- Triton mock mode is available via `ENABLE_MOCK_INFERENCE=1`; all other requests run through the Python backend shipped in the model repository.
