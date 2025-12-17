# syntax=docker/dockerfile:1.7

FROM nvcr.io/nvidia/tritonserver:24.12-py3 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENV=/opt/venv \
    UV_LINK_MODE=copy \
    UV_INSTALL_DIR=/usr/local/bin

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git git-lfs libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

COPY pyproject.toml ./pyproject.toml
COPY uv.lock ./uv.lock
COPY oim/common ./oim/common
COPY oim/nvidia/nemoretriever-ocr-v1 ./oim/nvidia/nemoretriever-ocr-v1

WORKDIR /workspace/oim/nvidia/nemoretriever-ocr-v1

RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev --frozen \
    && mv /workspace/.venv /opt/venv

FROM nvcr.io/nvidia/tritonserver:24.12-py3 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app/src:/models/scene_text_wrapper/1" \
    HOME="/tmp" \
    HF_HOME="/tmp/hf" \
    TRITON_CACHE_PATH="/tmp/triton"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git-lfs libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /workspace/oim/nvidia/nemoretriever-ocr-v1/src /app/src
COPY --from=builder /workspace/oim/nvidia/nemoretriever-ocr-v1/README.md README.md
COPY --from=builder /workspace/oim/nvidia/nemoretriever-ocr-v1/triton/model_repository /models
COPY --from=builder /workspace/oim/nvidia/nemoretriever-ocr-v1/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000 8001 8002 8003

USER 65532:65532

CMD ["/app/entrypoint.sh"]
