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
COPY oim/common ./oim/common
COPY oim/baidu/paddleocr ./oim/baidu/paddleocr

RUN --mount=type=cache,target=/root/.cache/uv uv sync --package oim-paddleocr --no-dev \
    && mv /workspace/.venv /opt/venv

FROM nvcr.io/nvidia/tritonserver:24.12-py3 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app/src:/models/paddleocr/1" \
    HOME="/tmp" \
    PADDLE_HOME="/tmp/paddle" \
    HF_HOME="/tmp/hf" \
    PPDLX_CACHE_DIR="/tmp/paddlex"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git-lfs libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /workspace/oim/baidu/paddleocr/src /app/src
COPY --from=builder /workspace/oim/baidu/paddleocr/README.md README.md
COPY --from=builder /workspace/oim/baidu/paddleocr/triton/model_repository /models
COPY --from=builder /workspace/oim/baidu/paddleocr/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000 8500 8501 8502

CMD ["/app/entrypoint.sh"]
