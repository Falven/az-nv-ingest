# syntax=docker/dockerfile:1.7

FROM nvcr.io/nvidia/tritonserver:24.12-py3 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENV=/opt/venv \
    UV_LINK_MODE=copy \
    UV_INSTALL_DIR=/usr/local/bin

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git git-lfs libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

COPY pyproject.toml ./pyproject.toml
COPY oim/common ./oim/common
COPY oim/nvidia/nemoretriever-page-elements-v3 ./oim/nvidia/nemoretriever-page-elements-v3

WORKDIR /workspace/oim/nvidia/nemoretriever-page-elements-v3

RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev \
    && mv /workspace/.venv /opt/venv

FROM nvcr.io/nvidia/tritonserver:24.12-py3 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app/src"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /workspace/oim/nvidia/nemoretriever-page-elements-v3/src /app/src
COPY --from=builder /workspace/oim/nvidia/nemoretriever-page-elements-v3/triton /app/triton
COPY --from=builder /workspace/oim/nvidia/nemoretriever-page-elements-v3/README.md README.md
COPY --from=builder /workspace/oim/common/entrypoint.sh /app/common-entrypoint.sh
COPY --from=builder /workspace/oim/nvidia/nemoretriever-page-elements-v3/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh /app/common-entrypoint.sh

EXPOSE 8000 8001 8002 8003 8100

USER 65532:65532

ENTRYPOINT ["bash", "/app/entrypoint.sh"]
