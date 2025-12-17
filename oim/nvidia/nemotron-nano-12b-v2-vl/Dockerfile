# syntax=docker/dockerfile:1.7

ARG TRITON_VERSION=24.12-py3
FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENV=/opt/venv \
    UV_LINK_MODE=copy \
    UV_INSTALL_DIR=/usr/local/bin

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git git-lfs python3-venv libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

COPY pyproject.toml ./pyproject.toml
COPY oim/common ./oim/common
COPY oim/nvidia/nemotron-nano-12b-v2-vl ./oim/nvidia/nemotron-nano-12b-v2-vl

WORKDIR /workspace/oim/nvidia/nemotron-nano-12b-v2-vl

RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev \
    && mv /workspace/.venv /opt/venv

FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app/src"

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git-lfs libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /workspace/oim/nvidia/nemotron-nano-12b-v2-vl/src /app/src
COPY --from=builder /workspace/oim/nvidia/nemotron-nano-12b-v2-vl/triton /app/triton
COPY --from=builder /workspace/oim/nvidia/nemotron-nano-12b-v2-vl/README.md README.md
COPY --from=builder /workspace/oim/common/entrypoint.sh /app/common-entrypoint.sh
COPY --from=builder /workspace/oim/nvidia/nemotron-nano-12b-v2-vl/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh /app/common-entrypoint.sh \
    && groupadd -g 65532 appuser \
    && useradd -u 65532 -g 65532 -m appuser

EXPOSE 8000 8002 8003 8004 8005

USER appuser

CMD ["/app/entrypoint.sh"]
