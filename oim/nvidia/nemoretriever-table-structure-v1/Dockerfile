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

RUN git lfs install \
    && git clone https://huggingface.co/nvidia/nemotron-table-structure-v1 /opt/nemotron-table-structure-v1

COPY pyproject.toml ./pyproject.toml
COPY nim/common ./nim/common
COPY nim/nvidia/nemoretriever-table-structure-v1 ./nim/nvidia/nemoretriever-table-structure-v1

WORKDIR /workspace/nim/nvidia/nemoretriever-table-structure-v1

RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev \
    && mv /workspace/.venv /opt/venv \
    && /opt/venv/bin/python -m ensurepip \
    && /opt/venv/bin/python -m pip install --no-cache-dir /opt/nemotron-table-structure-v1

FROM nvcr.io/nvidia/tritonserver:24.12-py3 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app/src"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git-lfs libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /workspace/nim/nvidia/nemoretriever-table-structure-v1/src /app/src
COPY --from=builder /workspace/nim/nvidia/nemoretriever-table-structure-v1/triton /app/triton
COPY --from=builder /workspace/nim/nvidia/nemoretriever-table-structure-v1/README.md README.md

EXPOSE 8000 8001 8002 8003 8100

USER 65532:65532

CMD ["uvicorn", "nemoretriever_table_structure_v1.server:app", "--host", "0.0.0.0", "--port", "8000"]
