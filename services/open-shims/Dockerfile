# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12.10

FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENV=.venv

WORKDIR /app

RUN pip install --no-cache-dir uv==0.7.6

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev

FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    SHIM_KIND=page-elements

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/pyproject.toml
COPY --from=builder /app/README.md /app/README.md

EXPOSE 8000

USER 65532:65532

CMD ["uvicorn", "open_shims.main:app", "--host", "0.0.0.0", "--port", "8000"]
