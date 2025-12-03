# src/ — AGENTS.md (pipeline + adapters)

## Scope of this area

This directory contains the nv-ingest pipeline implementation: stages, schemas, orchestration, and model integrations.

Your #1 goal is to preserve output compatibility while swapping implementations behind interfaces.

## Golden rules

1. **Do not change output schema lightly.**
   - Prefer adding optional fields over renaming/removing fields.
2. Prefer adding new “provider” classes behind stable interfaces rather than rewriting stage logic.
3. All Azure integrations must be optional; defaults should still function.

## az-nv-ingest target behavior (in this directory)

### Model parsing

- Primary parser: **Nemotron Parse 1.1**.
- Ensure the pipeline can consume layout-rich outputs (blocks, bounding boxes, page structure) when available.

### Detection/table handling

- Keep the “extract tables / charts / graphics” behavior equivalent to upstream.
- If upstream expects separate page-elements / table-structure / graphic-elements outputs, provide compatible adapters.

### Embeddings

- Implement an embedding provider that can call **Azure OpenAI**.
- Ensure embedding logic can be disabled or switched without changing upstream task syntax.

### Vector store

- Implement a Vector DB operator/provider for **Azure Cognitive Search**.
- Preserve “upload” semantics: idempotency, metadata, and vector dimension handling.

## Testing requirements for changes here

- Add unit tests for:
  - adapter output shape (golden JSON fixtures)
  - error handling (timeouts, 429/5xx from Azure)
- Keep tests deterministic (no network). Use mocks and fixture JSON.

## Patterns to follow

- Prefer dependency injection via config objects (avoid global singletons).
- Timeouts/retries must be configurable and safe by default.
- Log structured events (job_id, document_id, stage_name, duration_ms).
