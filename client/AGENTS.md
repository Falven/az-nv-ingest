# client/ — AGENTS.md (CLI + SDK)

## Scope

This folder contains nv-ingest’s client utilities and examples.
For az-nv-ingest, compatibility matters more than elegance.

## Rules

- Maintain CLI flags and Python API compatibility unless explicitly migrating.
- If you add Azure configuration support:
  - keep existing options working
  - add new flags/env vars without breaking old ones

## Examples

- Update notebooks/scripts to show:
  - embeddings via Azure OpenAI (text-embedding-3-large example)
  - vector upload via Azure Cognitive Search
- Never include secrets in example outputs.

## Tests

- Add minimal tests for:
  - config parsing
  - request formation (no network calls)
