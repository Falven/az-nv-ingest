# api/ — AGENTS.md

## Scope

Shared API logic used across python modules. Treat this as “public surface area”.

## Rules

- Avoid breaking public request/response types.
- If you add a new endpoint for az-nv-ingest, ensure it is:
  - versioned, or
  - optional and backward compatible.

## Tests

- Add contract tests for any endpoint changes.
- Prefer fast tests; use mocked downstream integrations.
