# docker/ — AGENTS.md (container build + compose)

## Scope

Dockerfiles/scripts/compose wiring.

## az-nv-ingest rules

- Default compose must work without NGC/NIM authentication.
- If you keep optional NIM support, it must be behind an explicit profile/flag, never the default.

## Deliverables

- `docker-compose.yaml` (or additional compose files) for:
  - open path (default)
  - optional NVIDIA NIM path (non-default, clearly labeled)

## Testing

- Provide a smoke test script that:
  - starts services
  - ingests a sample file from `data/`
  - asserts non-empty extracted chunks + tables
