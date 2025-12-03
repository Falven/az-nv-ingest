# az-nv-ingest (fork of NVIDIA/nv-ingest) — AGENTS.md

## Purpose (read this first)

This fork (“az-nv-ingest”) keeps nv-ingest’s public behavior and JSON outputs **as close to upstream as possible**, while:

- Removing hard reliance on NVIDIA NIM / NGC API keys for core ingestion.
- Supporting **Nemotron Parse 1.1** as the primary VLM parser.
- Externalizing embeddings to **Azure OpenAI** (when configured).
- Externalizing vector storage to **Azure Cognitive Search** (when configured), otherwise retain upstream defaults (e.g., Milvus).

AGENTS: When working, always preserve compatibility contracts unless your task explicitly says “breaking change”.

## How agents should use instructions

- This repo uses **nested `AGENTS.md`** files: look for the closest one in the directory you are editing.
- If guidance conflicts, the **closest AGENTS.md** should be treated as authoritative for that area.

## Non‑negotiable invariants (do not violate)

1. **No required NGC/NIM keys** for default “open” ingestion path.
2. Preserve nv-ingest’s **job spec**, **chunk schema**, **content type labeling**, and **metadata fields** as much as possible.
3. Azure integrations must be **optional**:
   - If Azure config is missing, fallback stays functional with upstream defaults.
4. Never commit secrets (keys, tokens, connection strings). Use env vars / Kubernetes secrets / Key Vault references.
5. IaC must **NOT** deploy Azure OpenAI resources (consume an existing AOAI endpoint only).

## Repo map (high level)

- `src/` — server pipelines + orchestration + extraction stages (most core changes happen here)
- `api/` — api logic shared across python modules
- `client/` — nv-ingest-cli and client SDK; keep backward compatible
- `helm/` — Helm chart for K8s deployment
- `infra/` — (added by this fork) Bicep IaC for AKS + dependencies
- `config/` — OTEL/Prom config files
- `docker/` — container scripts and entrypoints

## Local dev: what to do before changing code

- Prefer working inside the repo’s devcontainer (`.devcontainer/`) if available.
- Run formatting/lint if configured: `pre-commit run -a`
- Run unit tests: `pytest -q` (or `pytest`)

If commands differ, follow the nearest AGENTS.md instructions for that directory.

## Definition of Done (DoD) for any PR

- Tests pass for the touched package(s).
- You updated docs/examples where behavior/config changed.
- You preserved public schemas/contracts, or you documented the break and added migration notes.
- You added at least one focused test for non-trivial logic (mock external services).

## “Open” replacements design principle

When replacing NIM services, build drop-in replacements that:

- return data in the exact shapes nv-ingest expects,
- are configurable (endpoints/timeouts),
- are testable via local docker compose and unit tests.

Do not refactor the entire pipeline unless necessary—prefer adapters/shims.
