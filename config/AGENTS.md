# config/ — AGENTS.md (observability config)

## Scope

OTEL, Prometheus, logging configs.

## Rules

- Keep defaults safe and low-noise.
- If enabling Application Insights:
  - prefer OTEL exporter configuration via env vars
  - do not bake secrets into files

## Deliverables

- Document required env vars / helm values to enable telemetry.
- Provide sample dashboards (links/notes) but no secrets.
