# Open shim services

Deterministic HTTP shims that mirror the nv-ingest contracts for page elements, table structure, graphic elements, OCR, and Nemotron Parse 1.1. Each service exposes `/health`, `/v1/health/ready`, and an inference endpoint (`/v1/infer` or `/v1/chat/completions`) that returns fixed, contract-compliant payloads suitable for local testing and containerized use.
