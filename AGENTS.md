# open-nv-ingest

## Overview

- Helm chart in `helm/` (native Deployments/Services for NIMs; no NIM Operator). Main overrides for Azure live in `helm/overrides/values-azure-acr.yaml`.
- Deployment docs: `docs/azure/05-gpu-time-slicing.md`, `docs/azure/06-helm-values-acr.md`, `docs/azure/07-helm-deploy-open-nv-ingest.md`, `docs/azure/08-smoke-and-expose.md`.
- Python/TypeScript repo; keep changes minimal to stay near upstream defaults.

## Build and push nv-ingest image to ACR (required before Helm)

- Ensure Azure CLI logged in: `az account show` should work.
- Preferred (local buildx, runtime stage):  
  `docker buildx build --builder codex-builder --platform linux/amd64 -t acrapisixprodeusplatformecynr3.azurecr.io/nvidia/nemo-microservices/nv-ingest:25.9.0 --build-arg RELEASE_TYPE=release --build-arg VERSION=25.9.0 --build-arg VERSION_REV=0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) --target runtime --push .`
- Remote ACR build (slower, BuildKit-less):  
  `az acr build -r acrapisixprodeusplatformecynr3 -t nvidia/nemo-microservices/nv-ingest:25.9.0 --platform linux/amd64 --build-arg RELEASE_TYPE=release --build-arg VERSION=25.9.0 --build-arg VERSION_REV=0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .`
- Chart expects image at `acrapisixprodeusplatformecynr3.azurecr.io/nvidia/nemo-microservices/nv-ingest:25.9.0`; verify with `az acr repository show-tags -n acrapisixprodeusplatformecynr3 --repository nvidia/nemo-microservices/nv-ingest`.
- Packaging quirk: two top-level packages (`nv_ingest`, `az_nv_ingest`); `src/pyproject.toml` pins package discovery to include both to avoid setuptools “multiple top-level packages” errors during the Docker build.

## Helm deploy (Azure opinionated path)

- Update deps: `helm dependency update ./helm`.
- Deploy: `helm upgrade --install nv-ingest ./helm -n nv-ingest -f helm/overrides/values-azure-acr.yaml`.
- `helm/overrides/values-azure-acr.yaml` sets ACR repos, `imagePullSecrets: acr-pull`, disables NGC secrets, trims nv-ingest resources, adds GPU nodeSelectors/tolerations for NIMs. Keep overrides minimal; prefer editing this file instead of base values.

## GPU/time-slicing requirements

- Apply time-slicing config and patch ClusterPolicy per `docs/azure/05-gpu-time-slicing.md` (use `helm/time-slicing/time-slicing-config.yaml`).
- Ensure GPU node pool exists; label/taint GPU nodes: `kubernetes.azure.com/agentpool=gpu` and taint `nvidia.com/gpu=true:NoSchedule`.
- NIM Deployments request 1 GPU; without labels/taints they stay Pending.

## Image pull auth

- Prefer ACR attach. If using pull secrets, namespace `nv-ingest` needs a working secret named `acr-pull` with credentials that can read `acrapisixprodeusplatformecynr3.azurecr.io/*`. ImagePullBackOff usually means missing image or bad credentials.

## Smoke test

- Follow `docs/azure/08-smoke-and-expose.md`: port-forward nv-ingest, check `/health/ready`, submit sample PDF, wait for completion.

## Conventions

- Use `pnpm` when touching JS/TS tooling; prefer ESM, named imports, arrow functions.
- Prefer pure, modular functions; avoid classes unless stateful behavior is required.
- Validate external inputs; fail fast with typed errors; avoid `any`/non-null assertions.
