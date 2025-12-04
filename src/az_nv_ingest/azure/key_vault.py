# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal helpers for optional Azure Key Vault integration.

These helpers intentionally avoid side effects unless a Key Vault URI and
explicit secret mappings are provided. When enabled they:
- Prefer Azure Workload Identity for authentication.
- Fall back to the broader DefaultAzureCredential chain when WLI is absent.
- Populate the provided environment mapping with fetched secrets, without
  logging or returning secret values.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Mapping, MutableMapping

from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential, WorkloadIdentityCredential
from azure.keyvault.secrets import SecretClient

logger = logging.getLogger(__name__)

_WORKLOAD_IDENTITY_ENV_KEYS = (
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_FEDERATED_TOKEN_FILE",
)


def _has_workload_identity(env: Mapping[str, str]) -> bool:
    return all(env.get(key) for key in _WORKLOAD_IDENTITY_ENV_KEYS)


def build_key_vault_credential(env: Mapping[str, str] | None = None):
    """Return a credential preferring workload identity over env credentials."""

    env = env or os.environ

    if _has_workload_identity(env):
        return WorkloadIdentityCredential(
            tenant_id=env.get("AZURE_TENANT_ID"),
            client_id=env.get("AZURE_CLIENT_ID"),
            token_file_path=env.get("AZURE_FEDERATED_TOKEN_FILE"),
        )

    return DefaultAzureCredential(
        exclude_interactive_browser_credential=True,
        exclude_powershell_credential=True,
        exclude_visual_studio_code_credential=True,
    )


def parse_secret_mappings(raw_mappings: str | None) -> Dict[str, str]:
    """Parse a comma-separated mapping string like ``ENV_VAR=secret-name``."""

    if not raw_mappings:
        return {}

    mappings: Dict[str, str] = {}
    for pair in raw_mappings.split(","):
        if "=" not in pair:
            continue
        env_key, secret_name = pair.split("=", 1)
        env_key = env_key.strip()
        secret_name = secret_name.strip()
        if not env_key or not secret_name:
            continue
        mappings[env_key] = secret_name

    return mappings


def load_key_vault_secrets(
    env: MutableMapping[str, str] | None = None,
    secret_mappings: Mapping[str, str] | None = None,
    credential=None,
    client: SecretClient | None = None,
) -> Dict[str, str]:
    """Optionally hydrate environment values from Azure Key Vault.

    Returns a mapping of environment variable names to the Key Vault secret
    names used. Secret values are written into ``env`` (defaults to
    ``os.environ``) but never logged or returned.
    """

    env = env or os.environ
    vault_uri = env.get("AZURE_KEY_VAULT_URI")
    if not vault_uri:
        return {}

    mappings = secret_mappings or parse_secret_mappings(env.get("AZURE_KEY_VAULT_SECRET_MAPPINGS"))
    if not mappings:
        logger.info(
            "AZURE_KEY_VAULT_URI provided without AZURE_KEY_VAULT_SECRET_MAPPINGS; skipping Key Vault lookups.",
        )
        return {}

    missing_mappings = {env_name: secret_name for env_name, secret_name in mappings.items() if not env.get(env_name)}
    if not missing_mappings:
        return {}

    active_credential = credential or build_key_vault_credential(env)
    secret_client = client or SecretClient(vault_url=vault_uri, credential=active_credential)

    loaded: Dict[str, str] = {}
    for env_name, secret_name in missing_mappings.items():
        try:
            secret_value = secret_client.get_secret(secret_name).value
        except HttpResponseError as exc:  # pragma: no cover - relies on azure internals
            raise RuntimeError(
                f"Failed to read secret '{secret_name}' from Key Vault at '{vault_uri}'.",
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Unable to fetch secret '{secret_name}' from Key Vault at '{vault_uri}'.",
            ) from exc

        env[env_name] = secret_value
        loaded[env_name] = secret_name

    if loaded:
        logger.info("Loaded %d secret(s) from Key Vault into runtime environment.", len(loaded))

    return loaded
