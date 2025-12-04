# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from az_nv_ingest.azure import key_vault


def test_load_key_vault_secrets_disabled(monkeypatch):
    env: dict[str, str] = {}
    secret_client_called = False

    def _raise_if_called(*args, **kwargs):  # pragma: no cover - defensive
        nonlocal secret_client_called
        secret_client_called = True
        raise AssertionError("SecretClient should not be constructed when KV is disabled")

    monkeypatch.setattr(key_vault, "SecretClient", _raise_if_called)

    loaded = key_vault.load_key_vault_secrets(env=env)

    assert loaded == {}
    assert secret_client_called is False


def test_load_key_vault_secrets_prefers_workload_identity(monkeypatch):
    env = {
        "AZURE_KEY_VAULT_URI": "https://unit-test.vault.azure.net/",
        "AZURE_KEY_VAULT_SECRET_MAPPINGS": "APPINSIGHTS_CONNECTION_STRING=app-conn",
        "AZURE_TENANT_ID": "tenant-id",
        "AZURE_CLIENT_ID": "client-id",
        "AZURE_FEDERATED_TOKEN_FILE": "/tmp/token",
    }

    workload_called = {}
    default_called = {}

    def fake_workload_credential(**kwargs):
        workload_called.update(kwargs)
        return "workload-cred"

    def fake_default_credential(**kwargs):  # pragma: no cover - should not run
        default_called.update(kwargs)
        return "default-cred"

    class FakeSecretClient:
        def __init__(self, vault_url, credential):
            self.vault_url = vault_url
            self.credential = credential

        def get_secret(self, name):
            return SimpleNamespace(value=f"value-for-{name}")

    monkeypatch.setattr(key_vault, "WorkloadIdentityCredential", fake_workload_credential)
    monkeypatch.setattr(key_vault, "DefaultAzureCredential", fake_default_credential)
    monkeypatch.setattr(key_vault, "SecretClient", FakeSecretClient)

    env_map = env.copy()
    loaded = key_vault.load_key_vault_secrets(env=env_map)

    assert loaded == {"APPINSIGHTS_CONNECTION_STRING": "app-conn"}
    assert env_map["APPINSIGHTS_CONNECTION_STRING"] == "value-for-app-conn"
    assert workload_called["tenant_id"] == "tenant-id"
    assert workload_called["client_id"] == "client-id"
    assert workload_called["token_file_path"] == "/tmp/token"
    assert default_called == {}


def test_load_key_vault_skips_when_env_already_set(monkeypatch):
    env = {
        "AZURE_KEY_VAULT_URI": "https://unit-test.vault.azure.net/",
        "AZURE_KEY_VAULT_SECRET_MAPPINGS": "EXISTING=existing-secret",
        "EXISTING": "do-not-overwrite",
    }

    secret_client_called = False

    class FakeSecretClient:  # pragma: no cover - should not be called
        def __init__(self, *args, **kwargs):
            nonlocal secret_client_called
            secret_client_called = True

        def get_secret(self, name):
            raise AssertionError("should not fetch when env already set")

    monkeypatch.setattr(key_vault, "SecretClient", FakeSecretClient)

    loaded = key_vault.load_key_vault_secrets(env=env)

    assert loaded == {}
    assert env["EXISTING"] == "do-not-overwrite"
    assert secret_client_called is False
