"""Tests for secrets_manager.py — Secrets management with encryption stubs.

Run: python3 -m pytest tests/test_secrets_manager.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.secrets_manager import SecretsManager


class TestSecretsManager:
    def test_create(self):
        mgr = SecretsManager()
        assert mgr.stats()["secrets"] == 0

    def test_set_and_get(self):
        mgr = SecretsManager()
        mgr.set_secret("api_key", "secret123")
        assert mgr.get_secret("api_key") == "secret123"

    def test_get_missing(self):
        mgr = SecretsManager()
        assert mgr.get_secret("missing") is None

    def test_delete_secret(self):
        mgr = SecretsManager()
        mgr.set_secret("api_key", "secret123")
        assert mgr.delete_secret("api_key") is True
        assert mgr.get_secret("api_key") is None
        assert mgr.delete_secret("missing") is False

    def test_list_secrets(self):
        mgr = SecretsManager()
        mgr.set_secret("a", "1")
        mgr.set_secret("b", "2")
        assert sorted(mgr.list_secrets()) == ["a", "b"]

    def test_has_secret(self):
        mgr = SecretsManager()
        mgr.set_secret("a", "1")
        assert mgr.has_secret("a") is True
        assert mgr.has_secret("b") is False

    def test_tamper_detection(self):
        mgr = SecretsManager()
        mgr.set_secret("api_key", "secret123")
        # Tamper with stored value
        mgr._secrets["api_key"]["value"] = "tampered"
        with pytest.raises(ValueError):
            mgr.get_secret("api_key")

    def test_repr(self):
        mgr = SecretsManager()
        assert "SecretsManager" in repr(mgr)
