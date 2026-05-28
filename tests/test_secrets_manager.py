"""Tests for secrets_manager.py — Secure secret storage.

Run: python3 -m pytest tests/test_secrets_manager.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.secrets_manager import SecretsManager, SecretNotFound


class TestSecretsManager:
    def test_create(self):
        sm = SecretsManager(master_key="test-key-32-bytes-long-for-test")
        assert sm.stats()["namespaces"] == 0

    def test_set_and_get(self):
        sm = SecretsManager(master_key="test-key-32-bytes-long-for-test")
        sm.set("api_key", "sk-12345")
        assert sm.get("api_key") == "sk-12345"

    def test_get_bytes(self):
        sm = SecretsManager(master_key="test-key-32-bytes-long-for-test")
        sm.set("binary", b"\x80\x81\x82")  # Invalid UTF-8
        with pytest.raises((UnicodeDecodeError, ValueError)):
            sm.get("binary")

    def test_namespace_isolation(self):
        sm = SecretsManager(master_key="test-key-32-bytes-long-for-test")
        sm.set("key", "prod-value", namespace="production")
        sm.set("key", "dev-value", namespace="development")
        assert sm.get("key", namespace="production") == "prod-value"
        assert sm.get("key", namespace="development") == "dev-value"

    def test_get_missing(self):
        sm = SecretsManager(master_key="test-key-32-bytes-long-for-test")
        with pytest.raises(SecretNotFound):
            sm.get("missing")

    def test_exists(self):
        sm = SecretsManager(master_key="test-key-32-bytes-long-for-test")
        sm.set("key", "value")
        assert sm.exists("key") is True
        assert sm.exists("missing") is False

    def test_delete(self):
        sm = SecretsManager(master_key="test-key-32-bytes-long-for-test")
        sm.set("key", "value")
        assert sm.delete("key") is True
        assert sm.exists("key") is False
        assert sm.delete("key") is False

    def test_list_secrets(self):
        sm = SecretsManager(master_key="test-key-32-bytes-long-for-test")
        sm.set("a", "1")
        sm.set("b", "2")
        assert sorted(sm.list_secrets()) == ["a", "b"]

    def test_list_namespaces(self):
        sm = SecretsManager(master_key="test-key-32-bytes-long-for-test")
        sm.set("k", "v", namespace="ns1")
        sm.set("k", "v", namespace="ns2")
        assert sorted(sm.list_namespaces()) == ["ns1", "ns2"]

    def test_rotate(self):
        sm = SecretsManager(master_key="test-key-32-bytes-long-for-test")
        sm.set("key", "value")
        v1 = sm._secrets["default"]["key"].version
        sm.rotate("key")
        v2 = sm._secrets["default"]["key"].version
        assert v2 == v1 + 1
        # Still readable after rotation
        assert sm.get("key") == "value"

    def test_stats(self):
        sm = SecretsManager(master_key="test-key-32-bytes-long-for-test")
        sm.set("a", "1", namespace="ns1")
        sm.set("b", "2", namespace="ns1")
        sm.set("c", "3", namespace="ns2")
        stats = sm.stats()
        assert stats["namespaces"] == 2
        assert stats["total_secrets"] == 3

    def test_different_keys(self):
        sm1 = SecretsManager(master_key="key-a")
        sm2 = SecretsManager(master_key="key-b")
        sm1.set("x", "secret-a")
        sm2.set("x", "secret-b")
        assert sm1.get("x") == "secret-a"
        assert sm2.get("x") == "secret-b"

    def test_metadata(self):
        sm = SecretsManager(master_key="test-key-32-bytes-long-for-test")
        sm.set("key", "value", metadata={"owner": "test"})
        entry = sm._secrets["default"]["key"]
        assert entry.metadata["owner"] == "test"

    def test_repr(self):
        sm = SecretsManager(master_key="test-key-32-bytes-long-for-test")
        assert "SecretsManager" in repr(sm)
