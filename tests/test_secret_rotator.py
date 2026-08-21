"""Tests for secret_rotator.py — Secret rotation with dual-key support and grace period.

Run: python3 -m pytest tests/test_secret_rotator.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.secret_rotator import SecretRotator


class TestSecretRotator:
    def test_create(self):
        rotator = SecretRotator(
            default_ttl_sec=3600, grace_period_sec=300, clock=lambda: 0
        )
        assert rotator.stats()["secrets"] == 0

    def test_set_get(self):
        rotator = SecretRotator(clock=lambda: 0)
        rotator.set_secret("api-key", "secret-123")
        assert rotator.get_secret("api-key") == "secret-123"

    def test_get_missing(self):
        rotator = SecretRotator(clock=lambda: 0)
        assert rotator.get_secret("missing") is None

    def test_expiration(self):
        rotator = SecretRotator(default_ttl_sec=10, clock=lambda: 0)
        rotator.set_secret("api-key", "secret-123")
        assert rotator.get_secret("api-key") == "secret-123"
        rotator._clock = lambda: 15
        assert rotator.get_secret("api-key") is None

    def test_rotate(self):
        rotator = SecretRotator(clock=lambda: 0)
        rotator.set_secret("api-key", "old-secret")
        assert rotator.rotate("api-key", "new-secret") is True
        assert rotator.get_secret("api-key") == "new-secret"
        assert rotator.get_previous("api-key") == "old-secret"

    def test_rotate_missing(self):
        rotator = SecretRotator(clock=lambda: 0)
        assert rotator.rotate("missing", "new") is False

    def test_grace_period(self):
        rotator = SecretRotator(grace_period_sec=10, clock=lambda: 0)
        rotator.set_secret("api-key", "old-secret")
        rotator.rotate("api-key", "new-secret")
        assert rotator.get_previous("api-key") == "old-secret"
        rotator._clock = lambda: 15
        assert rotator.get_previous("api-key") is None

    def test_verify(self):
        rotator = SecretRotator(clock=lambda: 0)
        rotator.set_secret("api-key", "secret-123")
        assert rotator.verify("api-key", "secret-123") is True
        assert rotator.verify("api-key", "wrong") is False

    def test_verify_grace_period(self):
        rotator = SecretRotator(grace_period_sec=10, clock=lambda: 0)
        rotator.set_secret("api-key", "old-secret")
        rotator.rotate("api-key", "new-secret")
        assert rotator.verify("api-key", "old-secret") is True
        assert rotator.verify("api-key", "new-secret") is True
        assert rotator.verify("api-key", "wrong") is False

    def test_remove(self):
        rotator = SecretRotator(clock=lambda: 0)
        rotator.set_secret("api-key", "secret")
        assert rotator.remove("api-key") is True
        assert rotator.get_secret("api-key") is None
        assert rotator.remove("missing") is False

    def test_list_secrets(self):
        rotator = SecretRotator(clock=lambda: 0)
        rotator.set_secret("a", "1")
        rotator.set_secret("b", "2")
        assert sorted(rotator.list_secrets()) == ["a", "b"]

    def test_is_expired(self):
        rotator = SecretRotator(default_ttl_sec=10, clock=lambda: 0)
        rotator.set_secret("api-key", "secret")
        assert rotator.is_expired("api-key") is False
        rotator._clock = lambda: 15
        assert rotator.is_expired("api-key") is True

    def test_time_to_expiry(self):
        rotator = SecretRotator(default_ttl_sec=100, clock=lambda: 0)
        rotator.set_secret("api-key", "secret")
        assert rotator.time_to_expiry("api-key") == 100.0
        rotator._clock = lambda: 50
        assert rotator.time_to_expiry("api-key") == 50.0

    def test_stats(self):
        rotator = SecretRotator(default_ttl_sec=10, clock=lambda: 0)
        rotator.set_secret("a", "1")
        rotator.set_secret("b", "2")
        rotator._clock = lambda: 15
        stats = rotator.stats()
        assert stats["secrets"] == 2
        assert stats["expired"] == 2
        assert stats["active"] == 0
        assert stats["default_ttl"] == 10
        assert stats["grace_period"] == 300

    def test_repr(self):
        rotator = SecretRotator()
        assert "SecretRotator" in repr(rotator)
