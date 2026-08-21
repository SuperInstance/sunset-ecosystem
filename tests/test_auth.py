"""Tests for auth.py — Token-based authentication and authorization.

Run: python3 -m pytest tests/test_auth.py -v --tb=short
"""

from __future__ import annotations

import time

import pytest

from fleet.auth import FleetAuth, TokenPayload, PermissionDenied


class TestFleetAuth:
    def test_create(self):
        auth = FleetAuth(secret_key="test-secret")
        assert repr(auth) == "FleetAuth(roles=4)"

    def test_create_token(self):
        auth = FleetAuth(secret_key="test-secret")
        token = auth.create_token("agent-1", roles=["breeder"], ttl=3600)
        assert isinstance(token, str)
        assert "." in token

    def test_validate_token(self):
        auth = FleetAuth(secret_key="test-secret")
        token = auth.create_token("agent-1", roles=["breeder"], ttl=3600)
        payload = auth.validate_token(token)
        assert payload is not None
        assert payload.subject == "agent-1"
        assert payload.roles == ["breeder"]
        assert payload.expires_at is not None

    def test_validate_expired(self):
        auth = FleetAuth(secret_key="test-secret")
        token = auth.create_token("agent-1", ttl=0.05)
        time.sleep(0.08)
        payload = auth.validate_token(token)
        assert payload is None

    def test_validate_bad_signature(self):
        auth = FleetAuth(secret_key="test-secret")
        token = auth.create_token("agent-1", ttl=3600)
        # Tamper with token
        bad_token = token[:-1] + ("1" if token[-1] != "1" else "0")
        payload = auth.validate_token(bad_token)
        assert payload is None

    def test_validate_malformed(self):
        auth = FleetAuth(secret_key="test-secret")
        assert auth.validate_token("not-a-token") is None
        assert auth.validate_token("only-one-part") is None

    def test_has_permission(self):
        auth = FleetAuth(secret_key="test-secret")
        token = auth.create_token("agent-1", roles=["breeder"])
        payload = auth.validate_token(token)
        assert auth.has_permission(payload, "breed") is True
        assert auth.has_permission(payload, "read") is True
        assert auth.has_permission(payload, "write") is True
        assert auth.has_permission(payload, "delete") is False

    def test_admin_permission(self):
        auth = FleetAuth(secret_key="test-secret")
        token = auth.create_token("admin", roles=["admin"])
        payload = auth.validate_token(token)
        assert auth.has_permission(payload, "anything") is True

    def test_require_permission(self):
        auth = FleetAuth(secret_key="test-secret")
        token = auth.create_token("agent-1", roles=["viewer"])
        payload = auth.validate_token(token)
        auth.require_permission(payload, "read")
        with pytest.raises(PermissionDenied):
            auth.require_permission(payload, "breed")

    def test_add_role(self):
        auth = FleetAuth(secret_key="test-secret")
        auth.add_role("custom", ["special"])
        token = auth.create_token("agent", roles=["custom"])
        payload = auth.validate_token(token)
        assert auth.has_permission(payload, "special") is True

    def test_token_info(self):
        auth = FleetAuth(secret_key="test-secret")
        token = auth.create_token("agent-1", roles=["breeder"], ttl=3600)
        info = auth.token_info(token)
        assert info is not None
        assert info["subject"] == "agent-1"
        assert info["roles"] == ["breeder"]
        assert info["is_expired"] is False

    def test_token_info_expired(self):
        auth = FleetAuth(secret_key="test-secret")
        token = auth.create_token("agent-1", ttl=0.1)
        time.sleep(0.15)
        info = auth.token_info(token)
        assert info is None

    def test_no_ttl(self):
        auth = FleetAuth(secret_key="test-secret")
        token = auth.create_token("agent-1", ttl=None)
        payload = auth.validate_token(token)
        assert payload.expires_at is None

    def test_default_roles(self):
        auth = FleetAuth(secret_key="test-secret")
        token = auth.create_token("agent-1")
        payload = auth.validate_token(token)
        assert payload.roles == ["viewer"]
