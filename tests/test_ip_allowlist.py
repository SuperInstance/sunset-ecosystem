"""Tests for ip_allowlist.py — IP allowlist/denylist with CIDR support.

Run: python3 -m pytest tests/test_ip_allowlist.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.ip_allowlist import IPAllowlist


class TestIPAllowlist:
    def test_create(self):
        acl = IPAllowlist(default_allow=False)
        assert acl.stats()["default_allow"] is False

    def test_allow_ip(self):
        acl = IPAllowlist()
        acl.allow("192.168.1.100")
        assert acl.is_allowed("192.168.1.100") is True
        assert acl.is_allowed("192.168.1.101") is False

    def test_allow_cidr(self):
        acl = IPAllowlist()
        acl.allow("192.168.1.0/24")
        assert acl.is_allowed("192.168.1.100") is True
        assert acl.is_allowed("192.168.2.1") is False

    def test_deny_ip(self):
        acl = IPAllowlist(default_allow=True)
        acl.deny("10.0.0.5")
        assert acl.is_allowed("10.0.0.5") is False
        assert acl.is_allowed("10.0.0.6") is True

    def test_deny_cidr(self):
        acl = IPAllowlist(default_allow=True)
        acl.deny("10.0.0.0/24")
        assert acl.is_allowed("10.0.0.5") is False
        assert acl.is_allowed("10.0.1.1") is True

    def test_deny_takes_precedence(self):
        acl = IPAllowlist()
        acl.allow("192.168.1.0/24")
        acl.deny("192.168.1.100")
        assert acl.is_allowed("192.168.1.100") is False
        assert acl.is_allowed("192.168.1.101") is True

    def test_remove_allow(self):
        acl = IPAllowlist()
        acl.allow("192.168.1.0/24")
        assert acl.remove_allow("192.168.1.0/24") is True
        assert acl.is_allowed("192.168.1.100") is False
        assert acl.remove_allow("missing") is False

    def test_remove_deny(self):
        acl = IPAllowlist(default_allow=True)
        acl.deny("10.0.0.5")
        assert acl.remove_deny("10.0.0.5") is True
        assert acl.is_allowed("10.0.0.5") is True

    def test_invalid_ip(self):
        acl = IPAllowlist()
        assert acl.is_allowed("invalid") is False
        assert acl.allow("invalid") is False
        assert acl.deny("invalid") is False

    def test_default_allow(self):
        acl = IPAllowlist(default_allow=True)
        assert acl.is_allowed("1.2.3.4") is True

    def test_default_deny(self):
        acl = IPAllowlist(default_allow=False)
        assert acl.is_allowed("1.2.3.4") is False

    def test_is_denied(self):
        acl = IPAllowlist(default_allow=True)
        acl.deny("10.0.0.5")
        assert acl.is_denied("10.0.0.5") is True
        assert acl.is_denied("1.2.3.4") is False

    def test_allowed_list(self):
        acl = IPAllowlist()
        acl.allow("192.168.1.0/24")
        assert acl.allowed_list() == ["192.168.1.0/24"]

    def test_denied_list(self):
        acl = IPAllowlist()
        acl.deny("10.0.0.0/24")
        assert acl.denied_list() == ["10.0.0.0/24"]

    def test_stats(self):
        acl = IPAllowlist()
        acl.allow("192.168.1.0/24")
        acl.deny("10.0.0.0/24")
        stats = acl.stats()
        assert stats["allowed"] == 1
        assert stats["denied"] == 1
        assert stats["default_allow"] is False

    def test_repr(self):
        acl = IPAllowlist()
        assert "IPAllowlist" in repr(acl)
