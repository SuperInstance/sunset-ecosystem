"""Tests for lease_manager.py — Distributed lease/lock with TTL.

Run: python3 -m pytest tests/test_lease_manager.py -v --tb=short
"""

from __future__ import annotations

import time

import pytest

from fleet.lease_manager import LeaseManager


class TestLeaseManager:
    def test_create(self):
        mgr = LeaseManager()
        assert mgr.stats()["active"] == 0

    def test_acquire(self):
        mgr = LeaseManager()
        assert mgr.acquire("leader", "node-1", ttl_sec=1.0) is True
        assert mgr.is_owner("leader", "node-1") is True

    def test_acquire_conflict(self):
        mgr = LeaseManager()
        mgr.acquire("leader", "node-1", ttl_sec=1.0)
        assert mgr.acquire("leader", "node-2", ttl_sec=1.0) is False

    def test_acquire_after_expiry(self):
        mgr = LeaseManager()
        mgr.acquire("leader", "node-1", ttl_sec=0.03)
        time.sleep(0.05)
        assert mgr.acquire("leader", "node-2", ttl_sec=1.0) is True

    def test_renew(self):
        mgr = LeaseManager()
        mgr.acquire("leader", "node-1", ttl_sec=0.05)
        assert mgr.renew("leader", "node-1", ttl_sec=1.0) is True
        time.sleep(0.06)
        assert mgr.is_owner("leader", "node-1") is True

    def test_renew_wrong_owner(self):
        mgr = LeaseManager()
        mgr.acquire("leader", "node-1", ttl_sec=1.0)
        assert mgr.renew("leader", "node-2", ttl_sec=1.0) is False

    def test_release(self):
        mgr = LeaseManager()
        mgr.acquire("leader", "node-1", ttl_sec=1.0)
        assert mgr.release("leader", "node-1") is True
        assert mgr.is_owner("leader", "node-1") is False
        assert mgr.release("leader", "node-1") is False

    def test_force_release(self):
        mgr = LeaseManager()
        mgr.acquire("leader", "node-1", ttl_sec=1.0)
        assert mgr.force_release("leader") is True
        assert mgr.is_owner("leader", "node-1") is False

    def test_ttl(self):
        mgr = LeaseManager()
        mgr.acquire("leader", "node-1", ttl_sec=10.0)
        ttl = mgr.ttl("leader")
        assert ttl is not None
        assert 9.0 < ttl <= 10.0

    def test_list_leases(self):
        mgr = LeaseManager()
        mgr.acquire("a", "node-1", ttl_sec=1.0)
        mgr.acquire("b", "node-1", ttl_sec=1.0)
        assert sorted(mgr.list_leases()) == ["a", "b"]

    def test_list_by_owner(self):
        mgr = LeaseManager()
        mgr.acquire("a", "node-1", ttl_sec=1.0)
        mgr.acquire("b", "node-2", ttl_sec=1.0)
        assert mgr.list_by_owner("node-1") == ["a"]

    def test_get_owner(self):
        mgr = LeaseManager()
        mgr.acquire("leader", "node-1", ttl_sec=1.0)
        assert mgr.get_owner("leader") == "node-1"
        assert mgr.get_owner("missing") is None

    def test_purge(self):
        mgr = LeaseManager()
        mgr.acquire("a", "node-1", ttl_sec=1.0)
        mgr.purge()
        assert mgr.stats()["active"] == 0

    def test_repr(self):
        mgr = LeaseManager()
        mgr.acquire("a", "node-1", ttl_sec=1.0)
        assert "LeaseManager" in repr(mgr)
