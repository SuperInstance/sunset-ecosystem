"""Tests for distributed_lock.py — Distributed locking.

Run: python3 -m pytest tests/test_distributed_lock.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.distributed_lock import DistributedLock, LockResult


class TestDistributedLock:
    def test_create(self):
        lock = DistributedLock(node_id="node-a")
        assert lock._node_id == "node-a"

    def test_acquire(self):
        lock = DistributedLock(node_id="node-a")
        result = lock.acquire("resource-x", ttl=10.0)
        assert result.success is True
        assert result.holder == "node-a"

    def test_acquire_conflict(self):
        # Same lock instance for shared state
        lock = DistributedLock(node_id="node-a")
        lock.acquire("resource-x", ttl=10.0)
        # Simulate another node by changing node_id temporarily
        lock_b = DistributedLock(node_id="node-b")
        # Copy lock state to lock_b for test
        lock_b._locks = lock._locks
        result = lock_b.acquire("resource-x", ttl=10.0)
        assert result.success is False
        assert result.holder == "node-a"

    def test_release(self):
        lock = DistributedLock(node_id="node-a")
        lock.acquire("resource-x", ttl=10.0)
        result = lock.release("resource-x")
        assert result.success is True
        assert not lock.is_held("resource-x")

    def test_release_not_holder(self):
        lock_a = DistributedLock(node_id="node-a")
        lock_b = DistributedLock(node_id="node-b")
        lock_a.acquire("resource-x", ttl=10.0)
        result = lock_b.release("resource-x")
        assert result.success is False

    def test_release_not_held(self):
        lock = DistributedLock(node_id="node-a")
        result = lock.release("missing")
        assert result.success is False

    def test_renew(self):
        lock = DistributedLock(node_id="node-a")
        lock.acquire("resource-x", ttl=1.0)
        before = lock.time_remaining("resource-x")
        time.sleep(0.1)
        result = lock.renew("resource-x", ttl=5.0)
        assert result.success is True
        after = lock.time_remaining("resource-x")
        assert after > before

    def test_renew_not_holder(self):
        lock_a = DistributedLock(node_id="node-a")
        lock_b = DistributedLock(node_id="node-b")
        lock_a.acquire("resource-x", ttl=10.0)
        result = lock_b.renew("resource-x", ttl=5.0)
        assert result.success is False

    def test_expiration(self):
        lock = DistributedLock(node_id="node-a")
        lock.acquire("resource-x", ttl=0.1)
        assert lock.is_held("resource-x") is True
        time.sleep(0.15)
        assert lock.is_held("resource-x") is False

    def test_expire_all(self):
        lock = DistributedLock(node_id="node-a")
        lock.acquire("a", ttl=0.05)
        lock.acquire("b", ttl=10.0)
        time.sleep(0.1)
        removed = lock.expire_all()
        assert removed == 1
        assert not lock.is_held("a")
        assert lock.is_held("b")

    def test_double_acquire_same_node(self):
        lock = DistributedLock(node_id="node-a")
        r1 = lock.acquire("x", ttl=10.0)
        r2 = lock.acquire("x", ttl=10.0)
        assert r1.success is True
        assert r2.success is True
        assert "already held" in r2.message

    def test_is_holder(self):
        lock = DistributedLock(node_id="node-a")
        lock.acquire("x", ttl=10.0)
        assert lock.is_holder("x") is True

    def test_holder_query(self):
        lock_a = DistributedLock(node_id="node-a")
        lock_a.acquire("x", ttl=10.0)
        assert lock_a.holder("x") == "node-a"

    def test_time_remaining(self):
        lock = DistributedLock(node_id="node-a")
        lock.acquire("x", ttl=2.0)
        tr = lock.time_remaining("x")
        assert 1.0 < tr <= 2.0

    def test_stats(self):
        lock = DistributedLock(node_id="node-a")
        lock.acquire("x", ttl=10.0)
        s = lock.stats()
        assert s["locks_held"] == 1

    def test_stats_per_lock(self):
        lock = DistributedLock(node_id="node-a")
        lock.acquire("x", ttl=10.0)
        lock.release("x")
        s = lock.stats("x")
        assert s.get("acquisitions", 0) == 1
        assert s.get("releases", 0) == 1

    def test_repr(self):
        lock = DistributedLock(node_id="a")
        assert "DistributedLock" in repr(lock)
