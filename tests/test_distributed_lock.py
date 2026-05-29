"""Tests for distributed_lock.py — Distributed lock with TTL.

Run: python3 -m pytest tests/test_distributed_lock.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.distributed_lock import DistributedLock


class TestDistributedLock:
    def test_create(self):
        lock = DistributedLock("task-1", ttl_sec=30)
        assert lock.lock_id == "task-1"
        assert lock.ttl_sec == 30

    def test_acquire(self):
        lock = DistributedLock("task-1")
        assert lock.acquire("node-1") is True
        assert lock.is_locked() is True
        assert lock.owner() == "node-1"

    def test_acquire_blocked(self):
        lock = DistributedLock("task-1")
        lock.acquire("node-1")
        assert lock.acquire("node-2") is False

    def test_release(self):
        lock = DistributedLock("task-1")
        lock.acquire("node-1")
        assert lock.release("node-1") is True
        assert lock.is_locked() is False
        assert lock.owner() is None

    def test_release_wrong_owner(self):
        lock = DistributedLock("task-1")
        lock.acquire("node-1")
        assert lock.release("node-2") is False
        assert lock.is_locked() is True

    def test_renew(self):
        lock = DistributedLock("task-1", ttl_sec=1, clock=lambda: 0)
        lock.acquire("node-1")
        # Expire
        lock._clock = lambda: 2
        assert lock.is_locked() is False
        # Renew before expire
        lock._clock = lambda: 0
        lock.renew("node-1")
        lock._clock = lambda: 0.5
        assert lock.is_locked() is True

    def test_renew_wrong_owner(self):
        lock = DistributedLock("task-1")
        lock.acquire("node-1")
        assert lock.renew("node-2") is False

    def test_time_remaining(self):
        lock = DistributedLock("task-1", ttl_sec=10, clock=lambda: 0)
        lock.acquire("node-1")
        lock._clock = lambda: 3
        assert lock.time_remaining() == 7

    def test_time_remaining_expired(self):
        lock = DistributedLock("task-1", ttl_sec=10, clock=lambda: 0)
        lock.acquire("node-1")
        lock._clock = lambda: 15
        assert lock.time_remaining() == 0

    def test_acquire_after_expire(self):
        lock = DistributedLock("task-1", ttl_sec=1, clock=lambda: 0)
        lock.acquire("node-1")
        lock._clock = lambda: 2
        assert lock.acquire("node-2") is True

    def test_stats(self):
        lock = DistributedLock("task-1", ttl_sec=30)
        lock.acquire("node-1")
        stats = lock.stats()
        assert stats["locked"] is True
        assert stats["owner"] == "node-1"

    def test_repr(self):
        lock = DistributedLock("task-1")
        assert "DistributedLock" in repr(lock)
