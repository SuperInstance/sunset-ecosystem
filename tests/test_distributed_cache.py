"""Tests for distributed_cache.py — Distributed cache with TTL.

Run: python3 -m pytest tests/test_distributed_cache.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.distributed_cache import DistributedCache


class TestDistributedCache:
    def test_create(self):
        cache = DistributedCache(default_ttl_sec=300, clock=lambda: 0)
        assert cache.stats()["size"] == 0

    def test_set_get(self):
        cache = DistributedCache(clock=lambda: 0)
        cache.set("key-1", "value-1")
        assert cache.get("key-1") == "value-1"

    def test_get_missing(self):
        cache = DistributedCache(clock=lambda: 0)
        assert cache.get("missing") is None
        assert cache.stats()["misses"] == 1

    def test_ttl_expiration(self):
        cache = DistributedCache(clock=lambda: 0)
        cache.set("key-1", "value-1", ttl_sec=10)
        assert cache.get("key-1") == "value-1"
        cache._clock = lambda: 15
        assert cache.get("key-1") is None
        assert cache.stats()["misses"] == 1

    def test_delete(self):
        cache = DistributedCache(clock=lambda: 0)
        cache.set("key-1", "value-1")
        assert cache.delete("key-1") is True
        assert cache.get("key-1") is None
        assert cache.delete("missing") is False

    def test_invalidate(self):
        cache = DistributedCache(clock=lambda: 0)
        cache.set("key-1", "value-1")
        assert cache.invalidate("key-1") is True
        assert cache.get("key-1") is None

    def test_invalidate_pattern(self):
        cache = DistributedCache(clock=lambda: 0)
        cache.set("user:1", "a")
        cache.set("user:2", "b")
        cache.set("post:1", "c")
        count = cache.invalidate_pattern("user:")
        assert count == 2
        assert cache.get("user:1") is None
        assert cache.get("post:1") == "c"

    def test_clear(self):
        cache = DistributedCache(clock=lambda: 0)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.stats()["size"] == 0

    def test_keys(self):
        cache = DistributedCache(clock=lambda: 0)
        cache.set("a", 1)
        cache.set("b", 2)
        assert sorted(cache.keys()) == ["a", "b"]

    def test_has(self):
        cache = DistributedCache(clock=lambda: 0)
        cache.set("a", 1, ttl_sec=10)
        assert cache.has("a") is True
        cache._clock = lambda: 20
        assert cache.has("a") is False

    def test_ttl(self):
        cache = DistributedCache(clock=lambda: 0)
        cache.set("a", 1, ttl_sec=100)
        assert cache.ttl("a") == 100
        cache._clock = lambda: 50
        assert cache.ttl("a") == 50

    def test_ttl_expired(self):
        cache = DistributedCache(clock=lambda: 0)
        cache.set("a", 1, ttl_sec=10)
        cache._clock = lambda: 20
        assert cache.ttl("a") is None

    def test_stats(self):
        cache = DistributedCache(clock=lambda: 0)
        cache.set("a", 1)
        cache.get("a")  # hit
        cache.get("b")  # miss
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["hit_rate"] == 0.5

    def test_repr(self):
        cache = DistributedCache()
        assert "DistributedCache" in repr(cache)
