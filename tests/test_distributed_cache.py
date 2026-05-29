"""Tests for distributed_cache.py — TTL cache with pattern invalidation.

Run: python3 -m pytest tests/test_distributed_cache.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.distributed_cache import DistributedCache


class TestDistributedCache:
    def test_create(self):
        cache = DistributedCache()
        assert cache.size() == 0

    def test_set_get(self):
        cache = DistributedCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing(self):
        cache = DistributedCache()
        assert cache.get("missing", default="fallback") == "fallback"
        assert cache.get("missing") is None

    def test_ttl_expiry(self):
        cache = DistributedCache(ttl_sec=0.05)
        cache.set("key", "val")
        assert cache.get("key") == "val"
        time.sleep(0.06)
        assert cache.get("key") is None

    def test_custom_ttl(self):
        cache = DistributedCache(ttl_sec=10.0)
        cache.set("key", "val", ttl=0.05)
        time.sleep(0.06)
        assert cache.get("key") is None

    def test_delete(self):
        cache = DistributedCache()
        cache.set("key", "val")
        assert cache.delete("key") is True
        assert cache.delete("key") is False

    def test_invalidate_pattern(self):
        cache = DistributedCache()
        cache.set("room:a:trap", 1)
        cache.set("room:a:treasure", 2)
        cache.set("room:b:trap", 3)
        n = cache.invalidate("room:a:*")
        assert n == 2
        assert cache.get("room:a:trap") is None
        assert cache.get("room:b:trap") == 3

    def test_has(self):
        cache = DistributedCache()
        cache.set("key", "val")
        assert cache.has("key") is True
        assert cache.has("missing") is False

    def test_keys(self):
        cache = DistributedCache()
        cache.set("a", 1)
        cache.set("b", 2)
        assert sorted(cache.keys()) == ["a", "b"]

    def test_clear(self):
        cache = DistributedCache()
        cache.set("a", 1)
        cache.clear()
        assert cache.size() == 0

    def test_max_size_eviction(self):
        cache = DistributedCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)
        assert cache.size() == 3

    def test_hit_rate(self):
        cache = DistributedCache()
        cache.set("a", 1)
        cache.get("a")
        cache.get("a")
        cache.get("missing")
        assert cache.hit_rate() == 2 / 3

    def test_stats(self):
        cache = DistributedCache()
        cache.set("a", 1)
        cache.get("a")
        cache.get("missing")
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_repr(self):
        cache = DistributedCache()
        assert "DistributedCache" in repr(cache)
