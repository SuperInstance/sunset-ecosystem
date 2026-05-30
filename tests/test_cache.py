"""Tests for cache.py — LRU + TTL caching layer.

Run: python3 -m pytest tests/test_cache.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.cache import FleetCache, CacheStats


class TestFleetCache:
    def test_create(self):
        cache = FleetCache(max_size=100)
        assert cache.max_size() == 100
        assert cache.size() == 0

    def test_set_get(self):
        cache = FleetCache()
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_get_missing(self):
        cache = FleetCache()
        assert cache.get("missing") is None
        assert cache.get("missing", default="fallback") == "fallback"

    def test_ttl_expiration(self):
        cache = FleetCache()
        cache.set("key", "value", ttl=0.05)
        assert cache.get("key") == "value"
        time.sleep(0.08)
        assert cache.get("key") is None

    def test_default_ttl(self):
        cache = FleetCache(default_ttl=0.05)
        cache.set("key", "value")
        assert cache.get("key") == "value"
        time.sleep(0.08)
        assert cache.get("key") is None

    def test_override_default_ttl(self):
        cache = FleetCache(default_ttl=60.0)
        cache.set("key", "value", ttl=0.05)
        assert cache.get("key") == "value"
        time.sleep(0.08)
        assert cache.get("key") is None

    def test_lru_eviction(self):
        cache = FleetCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # evicts 'a'
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_lru_order(self):
        cache = FleetCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.get("a")  # move 'a' to most-recent
        cache.set("d", 4)  # evicts 'b'
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_delete(self):
        cache = FleetCache()
        cache.set("key", "value")
        assert cache.delete("key") is True
        assert cache.delete("key") is False
        assert cache.get("key") is None

    def test_has(self):
        cache = FleetCache()
        cache.set("key", "value")
        assert cache.has("key") is True
        cache.set("key2", "value", ttl=0.1)
        time.sleep(0.15)
        assert cache.has("key2") is False

    def test_clear(self):
        cache = FleetCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.size() == 0
        assert cache.get("a") is None

    def test_get_many(self):
        cache = FleetCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        result = cache.get_many(["a", "b", "missing"])
        assert result == {"a": 1, "b": 2}

    def test_set_many(self):
        cache = FleetCache()
        cache.set_many({"a": 1, "b": 2}, ttl=60.0)
        assert cache.get("a") == 1
        assert cache.get("b") == 2

    def test_delete_many(self):
        cache = FleetCache()
        cache.set_many({"a": 1, "b": 2, "c": 3})
        deleted = cache.delete_many(["a", "b", "missing"])
        assert deleted == 2
        assert cache.get("c") == 3

    def test_expire_all(self):
        cache = FleetCache()
        cache.set("a", 1, ttl=0.05)
        cache.set("b", 2, ttl=60.0)
        time.sleep(0.1)
        removed = cache.expire_all()
        assert removed == 1
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_keys(self):
        cache = FleetCache()
        cache.set("a", 1)
        cache.set("b", 2)
        keys = cache.keys()
        assert sorted(keys) == ["a", "b"]

    def test_stats(self):
        cache = FleetCache()
        cache.get("missing")
        cache.set("key", "value")
        cache.get("key")
        cache.get("key")
        stats = cache.stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.hit_rate == pytest.approx(2 / 3)
        assert stats.size == 1

    def test_stats_evictions(self):
        cache = FleetCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        stats = cache.stats()
        assert stats.evictions == 1

    def test_stats_expired(self):
        cache = FleetCache()
        cache.set("a", 1, ttl=0.05)
        time.sleep(0.1)
        cache.get("a")
        stats = cache.stats()
        assert stats.expired == 1

    def test_repr(self):
        cache = FleetCache(max_size=100)
        assert "FleetCache" in repr(cache)
