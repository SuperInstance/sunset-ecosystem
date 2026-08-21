"""Tests for dns_cache.py — DNS cache with TTL and negative caching.

Run: python3 -m pytest tests/test_dns_cache.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.dns_cache import DNSCache


class TestDNSCache:
    def test_create(self):
        cache = DNSCache(default_ttl_sec=300, negative_ttl_sec=60, clock=lambda: 0)
        assert cache.stats()["size"] == 0

    def test_put_resolve(self):
        cache = DNSCache(clock=lambda: 0)
        cache.put("svc-a.fleet.local", ["192.168.1.1"])
        assert cache.resolve("svc-a.fleet.local") == ["192.168.1.1"]

    def test_resolve_missing(self):
        cache = DNSCache(clock=lambda: 0)
        assert cache.resolve("missing") is None
        assert cache.stats()["misses"] == 1

    def test_ttl_expiration(self):
        cache = DNSCache(clock=lambda: 0)
        cache.put("svc-a", ["1.1.1.1"], ttl_sec=10)
        assert cache.resolve("svc-a") == ["1.1.1.1"]
        cache._clock = lambda: 15
        assert cache.resolve("svc-a") is None

    def test_negative_cache(self):
        cache = DNSCache(negative_ttl_sec=60, clock=lambda: 0)
        cache.put_negative("missing-svc")
        assert cache.resolve("missing-svc") == []
        assert cache.stats()["negative_hits"] == 1

    def test_negative_cache_expiration(self):
        cache = DNSCache(negative_ttl_sec=10, clock=lambda: 0)
        cache.put_negative("missing-svc")
        assert cache.resolve("missing-svc") == []
        cache._clock = lambda: 15
        assert cache.resolve("missing-svc") is None

    def test_get_one(self):
        cache = DNSCache(clock=lambda: 0)
        cache.put("svc-a", ["1.1.1.1", "2.2.2.2"])
        assert cache.get_one("svc-a") == "1.1.1.1"
        assert cache.get_one("missing") is None

    def test_invalidate(self):
        cache = DNSCache(clock=lambda: 0)
        cache.put("svc-a", ["1.1.1.1"])
        assert cache.invalidate("svc-a") is True
        assert cache.resolve("svc-a") is None
        assert cache.invalidate("missing") is False

    def test_invalidate_pattern(self):
        cache = DNSCache(clock=lambda: 0)
        cache.put("svc-a.fleet.local", ["1.1.1.1"])
        cache.put("svc-b.fleet.local", ["2.2.2.2"])
        cache.put("other.host", ["3.3.3.3"])
        count = cache.invalidate_pattern(".fleet.local")
        assert count == 2
        assert cache.resolve("svc-a.fleet.local") is None
        assert cache.resolve("other.host") == ["3.3.3.3"]

    def test_clear(self):
        cache = DNSCache(clock=lambda: 0)
        cache.put("a", ["1.1.1.1"])
        cache.put("b", ["2.2.2.2"])
        cache.clear()
        assert cache.stats()["size"] == 0

    def test_hostnames(self):
        cache = DNSCache(clock=lambda: 0)
        cache.put("a", ["1.1.1.1"])
        cache.put("b", ["2.2.2.2"])
        assert sorted(cache.hostnames()) == ["a", "b"]

    def test_ttl(self):
        cache = DNSCache(clock=lambda: 0)
        cache.put("a", ["1.1.1.1"], ttl_sec=100)
        assert cache.ttl("a") == 100
        cache._clock = lambda: 50
        assert cache.ttl("a") == 50

    def test_ttl_expired(self):
        cache = DNSCache(clock=lambda: 0)
        cache.put("a", ["1.1.1.1"], ttl_sec=10)
        cache._clock = lambda: 20
        assert cache.ttl("a") is None

    def test_stats(self):
        cache = DNSCache(clock=lambda: 0)
        cache.put("a", ["1.1.1.1"])
        cache.resolve("a")  # hit
        cache.resolve("b")  # miss
        cache.put_negative("c")
        cache.resolve("c")  # negative hit
        stats = cache.stats()
        assert stats["size"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["negative_hits"] == 1
        assert stats["total_lookups"] == 3

    def test_repr(self):
        cache = DNSCache()
        assert "DNSCache" in repr(cache)
