"""Tests for response_cache.py — Response cache with vary-by and etag.

Run: python3 -m pytest tests/test_response_cache.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.response_cache import ResponseCache


class TestResponseCache:
    def test_create(self):
        cache = ResponseCache(default_ttl_sec=300, clock=lambda: 0)
        assert cache.stats()["size"] == 0

    def test_put_get(self):
        cache = ResponseCache(clock=lambda: 0)
        cache.put("/users", {"body": "users-list"})
        assert cache.get("/users") == {"body": "users-list"}

    def test_get_missing(self):
        cache = ResponseCache(clock=lambda: 0)
        assert cache.get("/missing") is None
        assert cache.stats()["misses"] == 1

    def test_ttl_expiration(self):
        cache = ResponseCache(clock=lambda: 0)
        cache.put("/users", {"body": "users"}, ttl_sec=10)
        assert cache.get("/users") == {"body": "users"}
        cache._clock = lambda: 15
        assert cache.get("/users") is None

    def test_etag(self):
        cache = ResponseCache(clock=lambda: 0)
        cache.put("/users", {"body": "users-list"}, etag='"abc123"')
        response, not_modified = cache.get_with_etag("/users", if_none_match='"abc123"')
        assert response is None
        assert not_modified is True

    def test_etag_mismatch(self):
        cache = ResponseCache(clock=lambda: 0)
        cache.put("/users", {"body": "users-list"})
        response, not_modified = cache.get_with_etag("/users", if_none_match='"old"')
        assert response == {"body": "users-list"}
        assert not_modified is False

    def test_invalidate(self):
        cache = ResponseCache(clock=lambda: 0)
        cache.put("/users", {"body": "users"})
        assert cache.invalidate("/users") is True
        assert cache.get("/users") is None
        assert cache.invalidate("/missing") is False

    def test_clear(self):
        cache = ResponseCache(clock=lambda: 0)
        cache.put("/a", {"body": "a"})
        cache.put("/b", {"body": "b"})
        cache.clear()
        assert cache.stats()["size"] == 0

    def test_vary_headers(self):
        cache = ResponseCache(clock=lambda: 0)
        cache.put("/users", {"body": "json"}, vary_headers=["accept"])
        # Simple cache doesn't differentiate by vary values yet
        assert cache.get("/users", headers={"accept": "json"}) == {"body": "json"}

    def test_hit_rate(self):
        cache = ResponseCache(clock=lambda: 0)
        cache.put("/a", {"body": "a"})
        cache.get("/a")  # hit
        cache.get("/b")  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_repr(self):
        cache = ResponseCache()
        assert "ResponseCache" in repr(cache)
