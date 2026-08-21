"""Tests for request_proxy.py — Request proxy with backends and retry.

Run: python3 -m pytest tests/test_request_proxy.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.request_proxy import RequestProxy


class TestRequestProxy:
    def test_create(self):
        proxy = RequestProxy()
        assert proxy.stats()["services"] == 0

    def test_add_backend(self):
        proxy = RequestProxy()
        proxy.add_backend("api", "http://api-1:8080")
        assert proxy.stats()["backends"] == 1

    def test_remove_backend(self):
        proxy = RequestProxy()
        proxy.add_backend("api", "http://api-1:8080")
        assert proxy.remove_backend("api", "http://api-1:8080") is True
        assert proxy.remove_backend("api", "missing") is False

    def test_forward(self):
        proxy = RequestProxy()
        proxy.add_backend("api", "http://api-1:8080")
        result = proxy.forward("api", {"path": "/users"})
        assert result is not None
        assert result["backend"] == "http://api-1:8080"

    def test_forward_unhealthy(self):
        proxy = RequestProxy()
        proxy.add_backend("api", "http://api-1:8080")
        proxy.set_health("http://api-1:8080", False)
        result = proxy.forward("api", {"path": "/users"})
        assert result is None

    def test_forward_round_robin(self):
        proxy = RequestProxy()
        proxy.add_backend("api", "http://api-1:8080")
        proxy.add_backend("api", "http://api-2:8080")
        r1 = proxy.forward("api", {"path": "/a"})
        r2 = proxy.forward("api", {"path": "/b"})
        assert r1["backend"] != r2["backend"]

    def test_forward_no_backends(self):
        proxy = RequestProxy()
        result = proxy.forward("missing", {"path": "/users"})
        assert result is None

    def test_repr(self):
        proxy = RequestProxy()
        assert "RequestProxy" in repr(proxy)
