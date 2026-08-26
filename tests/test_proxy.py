"""Tests for proxy.py — Reverse proxy with load balancing.

Run: python3 -m pytest tests/test_proxy.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.proxy import ReverseProxy, ProxyError, Backend


class TestReverseProxy:
    def test_create(self):
        proxy = ReverseProxy()
        assert proxy.pick() is None

    def test_add_remove_backend(self):
        proxy = ReverseProxy()
        proxy.add_backend("http://node-1:8080")
        assert len(proxy.backends()) == 1
        assert proxy.remove_backend("http://node-1:8080") is True
        assert proxy.remove_backend("missing") is False

    def test_round_robin(self):
        proxy = ReverseProxy(strategy="round_robin")
        proxy.add_backend("a")
        proxy.add_backend("b")
        assert proxy.pick() == "a"
        assert proxy.pick() == "b"
        assert proxy.pick() == "a"

    def test_random(self):
        proxy = ReverseProxy(strategy="random")
        proxy.add_backend("a")
        proxy.add_backend("b")
        result = proxy.pick()
        assert result in ("a", "b")

    def test_least_connections(self):
        proxy = ReverseProxy(strategy="least_connections")
        proxy.add_backend("a")
        proxy.add_backend("b")
        a = proxy.pick()
        assert a == "a"
        proxy.pick()  # b
        proxy.pick()  # a (tie, first)
        proxy.release("a")
        assert proxy.pick() == "a"

    def test_no_healthy_backends(self):
        proxy = ReverseProxy()
        proxy.add_backend("a")
        proxy._backends["a"].healthy = False
        assert proxy.pick() is None

    def test_health_check_mark_failed(self):
        proxy = ReverseProxy(check_fn=lambda url: False)
        proxy.add_backend("a", check_interval=0.0, max_failures=1)
        proxy.health_check()
        assert proxy._backends["a"].healthy is False

    def test_health_check_mark_healthy(self):
        proxy = ReverseProxy(check_fn=lambda url: True)
        proxy.add_backend("a", check_interval=0.0)
        proxy._backends["a"].mark_failed()
        proxy._backends["a"].healthy = False
        proxy.health_check()
        assert proxy._backends["a"].healthy is True

    def test_release(self):
        proxy = ReverseProxy()
        proxy.add_backend("a")
        proxy.pick()
        assert proxy._backends["a"].connection_count == 1
        proxy.release("a")
        assert proxy._backends["a"].connection_count == 0

    def test_stats(self):
        proxy = ReverseProxy()
        proxy.add_backend("a")
        proxy.pick()
        proxy.health_check()
        stats = proxy.stats()
        assert stats["requests"] == 1

    def test_repr(self):
        proxy = ReverseProxy()
        proxy.add_backend("a")
        assert "round_robin" in repr(proxy)
        assert "1/1" in repr(proxy)

    def test_unknown_strategy(self):
        proxy = ReverseProxy(strategy="bogus")
        proxy.add_backend("a")
        with pytest.raises(ProxyError):
            proxy.pick()
