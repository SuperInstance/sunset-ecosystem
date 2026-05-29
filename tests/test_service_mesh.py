"""Tests for service_mesh.py — Service mesh with routing and health.

Run: python3 -m pytest tests/test_service_mesh.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.service_mesh import ServiceMesh


class TestServiceMesh:
    def test_create(self):
        mesh = ServiceMesh()
        assert mesh.stats()["services"] == 0

    def test_register(self):
        mesh = ServiceMesh()
        mesh.register("users", ["http://user-1:8080"])
        assert mesh.stats()["services"] == 1
        assert mesh.stats()["endpoints"] == 1

    def test_deregister(self):
        mesh = ServiceMesh()
        mesh.register("users", ["http://user-1:8080"])
        assert mesh.deregister("users") is True
        assert mesh.deregister("missing") is False

    def test_add_endpoint(self):
        mesh = ServiceMesh()
        mesh.register("users", ["http://user-1:8080"])
        mesh.add_endpoint("users", "http://user-2:8080")
        assert mesh.stats()["endpoints"] == 2

    def test_remove_endpoint(self):
        mesh = ServiceMesh()
        mesh.register("users", ["http://user-1:8080", "http://user-2:8080"])
        assert mesh.remove_endpoint("users", "http://user-1:8080") is True
        assert mesh.remove_endpoint("users", "missing") is False

    def test_set_health(self):
        mesh = ServiceMesh()
        mesh.register("users", ["http://user-1:8080"])
        mesh.set_health("http://user-1:8080", False)
        assert mesh.is_healthy("http://user-1:8080") is False

    def test_call(self):
        mesh = ServiceMesh()
        mesh.register("users", ["http://user-1:8080"])
        result = mesh.call("users", "/get/123")
        assert result is not None
        assert result["endpoint"] == "http://user-1:8080"

    def test_call_unhealthy(self):
        mesh = ServiceMesh()
        mesh.register("users", ["http://user-1:8080"])
        mesh.set_health("http://user-1:8080", False)
        result = mesh.call("users", "/get/123")
        assert result is None

    def test_call_round_robin(self):
        mesh = ServiceMesh()
        mesh.register("users", ["http://user-1:8080", "http://user-2:8080"])
        r1 = mesh.call("users", "/a")
        r2 = mesh.call("users", "/b")
        assert r1["endpoint"] != r2["endpoint"]

    def test_call_missing_service(self):
        mesh = ServiceMesh()
        result = mesh.call("missing", "/path")
        assert result is None

    def test_services(self):
        mesh = ServiceMesh()
        mesh.register("a", ["http://a:8080"])
        mesh.register("b", ["http://b:8080"])
        assert sorted(mesh.services()) == ["a", "b"]

    def test_endpoints(self):
        mesh = ServiceMesh()
        mesh.register("users", ["http://user-1:8080", "http://user-2:8080"])
        assert len(mesh.endpoints("users")) == 2

    def test_stats(self):
        mesh = ServiceMesh()
        mesh.register("users", ["http://user-1:8080", "http://user-2:8080"])
        stats = mesh.stats()
        assert stats["services"] == 1
        assert stats["endpoints"] == 2
        assert stats["healthy"] == 2

    def test_repr(self):
        mesh = ServiceMesh()
        assert "ServiceMesh" in repr(mesh)
