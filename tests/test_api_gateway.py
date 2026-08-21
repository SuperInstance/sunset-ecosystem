"""Tests for api_gateway.py — API gateway with routing and middleware.

Run: python3 -m pytest tests/test_api_gateway.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.api_gateway import APIGateway


class TestAPIGateway:
    def test_create(self):
        gw = APIGateway()
        assert gw.stats()["routes"] == 0
        assert gw.stats()["middleware"] == 0

    def test_route(self):
        gw = APIGateway()
        gw.route("/users", "users-service")
        assert "/users" in gw.routes()
        assert gw.get_target("/users") == "users-service"

    def test_remove_route(self):
        gw = APIGateway()
        gw.route("/users", "users-service")
        assert gw.remove_route("/users") is True
        assert gw.remove_route("/missing") is False

    def test_prefix_routing(self):
        gw = APIGateway()
        gw.route("/api/v1", "api-service")
        assert gw.get_target("/api/v1/users") == "api-service"

    def test_exact_over_prefix(self):
        gw = APIGateway()
        gw.route("/api", "api-service")
        gw.route("/api/v1", "v1-service")
        assert gw.get_target("/api/v1/users") == "v1-service"

    def test_middleware(self):
        gw = APIGateway()
        gw.add_middleware("auth", lambda req: req.get("token") == "valid")
        assert "auth" in gw.middleware_names()

    def test_middleware_priority(self):
        gw = APIGateway()
        order = []
        gw.add_middleware("b", lambda req: order.append("b") or True, priority=1)
        gw.add_middleware("a", lambda req: order.append("a") or True, priority=0)
        gw.process({"path": "/"})
        assert order == ["a", "b"]

    def test_middleware_rejection(self):
        gw = APIGateway()
        gw.add_middleware("auth", lambda req: False)
        result = gw.process({"path": "/users"})
        assert result["status"] == "rejected"
        assert result["reason"] == "middleware:auth"

    def test_remove_middleware(self):
        gw = APIGateway()
        gw.add_middleware("auth", lambda req: True)
        assert gw.remove_middleware("auth") is True
        assert gw.remove_middleware("missing") is False

    def test_process_no_route(self):
        gw = APIGateway()
        result = gw.process({"path": "/missing"})
        assert result["status"] == "not_found"

    def test_process_with_handler(self):
        gw = APIGateway()
        gw.route("/users", "users")
        gw.register_handler("users", lambda req: {"users": [1, 2, 3]})
        result = gw.process({"path": "/users"})
        assert result["status"] == "ok"
        assert result["result"] == {"users": [1, 2, 3]}

    def test_process_handler_error(self):
        gw = APIGateway()
        gw.route("/users", "users")
        gw.register_handler("users", lambda req: 1 / 0)
        result = gw.process({"path": "/users"})
        assert result["status"] == "error"

    def test_stats(self):
        gw = APIGateway()
        gw.route("/a", "svc-a")
        gw.add_middleware("auth", lambda req: True)
        gw.process({"path": "/a"})
        stats = gw.stats()
        assert stats["routes"] == 1
        assert stats["middleware"] == 1
        assert stats["requests"] == 1

    def test_repr(self):
        gw = APIGateway()
        assert "APIGateway" in repr(gw)
