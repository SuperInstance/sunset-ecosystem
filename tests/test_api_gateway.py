"""Tests for api_gateway.py — Unified API gateway.

Run: python3 -m pytest tests/test_api_gateway.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.auth import FleetAuth
from fleet.rate_limiter import TokenBucket
from nexus.api_gateway import APIGateway, Request, Response


class TestAPIGateway:
    def test_create(self):
        gw = APIGateway()
        assert gw._routes == {}

    def test_add_route(self):
        gw = APIGateway()
        gw.add_route("/test", lambda r: Response(200, body="ok"), methods=["GET"])
        assert "/test" in gw._routes

    def test_handle_404(self):
        gw = APIGateway()
        resp = gw.handle_request(Request(path="/missing"))
        assert resp.status == 404

    def test_handle_405(self):
        gw = APIGateway()
        gw.add_route("/test", lambda r: Response(200), methods=["POST"])
        resp = gw.handle_request(Request(path="/test", method="GET"))
        assert resp.status == 405

    def test_handle_success(self):
        gw = APIGateway()
        gw.add_route("/test", lambda r: Response(200, body={"ok": True}), methods=["GET"])
        resp = gw.handle_request(Request(path="/test"))
        assert resp.status == 200
        assert resp.body == {"ok": True}

    def test_handle_with_auth(self):
        auth = FleetAuth(secret_key="test")
        gw = APIGateway(auth=auth)
        gw.add_route("/protected", lambda r: Response(200, body={"user": r.client_id}), methods=["GET"], require_auth=True)

        # No auth
        resp = gw.handle_request(Request(path="/protected"))
        assert resp.status == 401

        # Valid auth
        token = auth.create_token("user-1", roles=["viewer"])
        resp = gw.handle_request(Request(
            path="/protected",
            headers={"Authorization": f"Bearer {token}"},
        ))
        assert resp.status == 200
        assert resp.body["user"] == "user-1"

    def test_handle_invalid_auth(self):
        auth = FleetAuth(secret_key="test")
        gw = APIGateway(auth=auth)
        gw.add_route("/protected", lambda r: Response(200), methods=["GET"], require_auth=True)
        resp = gw.handle_request(Request(
            path="/protected",
            headers={"Authorization": "Bearer bad-token"},
        ))
        assert resp.status == 401

    def test_rate_limiting(self):
        rl = TokenBucket(rate=1.0, burst=1)
        gw = APIGateway(rate_limiter=rl)
        gw.add_route("/api", lambda r: Response(200), methods=["GET"], rate_limit=True)
        resp = gw.handle_request(Request(path="/api", client_id="a"))
        assert resp.status == 200
        # Second request should be rate limited
        resp = gw.handle_request(Request(path="/api", client_id="a"))
        assert resp.status == 429

    def test_rate_limiting_disabled(self):
        rl = TokenBucket(rate=1.0, burst=1)
        gw = APIGateway(rate_limiter=rl)
        gw.add_route("/api", lambda r: Response(200), methods=["GET"], rate_limit=False)
        resp1 = gw.handle_request(Request(path="/api", client_id="a"))
        resp2 = gw.handle_request(Request(path="/api", client_id="a"))
        assert resp1.status == 200
        assert resp2.status == 200

    def test_handler_error(self):
        gw = APIGateway()
        gw.add_route("/error", lambda r: (_ for _ in ()).throw(ValueError("boom")), methods=["GET"])
        resp = gw.handle_request(Request(path="/error"))
        assert resp.status == 500
        assert "boom" in resp.body["error"]

    def test_middleware(self):
        gw = APIGateway()
        gw.add_middleware(lambda r: r if r.path != "/block" else Response(403, body={"error": "blocked"}))
        gw.add_route("/block", lambda r: Response(200), methods=["GET"])
        gw.add_route("/allow", lambda r: Response(200), methods=["GET"])
        resp1 = gw.handle_request(Request(path="/block"))
        resp2 = gw.handle_request(Request(path="/allow"))
        assert resp1.status == 403
        assert resp2.status == 200

    def test_stats(self):
        gw = APIGateway()
        gw.add_route("/test", lambda r: Response(200), methods=["GET"])
        gw.handle_request(Request(path="/test"))
        gw.handle_request(Request(path="/missing"))
        stats = gw.stats()
        assert stats["requests"] == 2
        assert stats["errors"] == 1
        assert stats["error_rate"] == 0.5

    def test_repr(self):
        gw = APIGateway()
        assert "APIGateway" in repr(gw)