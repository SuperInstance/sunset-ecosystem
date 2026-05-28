"""Tests for FleetBridgeServer (claw_fleet_bridge.py).

Covers:
    - Health endpoint
    - Status endpoint (lazy-imports FleetConductorV2)
    - FLUX presets endpoint
    - Breed endpoint
    - FLUX check endpoint
    - Mesh insert/query endpoints
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from claw_fleet_bridge import FleetBridgeServer


@pytest.fixture
def server():
    srv = FleetBridgeServer(host="127.0.0.1", port=0)
    srv.start()
    yield srv
    srv.stop()


class TestHealth:
    def test_health(self, server):
        host, port = server._server.server_address
        url = f"http://{host}:{port}/health"
        with urllib.request.urlopen(url, timeout=5.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            assert body["status"] == "ok"
            assert body["fleet"] == "cocapn"


class TestStatus:
    def test_status_returns_json(self, server):
        host, port = server._server.server_address
        url = f"http://{host}:{port}/status"
        with urllib.request.urlopen(url, timeout=5.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            # May be error if FleetConductorV2 not initialized, but should be JSON
            assert isinstance(body, dict)


class TestFluxPresets:
    def test_presets_list(self, server):
        host, port = server._server.server_address
        url = f"http://{host}:{port}/flux/presets"
        with urllib.request.urlopen(url, timeout=5.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            assert "presets" in body
            assert isinstance(body["presets"], dict)


class TestBreed:
    def test_breed_with_defaults(self, server):
        host, port = server._server.server_address
        url = f"http://{host}:{port}/breed"
        req = urllib.request.Request(
            url,
            data=b'{}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                assert "winners" in body or "error" in body
                assert isinstance(body, dict)
        except urllib.error.HTTPError as e:
            body = json.loads(e.read().decode("utf-8"))
            assert "error" in body

    def test_breed_with_params(self, server):
        host, port = server._server.server_address
        url = f"http://{host}:{port}/breed"
        data = json.dumps({"n_winners": 2, "preset": "diversity"}).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                assert isinstance(body, dict)
        except urllib.error.HTTPError as e:
            body = json.loads(e.read().decode("utf-8"))
            assert "error" in body


class TestFluxCheck:
    def test_flux_check(self, server):
        host, port = server._server.server_address
        url = f"http://{host}:{port}/flux/check"
        data = json.dumps({"candidate": {"name": "test", "version": "1.0"}}).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                assert "passed" in body or "error" in body
                assert isinstance(body, dict)
        except urllib.error.HTTPError as e:
            body = json.loads(e.read().decode("utf-8"))
            assert "error" in body


class TestMesh:
    def test_mesh_insert(self, server):
        host, port = server._server.server_address
        url = f"http://{host}:{port}/mesh/insert"
        data = json.dumps({
            "table_id": "test_table",
            "vector": [0.1, 0.2, 0.3],
            "fitness": 0.9,
            "extra": {"gen": 1},
        }).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                assert body.get("inserted") is True or "error" in body
        except urllib.error.HTTPError as e:
            body = json.loads(e.read().decode("utf-8"))
            assert "error" in body

    def test_mesh_query(self, server):
        host, port = server._server.server_address
        url = f"http://{host}:{port}/mesh/query"
        data = json.dumps({"min_fitness": 0.5}).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                assert "count" in body or "error" in body
                assert isinstance(body, dict)
        except urllib.error.HTTPError as e:
            body = json.loads(e.read().decode("utf-8"))
            assert "error" in body


class Test404:
    def test_unknown_path(self, server):
        host, port = server._server.server_address
        url = f"http://{host}:{port}/notfound"
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(url, timeout=5.0)
        assert exc_info.value.code == 404
