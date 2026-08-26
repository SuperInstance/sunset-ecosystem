"""Tests for the A2A HTTP server and task dispatch."""

import json
import urllib.request

import pytest

from a2a.server import A2AServer
from a2a.handlers import (
    handle_metronome_task,
    handle_breeder_task,
    handle_grid_task,
    handle_flux_task,
)


# ── Fixtures ──


@pytest.fixture
def server(tmp_path_factory):
    """Spin up an A2AServer on an ephemeral port, yield it, then stop."""
    srv = A2AServer(port=0)  # port=0 lets OS assign ephemeral port
    # We need to know the actual port after bind; patch to capture it
    srv.start()
    # Extract the actual bound port
    actual_port = srv._server.server_address[1]
    srv.port = actual_port
    yield srv
    srv.stop()


@pytest.fixture
def registered_server(server):
    """Server with all 4 fleet agents registered."""
    server.register_agent("metronome", handle_metronome_task)
    server.register_agent("breeder", handle_breeder_task)
    server.register_agent("grid", handle_grid_task)
    server.register_agent("flux", handle_flux_task)
    return server


def _post_json(url, payload):
    """Helper: POST JSON and return (status_code, response_dict)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


def _get_json(url):
    """Helper: GET JSON and return (status_code, response_dict)."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


# ── Tests ──


class TestAgentCards:
    """Static agent card serving."""

    def test_agent_card_served(self, registered_server):
        """GET /.well-known/agent-metronome.json returns the JSON card."""
        url = f"{registered_server.url}/.well-known/agent-metronome.json"
        status, body = _get_json(url)
        assert status == 200
        assert body["name"] == "fleet-metronome"
        assert "capabilities" in body
        assert "tick" in body["capabilities"]

    def test_unknown_agent_card_404(self, registered_server):
        """Requesting an unregistered agent card returns 404."""
        url = f"{registered_server.url}/.well-known/agent-unknown.json"
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _get_json(url)
        assert exc_info.value.code == 404


class TestTaskDispatch:
    """Task dispatch via POST /a2a/tasks/send."""

    def test_task_dispatch_metronome_tick(self, registered_server):
        """POST tick task to metronome handler."""
        url = f"{registered_server.url}/a2a/tasks/send"
        payload = {
            "id": "task-001",
            "agent": "metronome",
            "type": "tick",
            "input": {
                "signal": [0.1, -0.3, 0.7] + [0.0] * 61,
                "force": False,
            },
        }
        status, body = _post_json(url, payload)
        assert status == 200
        assert body["status"] == "ok"
        assert body["result"]["beat_number"] == 1423
        assert "fired_rooms" in body["result"]

    def test_task_dispatch_breeder_get_state(self, registered_server):
        """POST get_state task to breeder handler."""
        url = f"{registered_server.url}/a2a/tasks/send"
        payload = {
            "id": "task-002",
            "agent": "breeder",
            "type": "get_state",
            "input": {},
        }
        status, body = _post_json(url, payload)
        assert status == 200
        assert body["status"] == "ok"
        assert "agents" in body["result"]
        assert "phase_counts" in body["result"]

    def test_invalid_task_returns_error(self, registered_server):
        """POST unknown task type returns error status."""
        url = f"{registered_server.url}/a2a/tasks/send"
        payload = {
            "id": "task-003",
            "agent": "metronome",
            "type": "dance_party",
            "input": {},
        }
        status, body = _post_json(url, payload)
        assert status == 200  # HTTP OK, but business-level error
        assert body["status"] == "error"
        assert "Unknown task type" in body["result"]["message"]

    def test_missing_agent_field_inference(self, registered_server):
        """If 'agent' is missing, infer from 'type' field."""
        url = f"{registered_server.url}/a2a/tasks/send"
        payload = {
            "id": "task-004",
            "type": "get_status",  # known metronome task
        }
        status, body = _post_json(url, payload)
        assert status == 200
        assert body["status"] == "ok"
        assert "healthy" in body["result"]

    def test_unregistered_agent_returns_404(self, server):
        """Task for an agent with no handler registered returns 404."""
        url = f"{server.url}/a2a/tasks/send"
        payload = {
            "id": "task-005",
            "agent": "metronome",
            "type": "tick",
            "input": {},
        }
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post_json(url, payload)
        assert exc_info.value.code == 404

    def test_missing_type_returns_error(self, registered_server):
        """Payload without 'type' and without 'agent' returns 400."""
        url = f"{registered_server.url}/a2a/tasks/send"
        payload = {
            "id": "task-006",
            "input": {},
        }
        # With no agent and no inferable type, the server returns 400.
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post_json(url, payload)
        assert exc_info.value.code == 400

    def test_flux_check_constraints(self, registered_server):
        """FLUX handler returns constraint check with certificates."""
        url = f"{registered_server.url}/a2a/tasks/send"
        payload = {
            "id": "task-007",
            "agent": "flux",
            "type": "check_constraints",
            "input": {
                "values": [[0.5, -0.2, 0.1], [9.5, -8.2, 7.1]],
                "preset": "neural_bounds",
                "domain": "neural",
                "generate_certificate": True,
            },
        }
        status, body = _post_json(url, payload)
        assert status == 200
        assert body["status"] == "ok"
        assert "pass" in body["result"]
        assert "certificates" in body["result"]
        assert len(body["result"]["certificates"]) == 2

    def test_grid_tick(self, registered_server):
        """Grid handler tick returns fired room IDs."""
        url = f"{registered_server.url}/a2a/tasks/send"
        payload = {
            "id": "task-008",
            "agent": "grid",
            "type": "tick",
            "input": {
                "signal": [0.1] * 64,
                "room_ids": [0, 1, 2],
            },
        }
        status, body = _post_json(url, payload)
        assert status == 200
        assert body["status"] == "ok"
        assert body["result"]["ids"] == [0, 1, 2]

    def test_emergency_stop_requires_reason(self, registered_server):
        """Breeder emergency_stop without 'reason' returns validation error."""
        url = f"{registered_server.url}/a2a/tasks/send"
        payload = {
            "id": "task-009",
            "agent": "breeder",
            "type": "emergency_stop",
            "input": {},
        }
        status, body = _post_json(url, payload)
        assert status == 200
        assert body["status"] == "error"
        assert "Missing required keys" in body["result"]["message"]

    def test_server_stop_gracefully(self, registered_server):
        """Calling stop() shuts down the server without hanging."""
        srv = registered_server
        # Ensure it's alive
        url = f"{srv.url}/a2a/tasks/send"
        payload = {"id": "ping", "agent": "metronome", "type": "get_status"}
        status, _ = _post_json(url, payload)
        assert status == 200

        srv.stop()
        # After stop, connections should fail
        with pytest.raises(urllib.error.URLError):
            _post_json(url, payload)
