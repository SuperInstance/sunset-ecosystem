"""Tests for Federated Nexus endpoint validation and registration behaviour."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest
import requests

from nexus.federation import (
    ConnectionRefusedError,
    DEFAULT_NEXUS_IP,
    DEFAULT_NEXUS_PORT,
    FederationEndpoint,
    FederatedNexus,
    HEARTBEAT_PATH,
    InvalidEndpointError,
    RegistrationRecord,
    FEDERATION_PATH,
)


# ═══════════════════════════════════════════════════════════════
# FederationEndpoint validation
# ═══════════════════════════════════════════════════════════════

class TestFederationEndpoint:
    """Endpoint must reject localhost and accept real IPs."""

    def test_url_uses_correct_ip_not_localhost(self):
        """The canonical endpoint must use <BOAT_IP>, not localhost."""
        ep = FederationEndpoint(host=DEFAULT_NEXUS_IP, port=DEFAULT_NEXUS_PORT)
        assert ep.url == f"http://{DEFAULT_NEXUS_IP}:{DEFAULT_NEXUS_PORT}"
        assert "localhost" not in ep.url
        assert "127." not in ep.url

    def test_rejects_literal_localhost(self):
        """Literal 'localhost' must raise InvalidEndpointError."""
        with pytest.raises(InvalidEndpointError) as exc_info:
            FederationEndpoint(host="localhost")
        assert "localhost" in str(exc_info.value).lower()
        assert "loopback" in str(exc_info.value).lower()

    def test_rejects_127_0_0_1(self):
        """127.0.0.1 must raise InvalidEndpointError."""
        with pytest.raises(InvalidEndpointError):
            FederationEndpoint(host="127.0.0.1")

    def test_rejects_empty_host(self):
        """Empty host must raise InvalidEndpointError."""
        with pytest.raises(InvalidEndpointError):
            FederationEndpoint(host="")

    def test_accepts_valid_fleet_ip(self):
        """The Cocapn fleet IP must be accepted without error."""
        ep = FederationEndpoint(host=DEFAULT_NEXUS_IP)
        assert ep.host == DEFAULT_NEXUS_IP

    def test_tls_scheme(self):
        """TLS flag must flip the URL scheme to https."""
        ep = FederationEndpoint(host=DEFAULT_NEXUS_IP, port=4047, tls=True)
        assert ep.url.startswith("https://")

    @patch("socket.getaddrinfo")
    def test_rejects_localhost_alias(self, mock_getaddrinfo):
        """Any hostname that resolves to 127.x.x.x must be rejected."""
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))
        ]
        with pytest.raises(InvalidEndpointError):
            FederationEndpoint(host="nexus.local")


# ═══════════════════════════════════════════════════════════════
# FederatedNexus registration
# ═══════════════════════════════════════════════════════════════

class TestFederatedNexusRegister:
    """Registration heartbeat must target the correct endpoint."""

    def _mock_ok_response(self, **override) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = {
            "node_id": "node-42",
            "hostname": "test-host",
            "capabilities": ["gpu"],
            "metadata": {},
            **override,
        }
        resp.raise_for_status = MagicMock()
        return resp

    @patch("nexus.federation.requests.Session.post")
    def test_registration_sends_to_correct_endpoint(self, mock_post):
        """The registration POST must target the fleet IP, not localhost."""
        mock_post.return_value = self._mock_ok_response()

        ep = FederationEndpoint(host=DEFAULT_NEXUS_IP, port=DEFAULT_NEXUS_PORT)
        nexus = FederatedNexus(endpoint=ep, node_id="node-42")
        record = nexus.register()

        expected_url = f"http://{DEFAULT_NEXUS_IP}:{DEFAULT_NEXUS_PORT}{FEDERATION_PATH}"
        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        assert call_args[0] == expected_url
        assert "localhost" not in call_args[0]
        assert "127." not in call_args[0]
        assert record.node_id == "node-42"

    @patch("nexus.federation.requests.Session.post")
    def test_registration_connection_error(self, mock_post):
        """ConnectionError must be translated to ConnectionRefusedError."""
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")

        ep = FederationEndpoint(host=DEFAULT_NEXUS_IP, port=DEFAULT_NEXUS_PORT)
        nexus = FederatedNexus(endpoint=ep, node_id="node-42")

        with pytest.raises(ConnectionRefusedError) as exc_info:
            nexus.register()
        assert DEFAULT_NEXUS_IP in str(exc_info.value)

    @patch("nexus.federation.requests.Session.post")
    def test_heartbeat_sends_to_correct_endpoint(self, mock_post):
        """The heartbeat POST must target the fleet IP, not localhost."""
        mock_post.return_value = self._mock_ok_response()

        ep = FederationEndpoint(host=DEFAULT_NEXUS_IP, port=DEFAULT_NEXUS_PORT)
        nexus = FederatedNexus(endpoint=ep, node_id="node-42")
        nexus.heartbeat()

        expected_url = f"http://{DEFAULT_NEXUS_IP}:{DEFAULT_NEXUS_PORT}{HEARTBEAT_PATH}"
        mock_post.assert_called_once()
        call_args, _ = mock_post.call_args
        assert call_args[0] == expected_url
        assert "localhost" not in call_args[0]

    def test_from_defaults_uses_fleet_ip(self):
        """Factory method must default to the fleet nexus IP."""
        nexus = FederatedNexus.from_defaults("node-99")
        assert nexus.endpoint.host == DEFAULT_NEXUS_IP
        assert nexus.endpoint.port == DEFAULT_NEXUS_PORT
        assert "localhost" not in nexus.endpoint.url
        assert "127." not in nexus.endpoint.url


# ═══════════════════════════════════════════════════════════════
# RegistrationRecord
# ═══════════════════════════════════════════════════════════════

class TestRegistrationRecord:
    def test_stale_detection(self):
        """A record older than the timeout must report stale."""
        record = RegistrationRecord(
            node_id="node-1", hostname="h1", last_seen=0.0
        )
        assert record.is_stale(timeout_sec=1.0)

    def test_fresh_record(self):
        record = RegistrationRecord(node_id="node-1", hostname="h1")
        assert not record.is_stale(timeout_sec=120.0)
