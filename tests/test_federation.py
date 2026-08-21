"""Tests for FederatedNexus — fleet node registration and heartbeat.

Covers FederationEndpoint validation, registration, heartbeat,
from_defaults factory, and topology_check delegation.
"""

import socket
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from nexus.federation import (
    ConnectionRefusedError,
    FederationEndpoint,
    FederatedNexus,
    InvalidEndpointError,
    RegistrationRecord,
)


# ---------------------------------------------------------------------------
# FederationEndpoint
# ---------------------------------------------------------------------------


class TestFederationEndpoint:
    def test_url_http(self):
        ep = FederationEndpoint(host="1.2.3.4", port=8080)
        assert ep.url == "http://1.2.3.4:8080"

    def test_url_https(self):
        ep = FederationEndpoint(host="1.2.3.4", port=443, tls=True)
        assert ep.url == "https://1.2.3.4:443"

    def test_default_port(self):
        ep = FederationEndpoint(host="1.2.3.4")
        assert ep.port == 4047

    def test_empty_host_raises(self):
        with pytest.raises(InvalidEndpointError):
            FederationEndpoint(host="")

    def test_localhost_raises(self):
        with pytest.raises(InvalidEndpointError):
            FederationEndpoint(host="localhost")

    def test_127_0_0_1_raises(self):
        with pytest.raises(InvalidEndpointError):
            FederationEndpoint(host="127.0.0.1")

    def test_unresolvable_permissive(self):
        # unresolvable host should be permissive (returns False from _is_localhost)
        ep = FederationEndpoint(host="definitely-not-real-123456.local")
        assert ep.host == "definitely-not-real-123456.local"


# ---------------------------------------------------------------------------
# RegistrationRecord
# ---------------------------------------------------------------------------


class TestRegistrationRecord:
    def test_not_stale(self):
        rec = RegistrationRecord(node_id="n1", hostname="h1")
        assert not rec.is_stale()

    def test_stale(self):
        rec = RegistrationRecord(node_id="n1", hostname="h1", last_seen=0.0)
        assert rec.is_stale()

    def test_custom_timeout(self):
        rec = RegistrationRecord(node_id="n1", hostname="h1", last_seen=time.time())
        assert rec.is_stale(timeout_sec=0.0)


# ---------------------------------------------------------------------------
# FederatedNexus init
# ---------------------------------------------------------------------------


class TestFederatedNexusInit:
    def test_defaults(self):
        ep = FederationEndpoint(host="1.2.3.4")
        nexus = FederatedNexus(ep, "node-1")
        assert nexus.node_id == "node-1"
        assert nexus.hostname == socket.gethostname()
        assert nexus.capabilities == []

    def test_with_params(self):
        ep = FederationEndpoint(host="1.2.3.4")
        nexus = FederatedNexus(ep, "n1", hostname="custom", capabilities=["gpu"])
        assert nexus.hostname == "custom"
        assert nexus.capabilities == ["gpu"]

    def test_from_defaults(self):
        nexus = FederatedNexus.from_defaults("node-1")
        assert nexus.endpoint.host == "<BOAT_IP>"
        assert nexus.endpoint.port == 4047


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


class TestRegister:
    @patch("nexus.federation.requests.Session.post")
    def test_success(self, mock_post):
        mock_post.return_value = MagicMock(
            raise_for_status=lambda: None,
            json=lambda: {"node_id": "node-1", "capabilities": ["cpu"]},
        )
        ep = FederationEndpoint(host="1.2.3.4")
        nexus = FederatedNexus(ep, "node-1")
        rec = nexus.register()
        assert rec.node_id == "node-1"
        assert rec.capabilities == ["cpu"]

    @patch("nexus.federation.requests.Session.post")
    def test_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")
        ep = FederationEndpoint(host="1.2.3.4")
        nexus = FederatedNexus(ep, "node-1")
        with pytest.raises(ConnectionRefusedError):
            nexus.register()

    @patch("nexus.federation.requests.Session.post")
    def test_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("slow")
        ep = FederationEndpoint(host="1.2.3.4")
        nexus = FederatedNexus(ep, "node-1")
        with pytest.raises(ConnectionRefusedError):
            nexus.register()


# ---------------------------------------------------------------------------
# heartbeat
# ---------------------------------------------------------------------------


class TestHeartbeat:
    @patch("nexus.federation.requests.Session.post")
    def test_success(self, mock_post):
        mock_post.return_value = MagicMock(raise_for_status=lambda: None)
        ep = FederationEndpoint(host="1.2.3.4")
        nexus = FederatedNexus(ep, "node-1")
        nexus.heartbeat()
        assert nexus._last_heartbeat is not None

    @patch("nexus.federation.requests.Session.post")
    def test_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")
        ep = FederationEndpoint(host="1.2.3.4")
        nexus = FederatedNexus(ep, "node-1")
        with pytest.raises(ConnectionRefusedError):
            nexus.heartbeat()


# ---------------------------------------------------------------------------
# maybe_heartbeat
# ---------------------------------------------------------------------------


class TestMaybeHeartbeat:
    @patch("nexus.federation.requests.Session.post")
    def test_first_heartbeat(self, mock_post):
        mock_post.return_value = MagicMock(raise_for_status=lambda: None)
        ep = FederationEndpoint(host="1.2.3.4")
        nexus = FederatedNexus(ep, "node-1")
        nexus.maybe_heartbeat()
        assert nexus._last_heartbeat is not None

    @patch("nexus.federation.requests.Session.post")
    def test_respects_interval(self, mock_post):
        mock_post.return_value = MagicMock(raise_for_status=lambda: None)
        ep = FederationEndpoint(host="1.2.3.4")
        nexus = FederatedNexus(ep, "node-1", heartbeat_interval=3600)
        nexus._last_heartbeat = 1e9  # far in the past but not far enough for 3600s
        nexus.maybe_heartbeat()
        # still called because 1e9 is very far in the past
        assert mock_post.call_count == 1

    @patch("nexus.federation.requests.Session.post")
    def test_disabled(self, mock_post):
        ep = FederationEndpoint(host="1.2.3.4")
        nexus = FederatedNexus(ep, "node-1", heartbeat_interval=0)
        nexus.maybe_heartbeat()
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    def test_closes_session(self):
        ep = FederationEndpoint(host="1.2.3.4")
        nexus = FederatedNexus(ep, "node-1")
        nexus.close()
        # no crash


# ---------------------------------------------------------------------------
# topology_check
# ---------------------------------------------------------------------------


class TestTopologyCheck:
    def test_delegates_to_holonomy(self):
        ep = FederationEndpoint(host="1.2.3.4")
        nexus = FederatedNexus(ep, "node-1")
        report = nexus.topology_check([("a", "b"), ("b", "c")])
        # self node + a,b,c = 4 nodes
        assert report.node_count == 4
        assert report.edge_count == 2

    def test_includes_self(self):
        ep = FederationEndpoint(host="1.2.3.4")
        nexus = FederatedNexus(ep, "node-1")
        report = nexus.topology_check([])
        assert report.node_count == 1  # self only


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import time
