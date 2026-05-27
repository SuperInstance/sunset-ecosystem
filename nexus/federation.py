"""Federated Nexus — Fleet-wide agent registration and heartbeat.

This module handles registration of fleet nodes with a central Federated
Nexus so that agents can discover peers, propagate seed-bank archives,
and coordinate sunset events across the swarm.

Previously this code lived on Oracle1 (port 4047) and hardcoded
``localhost`` as the federation endpoint.  That breaks any
registration from a remote node because ``localhost`` resolves to the
loopback interface of the caller, not the nexus host.

The canonical nexus IP for the Cocapn fleet is **147.224.38.131**.
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from nexus.holonomy_bridge import HolonomyBridge

logger = logging.getLogger(__name__)

# ── constants ───────────────────────────────────────────────────

import os

# Default nexus IP — override with NEXUS_IP env var for non-fleet deployments
DEFAULT_NEXUS_IP: str = os.getenv("NEXUS_IP", "147.224.38.131")
DEFAULT_NEXUS_PORT: int = int(os.getenv("NEXUS_PORT", "4047"))
DEFAULT_NEXUS_PORT: int = 4047
FEDERATION_PATH: str = "/v1/federation/register"
HEARTBEAT_PATH: str = "/v1/federation/heartbeat"
HEARTBEAT_INTERVAL_SEC: int = 30


class NexusError(Exception):
    """Base exception for federation failures."""


class ConnectionRefusedError(NexusError):  # noqa: A001
    """Raised when the nexus refuses or cannot accept a connection."""


class InvalidEndpointError(NexusError):
    """Raised when the endpoint contains a forbidden address such as localhost."""


# ── data structures ─────────────────────────────────────────────

@dataclass(slots=True)
class FederationEndpoint:
    """An HTTP endpoint for the Federated Nexus."""

    host: str
    port: int = DEFAULT_NEXUS_PORT
    tls: bool = False

    def __post_init__(self) -> None:
        if not self.host or self.host.strip() == "":
            raise InvalidEndpointError("Endpoint host must not be empty.")
        # Reject localhost / 127.0.0.1 / ::1 — they break remote registration
        if self._is_localhost(self.host):
            raise InvalidEndpointError(
                f"Endpoint host '{self.host}' resolves to loopback. "
                f"Use the external IP (e.g. {DEFAULT_NEXUS_IP}) for fleet-wide registration."
            )

    @property
    def url(self) -> str:
        scheme = "https" if self.tls else "http"
        return f"{scheme}://{self.host}:{self.port}"

    @staticmethod
    def _is_localhost(host: str) -> bool:
        """Return True if *host* resolves to a loopback address."""
        try:
            addrinfo = socket.getaddrinfo(host, None)
            for _, _, _, _, sockaddr in addrinfo:
                ip = sockaddr[0]
                if ip.startswith("127.") or ip == "::1":
                    return True
        except socket.gaierror:
            # If we cannot resolve, be permissive; the network layer will fail later.
            return False
        return False


@dataclass(slots=True)
class RegistrationRecord:
    """A single fleet node's registration record."""

    node_id: str
    hostname: str
    capabilities: list[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_stale(self, timeout_sec: float = 120.0) -> bool:
        return (time.time() - self.last_seen) > timeout_sec


# ── core class ──────────────────────────────────────────────────

class FederatedNexus:
    """Client for registering and heart-beating with the fleet nexus.

    Parameters
    ----------
    endpoint:
        A :class:`FederationEndpoint` describing where the nexus lives.
    node_id:
        Unique identifier for this fleet node.
    heartbeat_interval:
        Seconds between automatic heartbeats (0 disables auto-heartbeat).
    """

    def __init__(
        self,
        endpoint: FederationEndpoint,
        node_id: str,
        hostname: str | None = None,
        capabilities: list[str] | None = None,
        heartbeat_interval: int = HEARTBEAT_INTERVAL_SEC,
    ):
        self.endpoint = endpoint
        self.node_id = node_id
        self.hostname = hostname or socket.gethostname()
        self.capabilities = capabilities or []
        self._heartbeat_interval = heartbeat_interval
        self._last_heartbeat: float | None = None
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ── public API ────────────────────────────────────────────

    def register(self, metadata: dict[str, Any] | None = None) -> RegistrationRecord:
        """Register this node with the federated nexus.

        Returns
        -------
        The server's echoed :class:`RegistrationRecord`.

        Raises
        ------
        ConnectionRefusedError
            If the nexus cannot be reached.
        """
        payload = {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "capabilities": self.capabilities,
            "metadata": metadata or {},
        }
        url = f"{self.endpoint.url}{FEDERATION_PATH}"
        try:
            resp = self._session.post(url, json=payload, timeout=10)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionRefusedError(
                f"Nexus at {self.endpoint.url} refused connection. "
                "Ensure the nexus is running and the IP is reachable."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise ConnectionRefusedError(
                f"Nexus at {self.endpoint.url} timed out."
            ) from exc

        data = resp.json()
        record = RegistrationRecord(
            node_id=data["node_id"],
            hostname=data.get("hostname", self.hostname),
            capabilities=data.get("capabilities", self.capabilities),
            last_seen=time.time(),
            metadata=data.get("metadata", {}),
        )
        logger.info("Registered with nexus %s as %s", self.endpoint.url, self.node_id)
        return record

    def heartbeat(self, metadata: dict[str, Any] | None = None) -> None:
        """Send a lightweight heartbeat to the nexus."""
        payload = {
            "node_id": self.node_id,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        url = f"{self.endpoint.url}{HEARTBEAT_PATH}"
        try:
            resp = self._session.post(url, json=payload, timeout=5)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionRefusedError(
                f"Heartbeat to {self.endpoint.url} failed — nexus unreachable."
            )
        self._last_heartbeat = time.time()
        logger.debug("Heartbeat OK to %s", self.endpoint.url)

    def maybe_heartbeat(self) -> None:
        """Send a heartbeat only if the interval has elapsed."""
        if self._heartbeat_interval <= 0:
            return
        if self._last_heartbeat is None:
            self.heartbeat()
            return
        if (time.time() - self._last_heartbeat) >= self._heartbeat_interval:
            self.heartbeat()

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    # ── holonomy bridge integration ───────────────────────────

    def topology_check(
        self,
        peer_edges: list[tuple[str, str]],
        node_states: dict[str, float] | None = None,
    ) -> "BridgeReport":
        """Delegate fleet topology verification to holonomy-consensus.

        Builds a :class:`HolonomyBridge` from registered peers and
        runs cycle verification + H¹ cohomology + emergence detection.

        Args:
            peer_edges: List of (node_a, node_b) edges in the fleet.
            node_states: Optional consensus state per node.

        Returns:
            :class:`BridgeReport` summarizing topology health.
        """
        bridge = HolonomyBridge.from_fleet_edges(peer_edges, node_states=node_states)
        # Include self in the graph
        bridge.add_fleet_node(self.node_id)
        return bridge.check()

    # ── convenience factory ───────────────────────────────────

    @classmethod
    def from_defaults(
        cls,
        node_id: str,
        *,
        host: str = DEFAULT_NEXUS_IP,
        port: int = DEFAULT_NEXUS_PORT,
        **kwargs: Any,
    ) -> "FederatedNexus":
        """Create a nexus client using the fleet-default endpoint.

        *host* defaults to ``147.224.38.131`` (the Cocapn fleet nexus).
        """
        endpoint = FederationEndpoint(host=host, port=port)
        return cls(endpoint, node_id, **kwargs)
