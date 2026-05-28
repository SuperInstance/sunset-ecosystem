"""service_discovery.py — Fleet node discovery and health registry.

Provides:
1. Node registration with metadata (capabilities, version, load)
2. Heartbeat-based liveness tracking
3. Service lookup by capability or tag
4. Gossip protocol for propagation (via MeshVectorGossip integration)
5. Node eviction on missed heartbeats

Usage:
    sd = ServiceDiscovery(ttl=30.0)
    sd.register("node-1", {"ip": "10.0.0.1", "cpu": 8, "capabilities": ["gpu"]})
    nodes = sd.find_by_capability("gpu")
"""
from __future__ import annotations

__all__ = [
    "ServiceDiscovery",
    "NodeInfo",
]

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a fleet node."""
    node_id: str
    metadata: dict[str, Any]
    last_heartbeat: float
    ttl: float
    healthy: bool = True


class ServiceDiscovery:
    """Fleet node discovery with heartbeat-based liveness."""

    def __init__(self, ttl: float = 30.0, cleanup_interval: float = 60.0) -> None:
        self._ttl = ttl
        self._cleanup_interval = cleanup_interval
        self._nodes: dict[str, NodeInfo] = {}
        self._last_cleanup = time.time()

    # ── registration ────────────────────────────────────

    def register(
        self,
        node_id: str,
        metadata: dict[str, Any],
        ttl: float | None = None,
    ) -> None:
        """Register or update a node."""
        now = time.time()
        self._nodes[node_id] = NodeInfo(
            node_id=node_id,
            metadata=metadata,
            last_heartbeat=now,
            ttl=ttl or self._ttl,
        )
        logger.debug(f"Registered node {node_id}")

    def heartbeat(self, node_id: str) -> bool:
        """Record a heartbeat from a node."""
        if node_id not in self._nodes:
            return False
        self._nodes[node_id].last_heartbeat = time.time()
        self._nodes[node_id].healthy = True
        return True

    def unregister(self, node_id: str) -> bool:
        if node_id in self._nodes:
            del self._nodes[node_id]
            return True
        return False

    # ── query ──────────────────────────────────────────

    def get(self, node_id: str) -> NodeInfo | None:
        self._cleanup()
        return self._nodes.get(node_id)

    def all_nodes(self) -> list[NodeInfo]:
        self._cleanup()
        return list(self._nodes.values())

    def healthy_nodes(self) -> list[NodeInfo]:
        return [n for n in self.all_nodes() if n.healthy]

    def find_by_capability(self, capability: str) -> list[NodeInfo]:
        """Find nodes with a specific capability."""
        self._cleanup()
        return [
            n for n in self._nodes.values()
            if capability in n.metadata.get("capabilities", [])
        ]

    def find_by_tag(self, tag: str, value: Any) -> list[NodeInfo]:
        """Find nodes by metadata tag."""
        self._cleanup()
        return [
            n for n in self._nodes.values()
            if n.metadata.get(tag) == value
        ]

    def node_count(self) -> int:
        self._cleanup()
        return len(self._nodes)

    # ── cleanup ─────────────────────────────────────────

    def _cleanup(self) -> None:
        """Remove stale nodes."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        stale = [
            node_id for node_id, info in self._nodes.items()
            if now - info.last_heartbeat > info.ttl
        ]
        for node_id in stale:
            logger.warning(f"Node {node_id} expired")
            del self._nodes[node_id]

        self._last_cleanup = now

    def force_cleanup(self) -> int:
        """Force cleanup and return count removed."""
        self._last_cleanup = 0.0
        before = len(self._nodes)
        self._cleanup()
        return before - len(self._nodes)

    # ── stats ─────────────────────────────────────────

    def report(self) -> dict[str, Any]:
        self._cleanup()
        healthy = sum(1 for n in self._nodes.values() if n.healthy)
        return {
            "total": len(self._nodes),
            "healthy": healthy,
            "stale": len(self._nodes) - healthy,
        }

    def __repr__(self) -> str:
        return f"ServiceDiscovery(nodes={len(self._nodes)})"
