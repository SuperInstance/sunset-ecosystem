"""Fleet node registry with metadata and health tracking.

Maintains a registry of fleet nodes with their metadata, capabilities,
and health status. Supports query by capability, health, and custom tags.

Usage:
    reg = NodeRegistry()
    reg.register("node-1", {"host": "10.0.0.1", "capabilities": ["breed", "train"]})
    nodes = reg.find_by_capability("breed")
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class NodeRegistry:
    """
    Fleet node registry with health and capability tracking.
    """

    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        node_id: str,
        metadata: Dict[str, Any],
        capabilities: Optional[List[str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Register or update a node."""
        self._nodes[node_id] = {
            "metadata": dict(metadata),
            "capabilities": list(capabilities or []),
            "tags": dict(tags or {}),
            "registered_at": time.time(),
            "last_heartbeat": time.time(),
            "healthy": True,
        }

    def unregister(self, node_id: str) -> bool:
        """Remove a node from the registry."""
        if node_id in self._nodes:
            del self._nodes[node_id]
            return True
        return False

    def heartbeat(self, node_id: str) -> bool:
        """Update last heartbeat timestamp."""
        node = self._nodes.get(node_id)
        if node:
            node["last_heartbeat"] = time.time()
            node["healthy"] = True
            return True
        return False

    def mark_unhealthy(self, node_id: str) -> bool:
        """Mark a node as unhealthy."""
        node = self._nodes.get(node_id)
        if node:
            node["healthy"] = False
            return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node info."""
        return self._nodes.get(node_id)

    def list_nodes(self) -> List[str]:
        return list(self._nodes.keys())

    def find_by_capability(self, capability: str) -> List[str]:
        """Find nodes with a specific capability."""
        return [
            nid
            for nid, info in self._nodes.items()
            if capability in info.get("capabilities", [])
        ]

    def find_by_tag(self, key: str, value: str) -> List[str]:
        """Find nodes with a specific tag."""
        return [
            nid
            for nid, info in self._nodes.items()
            if info.get("tags", {}).get(key) == value
        ]

    def healthy_nodes(self) -> List[str]:
        """List healthy nodes."""
        return [nid for nid, info in self._nodes.items() if info.get("healthy", False)]

    def unhealthy_nodes(self) -> List[str]:
        """List unhealthy nodes."""
        return [
            nid for nid, info in self._nodes.items() if not info.get("healthy", False)
        ]

    def stale_nodes(self, threshold_sec: float = 300.0) -> List[str]:
        """Find nodes whose heartbeat is older than threshold."""
        now = time.time()
        return [
            nid
            for nid, info in self._nodes.items()
            if now - info.get("last_heartbeat", 0) > threshold_sec
        ]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {
            "total": len(self._nodes),
            "healthy": len(self.healthy_nodes()),
            "unhealthy": len(self.unhealthy_nodes()),
        }

    def __repr__(self) -> str:
        return f"<NodeRegistry nodes={len(self._nodes)} healthy={len(self.healthy_nodes())}>"
