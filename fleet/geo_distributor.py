"""Geographic distribution of fleet nodes by region.

Assigns fleet nodes to geographic regions for latency-aware routing
and compliance. Supports region-aware node selection and failover.

Usage:
    geo = GeoDistributor()
    geo.add_region("us-east", ["node-1", "node-2"])
    geo.add_region("eu-west", ["node-3"])
    node = geo.select("us-east")
    assert node in ["node-1", "node-2"]
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set


class GeoDistributor:
    """
    Geographic node distributor.

    :param strategy: Selection strategy ("round_robin", "random", "least_loaded").
    """

    def __init__(self, strategy: str = "random"):
        self._strategy = strategy
        self._regions: Dict[str, List[str]] = {}
        self._node_region: Dict[str, str] = {}
        self._round_robin_idx: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Region management
    # ------------------------------------------------------------------

    def add_region(self, region: str, nodes: List[str]) -> None:
        """Register nodes for a region."""
        self._regions[region] = list(nodes)
        for node in nodes:
            self._node_region[node] = region
        self._round_robin_idx[region] = 0

    def remove_region(self, region: str) -> bool:
        """Remove a region and its nodes."""
        if region not in self._regions:
            return False
        for node in self._regions[region]:
            self._node_region.pop(node, None)
        del self._regions[region]
        del self._round_robin_idx[region]
        return True

    def add_node(self, region: str, node: str) -> None:
        """Add a node to a region."""
        if region not in self._regions:
            self._regions[region] = []
        self._regions[region].append(node)
        self._node_region[node] = region

    def remove_node(self, node: str) -> bool:
        """Remove a node from its region."""
        region = self._node_region.get(node)
        if not region:
            return False
        if region in self._regions and node in self._regions[region]:
            self._regions[region].remove(node)
        del self._node_region[node]
        return True

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, region: str) -> Optional[str]:
        """Select a node from a region."""
        nodes = self._regions.get(region, [])
        if not nodes:
            return None
        if self._strategy == "random":
            return random.choice(nodes)
        if self._strategy == "round_robin":
            idx = self._round_robin_idx.get(region, 0) % len(nodes)
            self._round_robin_idx[region] = idx + 1
            return nodes[idx]
        return nodes[0]

    def select_any(self) -> Optional[str]:
        """Select any node from any region."""
        all_nodes = [n for nodes in self._regions.values() for n in nodes]
        if not all_nodes:
            return None
        return random.choice(all_nodes)

    def failover(self, region: str, failed_node: str) -> Optional[str]:
        """Select an alternative node from the same region."""
        nodes = self._regions.get(region, [])
        candidates = [n for n in nodes if n != failed_node]
        if candidates:
            return random.choice(candidates)
        return None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def regions(self) -> List[str]:
        return list(self._regions.keys())

    def nodes_in_region(self, region: str) -> List[str]:
        return list(self._regions.get(region, []))

    def node_region(self, node: str) -> Optional[str]:
        return self._node_region.get(node)

    def node_count(self) -> int:
        return sum(len(nodes) for nodes in self._regions.values())

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "regions": len(self._regions),
            "nodes": self.node_count(),
            "strategy": self._strategy,
        }

    def __repr__(self) -> str:
        return (
            f"<GeoDistributor regions={len(self._regions)} nodes={self.node_count()}>"
        )
