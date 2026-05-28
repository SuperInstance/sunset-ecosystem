"""load_balancer.py — Request load balancer for fleet nodes.

Provides:
1. Round-robin selection
2. Least-connections selection
3. Weighted round-robin
4. Health-aware selection (skip unhealthy nodes)
5. Sticky sessions (hash-based affinity)

Usage:
    lb = LoadBalancer(nodes=["a", "b", "c"])
    node = lb.pick(strategy="least_connections")
    lb.record_result(node, success=True)
"""
from __future__ import annotations

__all__ = [
    "LoadBalancer",
    "NodeStats",
]

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NodeStats:
    """Statistics for a backend node."""
    node_id: str
    weight: float = 1.0
    connections: int = 0
    successes: int = 0
    failures: int = 0
    healthy: bool = True
    last_used: float = 0.0


class LoadBalancer:
    """Load balancer with multiple strategies."""

    def __init__(self, nodes: list[str] | None = None) -> None:
        self._nodes: dict[str, NodeStats] = {}
        self._rr_index = 0
        if nodes:
            for n in nodes:
                self.add_node(n)

    def add_node(self, node_id: str, weight: float = 1.0) -> None:
        self._nodes[node_id] = NodeStats(node_id=node_id, weight=weight)

    def remove_node(self, node_id: str) -> bool:
        return self._nodes.pop(node_id, None) is not None

    def set_healthy(self, node_id: str, healthy: bool) -> None:
        if node_id in self._nodes:
            self._nodes[node_id].healthy = healthy

    # ── pick strategies ────────────────────────────────

    def pick(self, strategy: str = "round_robin", key: str | None = None) -> str | None:
        """Pick a node using the specified strategy."""
        healthy = [n for n in self._nodes.values() if n.healthy]
        if not healthy:
            return None

        if strategy == "round_robin":
            return self._pick_round_robin(healthy)
        elif strategy == "least_connections":
            return self._pick_least_connections(healthy)
        elif strategy == "weighted_round_robin":
            return self._pick_weighted(healthy)
        elif strategy == "sticky":
            return self._pick_sticky(healthy, key or "")
        else:
            return self._pick_round_robin(healthy)

    def _pick_round_robin(self, nodes: list[NodeStats]) -> str:
        idx = self._rr_index % len(nodes)
        self._rr_index += 1
        chosen = nodes[idx]
        chosen.last_used = time.time()
        return chosen.node_id

    def _pick_least_connections(self, nodes: list[NodeStats]) -> str:
        chosen = min(nodes, key=lambda n: n.connections)
        chosen.connections += 1
        chosen.last_used = time.time()
        return chosen.node_id

    def _pick_weighted(self, nodes: list[NodeStats]) -> str:
        import random
        total_weight = sum(n.weight for n in nodes)
        if total_weight <= 0:
            return self._pick_round_robin(nodes)
        pick = random.uniform(0, total_weight)
        cumulative = 0.0
        for n in nodes:
            cumulative += n.weight
            if pick <= cumulative:
                n.last_used = time.time()
                return n.node_id
        return nodes[-1].node_id

    def _pick_sticky(self, nodes: list[NodeStats], key: str) -> str:
        # Hash-based selection for session affinity
        digest = hashlib.md5(key.encode()).hexdigest()
        idx = int(digest, 16) % len(nodes)
        chosen = nodes[idx]
        chosen.last_used = time.time()
        return chosen.node_id

    # ── result tracking ───────────────────────────────

    def record_result(self, node_id: str, success: bool) -> None:
        if node_id not in self._nodes:
            return
        stats = self._nodes[node_id]
        if success:
            stats.successes += 1
            stats.connections = max(0, stats.connections - 1)
        else:
            stats.failures += 1
            stats.connections = max(0, stats.connections - 1)

    # ── stats ─────────────────────────────────────────

    def node_stats(self) -> dict[str, Any]:
        return {
            nid: {
                "weight": s.weight,
                "connections": s.connections,
                "successes": s.successes,
                "failures": s.failures,
                "healthy": s.healthy,
            }
            for nid, s in self._nodes.items()
        }

    def best_node(self) -> str | None:
        """Node with highest success rate."""
        healthy = [n for n in self._nodes.values() if n.healthy]
        if not healthy:
            return None
        total = {n.node_id: n.successes + n.failures for n in healthy}
        rates = {
            n.node_id: (n.successes / max(total[n.node_id], 1))
            for n in healthy
        }
        return max(rates, key=rates.get)

    def __repr__(self) -> str:
        return f"LoadBalancer(nodes={len(self._nodes)})"
