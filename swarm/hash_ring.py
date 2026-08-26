"""hash_ring.py — Consistent hash ring (Ketama-style) for fleet node assignment.

Uses MD5-based hashing with virtual node replication for balanced distribution.
Provides:
1. Add/remove nodes with minimal key remapping
2. Virtual node replication (default 150×) for uniform distribution
3. Binary search for O(log n) key-to-node lookup
4. Node weight support for heterogeneous capacity

Reference: Karger et al. (1997) "Consistent Hashing and Random Trees"
"""

from __future__ import annotations

__all__ = [
    "HashRing",
    "RingNode",
]

import bisect
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RingNode:
    """A physical node in the hash ring."""

    name: str
    weight: int = 1  # virtual node multiplier
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


class HashRing:
    """Consistent hash ring with virtual node replication.

    Usage:
        ring = HashRing([RingNode("node-a"), RingNode("node-b", weight=2)])
        node = ring.get_node("my-key")
        nodes = ring.get_nodes("my-key", n=3)  # 3 replicas
    """

    VIRTUALS_PER_WEIGHT = 150

    def __init__(
        self,
        nodes: list[RingNode] | None = None,
        replicas: int = 3,
    ) -> None:
        self.replicas = replicas
        self._nodes: dict[str, RingNode] = {}  # name -> RingNode
        self._ring: dict[int, str] = {}  # hash -> node_name
        self._sorted_hashes: list[int] = []  # sorted list for bisect

        if nodes:
            for n in nodes:
                self.add_node(n)

    # ── node management ────────────────────────────────

    def add_node(self, node: RingNode) -> None:
        """Add a node (and its virtual replicas) to the ring."""
        if node.name in self._nodes:
            self.remove_node(node.name)
        self._nodes[node.name] = node

        num_virtuals = node.weight * self.VIRTUALS_PER_WEIGHT
        for i in range(num_virtuals):
            key = f"{node.name}:{i}"
            h = self._hash(key)
            self._ring[h] = node.name

        self._sorted_hashes = sorted(self._ring.keys())
        logger.debug(f"Added node {node.name} with {num_virtuals} virtuals")

    def remove_node(self, name: str) -> None:
        """Remove a node and all its virtual replicas."""
        if name not in self._nodes:
            return
        node = self._nodes.pop(name)
        num_virtuals = node.weight * self.VIRTUALS_PER_WEIGHT

        to_remove: list[int] = []
        for i in range(num_virtuals):
            key = f"{name}:{i}"
            h = self._hash(key)
            if h in self._ring and self._ring[h] == name:
                to_remove.append(h)

        for h in to_remove:
            del self._ring[h]

        self._sorted_hashes = sorted(self._ring.keys())
        logger.debug(f"Removed node {name} ({len(to_remove)} virtuals)")

    def has_node(self, name: str) -> bool:
        return name in self._nodes

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def virtual_count(self) -> int:
        return len(self._ring)

    # ── key lookup ──────────────────────────────────────

    def get_node(self, key: str) -> RingNode | None:
        """Get the primary node for a key."""
        if not self._ring:
            return None
        h = self._hash(key)
        idx = bisect.bisect_right(self._sorted_hashes, h)
        if idx >= len(self._sorted_hashes):
            idx = 0
        chosen_hash = self._sorted_hashes[idx]
        name = self._ring[chosen_hash]
        return self._nodes.get(name)

    def get_nodes(self, key: str, n: int) -> list[RingNode]:
        """Get up to n unique nodes for a key (replica placement)."""
        if not self._ring or n <= 0:
            return []
        n = min(n, len(self._nodes))
        h = self._hash(key)
        idx = bisect.bisect_right(self._sorted_hashes, h)

        seen: set[str] = set()
        result: list[RingNode] = []
        while len(result) < n:
            if idx >= len(self._sorted_hashes):
                idx = 0
            chosen_hash = self._sorted_hashes[idx]
            name = self._ring[chosen_hash]
            if name not in seen:
                seen.add(name)
                node = self._nodes.get(name)
                if node:
                    result.append(node)
            idx += 1
            # Safety: break if we've gone full circle
            if len(seen) >= len(self._nodes):
                break

        return result

    def get_node_name(self, key: str) -> str | None:
        """Get just the node name for a key."""
        node = self.get_node(key)
        return node.name if node else None

    # ── distribution analysis ──────────────────────────

    def distribution(self, num_samples: int = 10_000) -> dict[str, int]:
        """Sample key distribution across nodes."""
        counts: dict[str, int] = {name: 0 for name in self._nodes}
        import random

        for i in range(num_samples):
            key = f"sample_key_{i}_{random.random()}"
            name = self.get_node_name(key)
            if name:
                counts[name] += 1
        return counts

    def balance_score(self, num_samples: int = 10_000) -> float:
        """Standard deviation of distribution (lower = more balanced)."""
        counts = self.distribution(num_samples)
        if not counts:
            return 0.0
        import numpy as np

        values = list(counts.values())
        return float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0.0

    def remapping_on_add(self, new_node: RingNode, num_samples: int = 1000) -> float:
        """Fraction of keys that would remap if new_node is added."""
        before = {}
        for i in range(num_samples):
            key = f"key_{i}"
            before[key] = self.get_node_name(key)

        self.add_node(new_node)
        changes = 0
        for i in range(num_samples):
            key = f"key_{i}"
            if self.get_node_name(key) != before[key]:
                changes += 1

        self.remove_node(new_node.name)
        return changes / num_samples

    # ── hash function ─────────────────────────────────────

    @staticmethod
    def _hash(key: str) -> int:
        """MD5-based hash returning a 64-bit integer."""
        digest = hashlib.md5(key.encode()).digest()
        # Use first 8 bytes as 64-bit unsigned int
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    # ── serialization ─────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "replicas": self.replicas,
            "node_count": self.node_count,
            "nodes": [
                {"name": n.name, "weight": n.weight, "metadata": n.metadata}
                for n in self._nodes.values()
            ],
            "virtual_count": self.virtual_count,
        }

    def __repr__(self) -> str:
        return f"HashRing(nodes={self.node_count}, virtuals={self.virtual_count}, replicas={self.replicas})"
