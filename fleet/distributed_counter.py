"""CRDT-based distributed counter (G-Counter).

Implements a grow-only counter that can be safely merged across
nodes without coordination. Used for fleet-wide metrics, rate
counting, and distributed tallies.

Usage:
    counter = DistributedCounter(node_id="node-1")
    counter.increment(5)
    other = DistributedCounter(node_id="node-2")
    other.increment(3)
    counter.merge(other)
    assert counter.value() == 8
"""

from __future__ import annotations

from typing import Dict, Any


class DistributedCounter:
    """
    G-Counter CRDT (grow-only counter).

    :param node_id: Unique node identifier.
    """

    def __init__(self, node_id: str):
        self._node_id = node_id
        self._counts: Dict[str, int] = {node_id: 0}

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def increment(self, delta: int = 1) -> None:
        """Increment this node's counter."""
        self._counts[self._node_id] += delta

    def value(self) -> int:
        """Get total counter value."""
        return sum(self._counts.values())

    def merge(self, other: "DistributedCounter") -> None:
        """Merge another counter into this one."""
        for node_id, count in other._counts.items():
            self._counts[node_id] = max(self._counts.get(node_id, 0), count)

    def get_node_value(self, node_id: str) -> int:
        """Get value for a specific node."""
        return self._counts.get(node_id, 0)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self._node_id, "counts": dict(self._counts)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistributedCounter":
        counter = cls(data["node_id"])
        counter._counts = dict(data.get("counts", {}))
        return counter

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "node_id": self._node_id,
            "nodes": len(self._counts),
            "total": self.value(),
        }

    def __repr__(self) -> str:
        return f"<DistributedCounter node={self._node_id} value={self.value()}>"
