"""Vector clocks for causal ordering of distributed events.

Implements Lamport-style vector clocks for tracking happens-before
relationships across fleet nodes. Used for conflict resolution,
causality tracking, and distributed debugging.

Usage:
    vc = VectorClock()
    vc.increment("node-1")
    vc.increment("node-2")
    other = VectorClock({"node-1": 1, "node-2": 0})
    assert vc.compare(other) == "greater"
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional


class VectorClock:
    """
    Vector clock for distributed causality tracking.

    Each node maintains a counter. The vector clock of an event
    is the maximum of all nodes' counters at that point.
    """

    def __init__(self, clocks: Optional[Dict[str, int]] = None):
        self._clocks: Dict[str, int] = dict(clocks) if clocks else {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def increment(self, node_id: str) -> None:
        """Increment this node's clock."""
        self._clocks[node_id] = self._clocks.get(node_id, 0) + 1

    def merge(self, other: "VectorClock") -> None:
        """Merge another vector clock (take component-wise max)."""
        for node, value in other._clocks.items():
            self._clocks[node] = max(self._clocks.get(node, 0), value)

    def compare(self, other: "VectorClock") -> str:
        """
        Compare with another vector clock.

        Returns: "equal", "greater", "less", "concurrent", or "incomparable"
        """
        all_nodes = set(self._clocks.keys()) | set(other._clocks.keys())
        if not all_nodes:
            return "equal"

        greater = False
        less = False
        for node in all_nodes:
            a = self._clocks.get(node, 0)
            b = other._clocks.get(node, 0)
            if a > b:
                greater = True
            elif a < b:
                less = True

        if greater and not less:
            return "greater"
        if less and not greater:
            return "less"
        if not greater and not less:
            return "equal"
        return "concurrent"

    def happens_before(self, other: "VectorClock") -> bool:
        """Check if this clock strictly happens before other."""
        return self.compare(other) == "less"

    def is_concurrent(self, other: "VectorClock") -> bool:
        """Check if clocks are concurrent (no causal relationship)."""
        return self.compare(other) == "concurrent"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, int]:
        return dict(self._clocks)

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "VectorClock":
        return cls(data)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __getitem__(self, node_id: str) -> int:
        return self._clocks.get(node_id, 0)

    def __repr__(self) -> str:
        return f"<VectorClock {self._clocks}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return False
        return self._clocks == other._clocks

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._clocks.items())))
