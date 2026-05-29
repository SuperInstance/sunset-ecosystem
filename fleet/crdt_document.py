"""CRDT-based collaborative document for fleet state synchronization.

A Last-Writer-Wins (LWW) Element Dictionary CRDT that supports concurrent
updates from multiple fleet nodes without coordination. Used for shared
configuration, fleet state views, and collaborative documents.

Usage:
    doc = CRDTDocument(node_id="node-1")
    doc.set("key", "value")
    remote = CRDTDocument(node_id="node-2")
    remote.merge(doc)
    assert remote.get("key") == "value"
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class CRDTDocument:
    """
    LWW-Element-Dict CRDT with vector clock versioning.

    Each key stores (value, node_id, timestamp). On conflict, the entry
    with the highest (timestamp, node_id) wins deterministically.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._data: Dict[str, Tuple[Any, str, float]] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any, timestamp: Optional[float] = None) -> None:
        """Set a key with current node as origin."""
        import time

        ts = timestamp or time.time()
        existing = self._data.get(key)
        if existing is None or self._compare(existing, (value, self.node_id, ts)) < 0:
            self._data[key] = (value, self.node_id, ts)

    def get(self, key: str) -> Optional[Any]:
        """Get value for a key, or None if absent."""
        entry = self._data.get(key)
        return entry[0] if entry else None

    def delete(self, key: str, timestamp: Optional[float] = None) -> None:
        """Delete a key using tombstone (None value)."""
        import time

        self.set(key, None, timestamp or time.time())

    def has(self, key: str) -> bool:
        """Check if key exists and is not deleted."""
        entry = self._data.get(key)
        return entry is not None and entry[0] is not None

    # ------------------------------------------------------------------
    # CRDT merge
    # ------------------------------------------------------------------

    def merge(self, other: "CRDTDocument") -> None:
        """Merge another document's state into this one."""
        for key, entry in other._data.items():
            existing = self._data.get(key)
            if existing is None or self._compare(existing, entry) < 0:
                self._data[key] = entry

    # ------------------------------------------------------------------
    # Comparison helper
    # ------------------------------------------------------------------

    @staticmethod
    def _compare(a: Tuple[Any, str, float], b: Tuple[Any, str, float]) -> int:
        """
        Compare two entries. Returns -1 if a < b, 1 if a > b, 0 if equal.
        Breaks ties by node_id lexicographic order for determinism.
        """
        _, node_a, ts_a = a
        _, node_b, ts_b = b
        if ts_a != ts_b:
            return -1 if ts_a < ts_b else 1
        if node_a != node_b:
            return -1 if node_a < node_b else 1
        return 0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def keys(self) -> list:
        """Return all non-deleted keys."""
        return [k for k, v in self._data.items() if v[0] is not None]

    def items(self) -> Dict[str, Any]:
        """Return all non-deleted key-value pairs."""
        return {k: v[0] for k, v in self._data.items() if v[0] is not None}

    def to_dict(self) -> Dict[str, Tuple[Any, str, float]]:
        """Serialize to dict (for storage/transmission)."""
        return dict(self._data)

    @classmethod
    def from_dict(cls, node_id: str, data: Dict[str, Tuple[Any, str, float]]) -> "CRDTDocument":
        """Restore from serialized dict."""
        doc = cls(node_id)
        doc._data = dict(data)
        return doc

    def __repr__(self) -> str:
        return f"<CRDTDocument node={self.node_id} keys={len(self.keys())}>"
