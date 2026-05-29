"""Snapshot and restore manager for fleet state.

Manages point-in-time snapshots of fleet state with incremental delta
support. Used for rollback, disaster recovery, and state migration between
fleet nodes.

Usage:
    mgr = SnapshotManager()
    mgr.snapshot("state-v1", {"nodes": ["a", "b"]})
    state = mgr.restore("state-v1")
    deltas = mgr.deltas_since("state-v1", newer_snapshot)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Snapshot:
    """A point-in-time snapshot."""

    name: str
    timestamp: float
    state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SnapshotManager:
    """
    In-memory snapshot manager with delta computation.

    Stores named snapshots and can compute JSON-patch-style deltas
    between any two snapshots.
    """

    def __init__(self):
        self._snapshots: Dict[str, Snapshot] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def snapshot(
        self,
        name: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Snapshot:
        """Create a named snapshot."""
        snap = Snapshot(
            name=name,
            timestamp=time.time(),
            state=dict(state),
            metadata=metadata or {},
        )
        self._snapshots[name] = snap
        return snap

    def restore(self, name: str) -> Optional[Dict[str, Any]]:
        """Restore state from a named snapshot."""
        snap = self._snapshots.get(name)
        return dict(snap.state) if snap else None

    def delete(self, name: str) -> bool:
        """Delete a snapshot."""
        if name in self._snapshots:
            del self._snapshots[name]
            return True
        return False

    # ------------------------------------------------------------------
    # Delta computation
    # ------------------------------------------------------------------

    def deltas_since(
        self, base_name: str, new_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Compute deltas between a snapshot and a new state.

        Returns list of {op, path, value} operations.
        """
        base = self._snapshots.get(base_name)
        if not base:
            return []
        return self._compute_delta(base.state, new_state)

    def delta_between(self, from_name: str, to_name: str) -> List[Dict[str, Any]]:
        """Compute deltas between two snapshots."""
        from_snap = self._snapshots.get(from_name)
        to_snap = self._snapshots.get(to_name)
        if not from_snap or not to_snap:
            return []
        return self._compute_delta(from_snap.state, to_snap.state)

    @staticmethod
    def _compute_delta(
        old: Dict[str, Any], new: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        deltas: List[Dict[str, Any]] = []
        all_keys = set(old.keys()) | set(new.keys())
        for key in all_keys:
            if key not in old:
                deltas.append({"op": "add", "path": f"/{key}", "value": new[key]})
            elif key not in new:
                deltas.append({"op": "remove", "path": f"/{key}"})
            elif old[key] != new[key]:
                deltas.append({"op": "replace", "path": f"/{key}", "value": new[key]})
        return deltas

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_snapshots(self) -> List[str]:
        return list(self._snapshots.keys())

    def get_snapshot(self, name: str) -> Optional[Snapshot]:
        return self._snapshots.get(name)

    def snapshot_count(self) -> int:
        return len(self._snapshots)

    def oldest_snapshot(self) -> Optional[str]:
        if not self._snapshots:
            return None
        return min(self._snapshots.items(), key=lambda x: x[1].timestamp)[0]

    def newest_snapshot(self) -> Optional[str]:
        if not self._snapshots:
            return None
        return max(self._snapshots.items(), key=lambda x: x[1].timestamp)[0]

    def apply_delta(self, state: Dict[str, Any], deltas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply deltas to a state copy."""
        result = dict(state)
        for d in deltas:
            op = d["op"]
            key = d["path"].lstrip("/")
            if op == "add" or op == "replace":
                result[key] = d["value"]
            elif op == "remove":
                result.pop(key, None)
        return result

    def __repr__(self) -> str:
        return f"<SnapshotManager snapshots={len(self._snapshots)}>"
