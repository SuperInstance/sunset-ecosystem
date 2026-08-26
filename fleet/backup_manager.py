from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Snapshot:
    """A snapshot of fleet state."""

    snapshot_id: str
    timestamp: float
    data: Dict[str, Any]
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "data": self.data,
            "tags": self.tags,
        }


class BackupManager:
    """
    Backup and restore for fleet state.

    Takes snapshots of key fleet state and supports restore.
    """

    def __init__(self, fleet_node_id: str = "default", max_snapshots: int = 50):
        self.fleet_node_id = fleet_node_id
        self._snapshots: List[Snapshot] = []
        self._max_snapshots = max_snapshots

    def snapshot(
        self, data: Dict[str, Any], tags: Optional[Dict[str, str]] = None
    ) -> Snapshot:
        """Take a snapshot of fleet state."""
        snap = Snapshot(
            snapshot_id=f"snap_{int(time.time() * 1000000)}",
            timestamp=time.time(),
            data=data,
            tags=tags or {},
        )
        self._snapshots.append(snap)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots :]
        return snap

    def restore(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Restore state from a snapshot."""
        for snap in self._snapshots:
            if snap.snapshot_id == snapshot_id:
                return snap.data
        return None

    def get_snapshots(
        self, tag_key: Optional[str] = None, tag_value: Optional[str] = None
    ) -> List[Snapshot]:
        """Get snapshots with optional tag filtering."""
        snapshots = self._snapshots
        if tag_key is not None:
            snapshots = [s for s in snapshots if s.tags.get(tag_key) == tag_value]
        return snapshots

    def get_latest(self) -> Optional[Snapshot]:
        """Get the most recent snapshot."""
        if not self._snapshots:
            return None
        return self._snapshots[-1]

    def delete(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        for i, snap in enumerate(self._snapshots):
            if snap.snapshot_id == snapshot_id:
                del self._snapshots[i]
                return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get backup statistics."""
        return {
            "total_snapshots": len(self._snapshots),
            "latest_timestamp": self._snapshots[-1].timestamp
            if self._snapshots
            else None,
        }

    def export_json(self) -> str:
        """Export all snapshots as JSON."""
        return json.dumps(
            {
                "node": self.fleet_node_id,
                "snapshots": [s.to_dict() for s in self._snapshots],
            },
            indent=2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
        }
