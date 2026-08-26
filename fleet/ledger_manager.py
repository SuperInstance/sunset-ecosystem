from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class LedgerEntry:
    """An immutable ledger entry."""

    entry_id: str
    timestamp: float
    action: str
    actor: str
    data: Dict[str, Any]
    previous_hash: str
    hash: str = ""

    def compute_hash(self) -> str:
        """Compute hash of this entry."""
        import hashlib

        content = json.dumps(
            {
                "entry_id": self.entry_id,
                "timestamp": self.timestamp,
                "action": self.action,
                "actor": self.actor,
                "data": self.data,
                "previous_hash": self.previous_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "actor": self.actor,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
        }


class LedgerManager:
    """
    Immutable ledger for breeding decisions.

    Every breeding decision is recorded with hash chain for audit.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self._entries: List[LedgerEntry] = []
        self._last_hash: str = "0" * 16

    def append(self, action: str, actor: str, data: Dict[str, Any]) -> LedgerEntry:
        """Append a new entry to the ledger."""
        entry_id = f"entry_{len(self._entries)}"
        entry = LedgerEntry(
            entry_id=entry_id,
            timestamp=time.time(),
            action=action,
            actor=actor,
            data=data,
            previous_hash=self._last_hash,
        )
        entry.hash = entry.compute_hash()
        self._entries.append(entry)
        self._last_hash = entry.hash
        return entry

    def get(self, entry_id: str) -> Optional[LedgerEntry]:
        """Get an entry by ID."""
        for entry in self._entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def get_range(self, start: int, end: int) -> List[LedgerEntry]:
        """Get entries in a range."""
        return self._entries[start:end]

    def verify(self) -> bool:
        """Verify the integrity of the ledger."""
        for i in range(len(self._entries)):
            entry = self._entries[i]
            if entry.hash != entry.compute_hash():
                return False
            if i > 0:
                prev = self._entries[i - 1]
                if entry.previous_hash != prev.hash:
                    return False
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get ledger statistics."""
        actions = {}
        for entry in self._entries:
            a = entry.action
            actions[a] = actions.get(a, 0) + 1
        return {
            "total_entries": len(self._entries),
            "actions": actions,
            "last_hash": self._last_hash,
        }

    def export_json(self) -> str:
        """Export ledger as JSON."""
        return json.dumps(
            {
                "node": self.fleet_node_id,
                "entries": [e.to_dict() for e in self._entries],
            },
            indent=2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
        }
