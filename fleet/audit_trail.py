from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class AuditEntry:
    """A single audit log entry."""
    timestamp: float
    action: str
    actor: str
    target: str
    details: Dict[str, Any] = field(default_factory=dict)
    hash: str = ""
    prev_hash: str = ""
    entry_id: str = field(default_factory=lambda: str(int(time.time() * 1000000)))

    def compute_hash(self) -> str:
        content = f"{self.prev_hash}:{self.timestamp}:{self.action}:{self.actor}:{self.target}:{json.dumps(self.details, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def seal(self) -> None:
        self.hash = self.compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "actor": self.actor,
            "target": self.target,
            "details": self.details,
            "hash": self.hash,
            "prev_hash": self.prev_hash,
        }


class AuditTrail:
    """
    Immutable, chained audit trail for fleet actions.

    Every action is logged with a hash chain (like a blockchain).
    Tamper detection via hash verification.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self.entries: List[AuditEntry] = []
        self._action_counts: Dict[str, int] = {}

    def log(self, action: str, actor: str, target: str,
            details: Optional[Dict[str, Any]] = None) -> AuditEntry:
        """Log an action to the audit trail."""
        prev_hash = self.entries[-1].hash if self.entries else ""
        entry = AuditEntry(
            timestamp=time.time(),
            action=action,
            actor=actor,
            target=target,
            details=details or {},
            prev_hash=prev_hash,
        )
        entry.seal()
        self.entries.append(entry)
        self._action_counts[action] = self._action_counts.get(action, 0) + 1
        return entry

    def log_breeding(self, generation: int, parent_ids: List[str],
                     child_id: str, actor: str = "breeder") -> AuditEntry:
        """Log a breeding event."""
        return self.log(
            action="breeding",
            actor=actor,
            target=child_id,
            details={
                "generation": generation,
                "parents": parent_ids,
            },
        )

    def log_deployment(self, model_id: str, target_node: str,
                       actor: str = "deployer") -> AuditEntry:
        """Log a deployment event."""
        return self.log(
            action="deployment",
            actor=actor,
            target=target_node,
            details={"model_id": model_id},
        )

    def log_consensus(self, proposal_id: str, votes: int,
                      actor: str = "consensus") -> AuditEntry:
        """Log a consensus vote."""
        return self.log(
            action="consensus",
            actor=actor,
            target=proposal_id,
            details={"votes": votes},
        )

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire hash chain."""
        for i in range(len(self.entries)):
            entry = self.entries[i]
            if entry.hash != entry.compute_hash():
                return False
            if i > 0 and entry.prev_hash != self.entries[i - 1].hash:
                return False
        return True

    def get_entries_by_action(self, action: str) -> List[AuditEntry]:
        """Get all entries for a specific action type."""
        return [e for e in self.entries if e.action == action]

    def get_entries_by_actor(self, actor: str) -> List[AuditEntry]:
        """Get all entries by a specific actor."""
        return [e for e in self.entries if e.actor == actor]

    def get_entries_by_target(self, target: str) -> List[AuditEntry]:
        """Get all entries affecting a specific target."""
        return [e for e in self.entries if e.target == target]

    def get_time_range(self, start: float, end: float) -> List[AuditEntry]:
        """Get entries within a time range."""
        return [e for e in self.entries if start <= e.timestamp <= end]

    def get_stats(self) -> Dict[str, Any]:
        """Get audit trail statistics."""
        return {
            "total_entries": len(self.entries),
            "action_counts": self._action_counts.copy(),
            "actors": list(set(e.actor for e in self.entries)),
            "targets": list(set(e.target for e in self.entries)),
            "chain_integrity": self.verify_chain(),
        }

    def export_json(self) -> str:
        """Export full audit trail as JSON."""
        return json.dumps({
            "node": self.fleet_node_id,
            "entries": [e.to_dict() for e in self.entries],
            "stats": self.get_stats(),
        }, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.fleet_node_id,
            "stats": self.get_stats(),
        }
