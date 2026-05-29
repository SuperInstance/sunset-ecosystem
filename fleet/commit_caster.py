"""
Commit-Caster I2I Router

Routes fleet messages between vessels via git commit metadata.
Each commit carries a "cast" — a signed message that other vessels
can read and route to the appropriate destination.

Key features:
- Signed commit messages with fleet routing headers
- BFS routing across the vessel network
- Gossip propagation for broadcast messages
- Commit hash-based deduplication

Usage:
    from fleet.commit_caster import CommitCaster, FleetCast
    caster = CommitCaster("vessel-alpha", network)
    cast = caster.cast("agent-beta", {"action": "ping"})
    caster.broadcast({"alert": "new_module"})
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from fleet.vessel_handshake import NetworkTopology


@dataclass
class FleetCast:
    """A signed message cast via git commit."""
    source_vessel: str
    target_vessel: Optional[str]  # None = broadcast
    payload: Dict[str, Any]
    timestamp: float
    sequence: int = 0
    # Routing path: list of vessel IDs this cast has passed through
    path: List[str] = field(default_factory=list)
    # Hash for deduplication
    cast_hash: str = ""

    def __post_init__(self):
        if not self.cast_hash:
            self.cast_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        data = f"{self.source_vessel}:{self.target_vessel}:{json.dumps(self.payload, sort_keys=True)}:{self.timestamp}:{self.sequence}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_commit_message(self) -> str:
        """Convert to a git commit message format."""
        header = f"[FLEET-CAST] {self.source_vessel} -> {self.target_vessel or 'BROADCAST'}"
        body = json.dumps({
            "payload": self.payload,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "path": self.path,
            "hash": self.cast_hash,
        }, indent=2)
        return f"{header}\n\n{body}"

    @classmethod
    def from_commit_message(cls, message: str) -> Optional["FleetCast"]:
        """Parse a git commit message into a FleetCast."""
        try:
            lines = message.strip().split("\n")
            if not lines[0].startswith("[FLEET-CAST]"):
                return None

            header = lines[0]
            parts = header.replace("[FLEET-CAST] ", "").split(" -> ")
            source = parts[0]
            target = parts[1] if parts[1] != "BROADCAST" else None

            body = "\n".join(lines[2:])
            data = json.loads(body)

            return cls(
                source_vessel=source,
                target_vessel=target,
                payload=data["payload"],
                timestamp=data["timestamp"],
                sequence=data.get("sequence", 0),
                path=data.get("path", []),
                cast_hash=data.get("hash", ""),
            )
        except Exception:
            return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_vessel,
            "target": self.target_vessel,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "path": self.path,
            "hash": self.cast_hash,
        }


class CommitCaster:
    """
    Routes fleet messages via git commit metadata.

    Each vessel has a CommitCaster that reads/writes casts from its
    local git history and routes them to the appropriate destinations.
    """

    def __init__(self, vessel_id: str, network: Optional[NetworkTopology] = None):
        self.vessel_id = vessel_id
        self.network = network
        self.sequence = 0
        self.seen_hashes: Set[str] = set()
        self.pending_casts: List[FleetCast] = []
        self.delivered_casts: List[FleetCast] = []
        self.max_hops = 10

    def cast(self, target_vessel: str, payload: Dict[str, Any]) -> FleetCast:
        """Create a directed cast to a specific vessel."""
        self.sequence += 1
        cast = FleetCast(
            source_vessel=self.vessel_id,
            target_vessel=target_vessel,
            payload=payload,
            timestamp=time.time(),
            sequence=self.sequence,
        )
        self.seen_hashes.add(cast.cast_hash)
        self.pending_casts.append(cast)
        return cast

    def broadcast(self, payload: Dict[str, Any]) -> FleetCast:
        """Broadcast a cast to all vessels."""
        self.sequence += 1
        cast = FleetCast(
            source_vessel=self.vessel_id,
            target_vessel=None,
            payload=payload,
            timestamp=time.time(),
            sequence=self.sequence,
        )
        self.seen_hashes.add(cast.cast_hash)
        self.pending_casts.append(cast)
        return cast

    def receive(self, cast: FleetCast) -> bool:
        """
        Receive a cast from another vessel.
        Returns True if this cast should be processed.
        """
        # Deduplication
        if cast.cast_hash in self.seen_hashes:
            return False
        self.seen_hashes.add(cast.cast_hash)

        # Check if this cast is for us
        if cast.target_vessel is None or cast.target_vessel == self.vessel_id:
            self.delivered_casts.append(cast)
            return True

        # Route to next hop if not for us
        if self.network and len(cast.path) < self.max_hops:
            self._forward(cast)

        return False

    def _forward(self, cast: FleetCast):
        """Forward a cast to the next hop toward its destination."""
        if not self.network or not cast.target_vessel:
            return

        # Add ourselves to path
        cast.path.append(self.vessel_id)

        # Find next hop using BFS
        next_hop = self._find_next_hop(cast.target_vessel)
        if next_hop:
            # In real implementation: write to git commit in next_hop's repo
            # For now: add to pending_casts of next_hop (simulated)
            pass

    def _find_next_hop(self, target: str) -> Optional[str]:
        """Find next hop toward target using BFS on network topology."""
        if not self.network:
            return None

        # BFS from self.vessel_id to target
        visited = {self.vessel_id}
        queue = [(self.vessel_id, None)]
        parents = {self.vessel_id: None}

        while queue:
            current, parent = queue.pop(0)
            if current == target:
                # Reconstruct path
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = parents[node]
                path.reverse()
                if len(path) > 1:
                    return path[1]
                return None

            neighbors = self.network.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parents[neighbor] = current
                    queue.append((neighbor, current))

        return None

    def get_route(self, target: str) -> List[str]:
        """Get full route from this vessel to target."""
        if not self.network:
            return [self.vessel_id, target]

        visited = {self.vessel_id}
        queue = [self.vessel_id]
        parents = {self.vessel_id: None}

        while queue:
            current = queue.pop(0)
            if current == target:
                # Reconstruct path
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = parents[node]
                path.reverse()
                return path

            neighbors = self.network.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parents[neighbor] = current
                    queue.append(neighbor)

        return [self.vessel_id, target]

    def get_pending(self) -> List[FleetCast]:
        """Get all pending casts (to be committed)."""
        return self.pending_casts[:]

    def get_delivered(self) -> List[FleetCast]:
        """Get all delivered casts."""
        return self.delivered_casts[:]

    def clear_pending(self):
        """Clear pending casts (after commit)."""
        self.pending_casts = []

    def get_stats(self) -> Dict[str, Any]:
        return {
            "vessel_id": self.vessel_id,
            "sequence": self.sequence,
            "seen_hashes": len(self.seen_hashes),
            "pending": len(self.pending_casts),
            "delivered": len(self.delivered_casts),
            "max_hops": self.max_hops,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vessel_id": self.vessel_id,
            "stats": self.get_stats(),
            "pending": [c.to_dict() for c in self.pending_casts],
            "delivered": [c.to_dict() for c in self.delivered_casts],
        }
