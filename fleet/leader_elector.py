"""Distributed leader election with TTL and heartbeat.

A fleet node can declare itself leader for a given role. Other nodes
watch for heartbeats; if the leader fails to renew within TTL, a new
election is triggered.

Usage:
    elector = LeaderElector(node_id="node-1", ttl_seconds=10)
    if elector.elect("breed-coordinator"):
        print("I am the leader")
    elector.heartbeat("breed-coordinator")  # Renew leadership

In production, back this with a distributed store (etcd, Redis,
Consul). The in-memory implementation here is for single-node testing
and can be swapped via the _store protocol.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class LeadershipRecord:
    leader_id: str
    acquired_at: float
    expires_at: float
    metadata: Dict[str, str] = field(default_factory=dict)


class LeaderElector:
    """
    In-memory leader election with TTL.

    :param node_id: Unique identifier for this node.
    :param ttl_seconds: How long a leadership claim lasts without renewal.
    :param clock: Optional monotonic time source (for testing).
    """

    def __init__(
        self,
        node_id: str,
        ttl_seconds: float = 10.0,
        clock: Optional[Callable[[], float]] = None,
    ):
        self._node_id = node_id
        self._ttl = ttl_seconds
        self._clock = clock or time.monotonic
        self._store: Dict[str, LeadershipRecord] = {}
        self._lock = threading.Lock()
        self._callbacks: Dict[str, list] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def elect(self, role: str, metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        Attempt to become leader for *role*.

        Returns True if this node won the election, False if another
        node currently holds an unexpired lease.
        """
        now = self._clock()
        meta = metadata or {}
        with self._lock:
            record = self._store.get(role)
            if record is not None:
                if record.leader_id != self._node_id and record.expires_at > now:
                    return False
            self._store[role] = LeadershipRecord(
                leader_id=self._node_id,
                acquired_at=now,
                expires_at=now + self._ttl,
                metadata=meta,
            )
            return True

    def heartbeat(self, role: str) -> bool:
        """Renew leadership lease for *role*. Returns False if not leader."""
        now = self._clock()
        with self._lock:
            record = self._store.get(role)
            if record is None or record.leader_id != self._node_id:
                return False
            record.expires_at = now + self._ttl
            return True

    def resign(self, role: str) -> bool:
        """Voluntarily give up leadership for *role*."""
        with self._lock:
            record = self._store.get(role)
            if record is not None and record.leader_id == self._node_id:
                del self._store[role]
                self._invoke_callbacks(role, "resigned")
                return True
            return False

    def is_leader(self, role: str) -> bool:
        """Check if this node currently holds an unexpired lease for *role*."""
        now = self._clock()
        with self._lock:
            record = self._store.get(role)
            if record is None:
                return False
            return record.leader_id == self._node_id and record.expires_at > now

    def get_leader(self, role: str) -> Optional[str]:
        """Return the current leader node ID for *role*, or None if expired."""
        now = self._clock()
        with self._lock:
            record = self._store.get(role)
            if record is None or record.expires_at <= now:
                return None
            return record.leader_id

    def get_leader_metadata(self, role: str) -> Optional[Dict[str, str]]:
        """Return metadata for the current leader, or None."""
        now = self._clock()
        with self._lock:
            record = self._store.get(role)
            if record is None or record.expires_at <= now:
                return None
            return dict(record.metadata)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def on_change(self, role: str, callback: Callable[[str, str], None]) -> None:
        """Register a callback for leadership changes: callback(role, event)."""
        with self._lock:
            self._callbacks.setdefault(role, []).append(callback)

    def _invoke_callbacks(self, role: str, event: str) -> None:
        for cb in self._callbacks.get(role, []):
            try:
                cb(role, event)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """Remove expired records. Returns count removed."""
        now = self._clock()
        removed = 0
        with self._lock:
            expired = [
                role for role, rec in self._store.items() if rec.expires_at <= now
            ]
            for role in expired:
                del self._store[role]
                self._invoke_callbacks(role, "expired")
                removed += 1
        return removed

    def list_roles(self) -> list:
        """List all roles with active (unexpired) leaders."""
        now = self._clock()
        with self._lock:
            return [
                role for role, rec in self._store.items() if rec.expires_at > now
            ]

    def stats(self) -> Dict[str, int]:
        return {
            "roles": len(self.list_roles()),
            "node_id": self._node_id,
        }

    def __repr__(self) -> str:
        return f"<LeaderElector node={self._node_id} ttl={self._ttl}s>"
