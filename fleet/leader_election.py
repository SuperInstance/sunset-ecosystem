"""Leader election with heartbeats and failover.

Implements a leader election protocol with heartbeats, timeouts, and
graceful failover. Used for fleet coordinator election, singleton
services, and master-slave role management.

Usage:
    election = LeaderElection(node_id="node-1", timeout_sec=5)
    election.become_leader()
    assert election.is_leader() is True
    election.heartbeat()
    election.step_down()
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class LeaderElection:
    """
    Leader election with heartbeats.

    :param node_id: This node's identifier.
    :param timeout_sec: Heartbeat timeout for leader failure detection.
    :param clock: Optional clock function for testing.
    """

    def __init__(
        self,
        node_id: str,
        timeout_sec: float = 5.0,
        clock: Optional[callable] = None,
    ):
        self.node_id = node_id
        self.timeout_sec = timeout_sec
        self._clock = clock or time.time
        self._leader: Optional[str] = None
        self._last_heartbeat: float = 0.0

    # ------------------------------------------------------------------
    # Leadership
    # ------------------------------------------------------------------

    def become_leader(self) -> bool:
        """Claim leadership."""
        self._leader = self.node_id
        self._last_heartbeat = self._clock()
        return True

    def step_down(self) -> None:
        """Relinquish leadership."""
        if self._leader == self.node_id:
            self._leader = None
            self._last_heartbeat = 0.0

    # ------------------------------------------------------------------
    # Heartbeats
    # ------------------------------------------------------------------

    def heartbeat(self, leader_id: Optional[str] = None) -> None:
        """
        Record a heartbeat.

        :param leader_id: Leader identifier (defaults to self if leader).
        """
        lid = leader_id or self._leader
        if lid:
            self._leader = lid
            self._last_heartbeat = self._clock()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_leader(self) -> bool:
        """Check if this node is the current leader."""
        return self._leader == self.node_id and not self.leader_expired()

    def get_leader(self) -> Optional[str]:
        """Get current leader (None if expired)."""
        if self.leader_expired():
            return None
        return self._leader

    def leader_expired(self) -> bool:
        """Check if leader heartbeat has expired."""
        if self._leader is None:
            return True
        elapsed = self._clock() - self._last_heartbeat
        return elapsed > self.timeout_sec

    def can_become_leader(self) -> bool:
        """Check if this node can claim leadership (no active leader)."""
        return self.leader_expired()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "timeout_sec": self.timeout_sec,
            "is_leader": self.is_leader(),
            "leader": self.get_leader(),
            "leader_expired": self.leader_expired(),
            "last_heartbeat": self._last_heartbeat,
        }

    def __repr__(self) -> str:
        return f"<LeaderElection node={self.node_id} leader={self._leader}>"
