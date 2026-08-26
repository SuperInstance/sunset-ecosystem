"""Distributed hash ring consensus for fleet node assignment.

Combines consistent hashing with a lightweight consensus layer.
Nodes agree on ring topology changes via gossip-synchronized votes.

Usage:
    ring = ConsensusHashRing(node_id="node-1")
    ring.propose_add("node-2", weight=1)
    ring.vote("add:node-2", True)
    if ring.check_quorum("add:node-2"):
        ring.commit("add:node-2")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from fleet.consistent_hash_ring import HashRing

logger = logging.getLogger(__name__)


class ConsensusError(Exception):
    pass


@dataclass
class Proposal:
    """A ring topology change proposal."""

    action: str  # "add" or "remove"
    node_id: str
    weight: int = 1
    votes: Dict[str, bool] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    committed: bool = False


class ConsensusHashRing:
    """
    Consistent hash ring with distributed consensus for changes.

    :param node_id: This node's identifier.
    :param quorum_fn: Callable(votes_dict) -> bool to check quorum.
    """

    def __init__(
        self,
        node_id: str,
        quorum_fn: Optional[Callable[[Dict[str, bool]], bool]] = None,
    ):
        self.node_id = node_id
        self._ring = HashRing()
        self._proposals: Dict[str, Proposal] = {}
        self._quorum_fn = quorum_fn or self._default_quorum
        self._peers: Set[str] = set()
        self._stats: Dict[str, int] = {"proposals": 0, "commits": 0, "rejects": 0}

    # ------------------------------------------------------------------
    # Ring delegation
    # ------------------------------------------------------------------

    def get_node(self, key: str) -> Optional[str]:
        return self._ring.get_node(key)

    def get_nodes(self, key: str, n: int = 3) -> List[str]:
        return self._ring.get_nodes(key, n)

    def add_node_direct(self, node_id: str, weight: int = 1) -> None:
        """Add a node directly (bypass consensus — use for bootstrap only)."""
        self._ring.add_node(node_id, weight)
        self._peers.add(node_id)

    def remove_node_direct(self, node_id: str) -> None:
        """Remove a node directly (bypass consensus)."""
        self._ring.remove_node(node_id)
        self._peers.discard(node_id)

    # ------------------------------------------------------------------
    # Proposal lifecycle
    # ------------------------------------------------------------------

    def propose_add(self, node_id: str, weight: int = 1) -> str:
        """Propose adding a node. Returns proposal key."""
        key = f"add:{node_id}"
        self._proposals[key] = Proposal(
            action="add",
            node_id=node_id,
            weight=weight,
        )
        self._stats["proposals"] += 1
        return key

    def propose_remove(self, node_id: str) -> str:
        """Propose removing a node. Returns proposal key."""
        key = f"remove:{node_id}"
        self._proposals[key] = Proposal(
            action="remove",
            node_id=node_id,
        )
        self._stats["proposals"] += 1
        return key

    def vote(
        self, proposal_key: str, approve: bool, voter: Optional[str] = None
    ) -> None:
        """Vote on a proposal."""
        if proposal_key not in self._proposals:
            raise ConsensusError(f"Unknown proposal: {proposal_key}")
        voter = voter or self.node_id
        self._proposals[proposal_key].votes[voter] = approve

    def check_quorum(self, proposal_key: str) -> bool:
        """Check if a proposal has reached quorum."""
        if proposal_key not in self._proposals:
            return False
        proposal = self._proposals[proposal_key]
        return self._quorum_fn(proposal.votes)

    def commit(self, proposal_key: str) -> None:
        """Commit a proposal (apply the ring change)."""
        if proposal_key not in self._proposals:
            raise ConsensusError(f"Unknown proposal: {proposal_key}")
        proposal = self._proposals[proposal_key]
        if proposal.committed:
            return
        if not self.check_quorum(proposal_key):
            raise ConsensusError("Quorum not reached")

        if proposal.action == "add":
            self._ring.add_node(proposal.node_id, proposal.weight)
            self._peers.add(proposal.node_id)
        elif proposal.action == "remove":
            self._ring.remove_node(proposal.node_id)
            self._peers.discard(proposal.node_id)

        proposal.committed = True
        self._stats["commits"] += 1

    def reject(self, proposal_key: str) -> None:
        """Mark a proposal as rejected."""
        if proposal_key in self._proposals:
            self._proposals[proposal_key].committed = False
            self._stats["rejects"] += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def proposals(self) -> List[Proposal]:
        return list(self._proposals.values())

    def ring_size(self) -> int:
        return len(self._peers)

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def _default_quorum(self, votes: Dict[str, bool]) -> bool:
        """Simple majority quorum."""
        if not votes:
            return False
        approvals = sum(1 for v in votes.values() if v)
        return approvals > len(votes) / 2

    def __repr__(self) -> str:
        return f"<ConsensusHashRing node={self.node_id} peers={len(self._peers)}>"
