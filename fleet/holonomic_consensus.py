"""fleet/holonomic_consensus.py — BFT consensus using holonomy verification.

Distributed consensus where a cycle of votes is considered consistent if it
has zero holonomy. This is a novel consensus mechanism: instead of just
counting votes, we verify that the sequence of decisions forms a geometrically
consistent cycle on the Pythagorean manifold.

Usage
-----
    from fleet.holonomic_consensus import HolonomicBFT

    # 4 nodes, tolerate 1 Byzantine fault
    consensus = HolonomicBFT(node_id="alpha", peers=["beta", "gamma", "delta"], f=1)
    
    # Propose a value
    consensus.propose("breed_batch_42", value=[0.6, 0.8])
    
    # After receiving votes from peers, check holonomy
    if consensus.check_holonomy("breed_batch_42"):
        consensus.commit("breed_batch_42")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from swarm.constraint_bridge import ConstraintBridge


@dataclass
class Vote:
    """A vote in the holonomic consensus."""
    node_id: str
    proposal_id: str
    value: List[float]  # Direction vector (will be snapped to Pythagorean triple)
    timestamp: float


@dataclass
class HolonomicBFT:
    """Byzantine Fault Tolerant consensus with holonomic verification."""
    node_id: str
    peers: List[str]
    f: int = 1  # Number of Byzantine faults tolerated

    _proposals: Dict[str, Dict[str, Vote]] = field(default_factory=dict, repr=False)
    _committed: Dict[str, List[float]] = field(default_factory=dict, repr=False)
    _bridge: Optional[ConstraintBridge] = field(default=None, repr=False)

    def __post_init__(self):
        self._bridge = ConstraintBridge(density=200)
        self.all_nodes = [self.node_id] + self.peers
        self.quorum = 2 * self.f + 1

    def propose(self, proposal_id: str, value: List[float]) -> None:
        """Propose a value to the network."""
        vote = Vote(
            node_id=self.node_id,
            proposal_id=proposal_id,
            value=value,
            timestamp=0.0,  # Simplified
        )
        if proposal_id not in self._proposals:
            self._proposals[proposal_id] = {}
        self._proposals[proposal_id][self.node_id] = vote

    def receive_vote(self, vote: Vote) -> None:
        """Receive a vote from another node."""
        if vote.proposal_id not in self._proposals:
            self._proposals[vote.proposal_id] = {}
        self._proposals[vote.proposal_id][vote.node_id] = vote

    def check_holonomy(self, proposal_id: str) -> bool:
        """Check if the vote cycle for a proposal has zero holonomy.

        A consistent cycle means all nodes agree geometrically, not just
        by count. Zero holonomy = the sequence of votes forms a closed,
        consistent loop on the Pythagorean manifold.
        """
        if proposal_id not in self._proposals:
            return False

        votes = self._proposals[proposal_id]
        if len(votes) < self.quorum:
            return False

        # Extract values and check holonomy
        values = [vote.value for vote in votes.values()]
        return self._bridge.check_holonomy(values)

    def get_holonomy_error(self, proposal_id: str) -> float:
        """Get the holonomy error for a proposal (0 = perfect)."""
        if proposal_id not in self._proposals:
            return float('inf')

        votes = self._proposals[proposal_id]
        values = [vote.value for vote in votes.values()]
        if len(values) < 3:
            return 0.0

        total = 0.0
        for i in range(len(values)):
            a = np.array(values[i])
            b = np.array(values[(i + 1) % len(values)])
            total += self._bridge._angle_between(a, b)
        remainder = abs(total) % (2 * math.pi)
        return min(remainder, abs(remainder - 2 * math.pi))

    def commit(self, proposal_id: str) -> bool:
        """Commit a proposal if holonomy is zero and quorum is reached."""
        if not self.check_holonomy(proposal_id):
            return False

        votes = self._proposals[proposal_id]
        # Use median value as committed value
        all_values = [vote.value for vote in votes.values()]
        median_value = np.median(all_values, axis=0).tolist()
        self._committed[proposal_id] = median_value
        return True

    def get_commit(self, proposal_id: str) -> Optional[List[float]]:
        """Get committed value for a proposal."""
        return self._committed.get(proposal_id)

    def get_stats(self, proposal_id: str) -> dict:
        """Get consensus statistics for a proposal."""
        if proposal_id not in self._proposals:
            return {"error": "Proposal not found"}

        votes = self._proposals[proposal_id]
        return {
            "proposal_id": proposal_id,
            "votes_received": len(votes),
            "quorum_required": self.quorum,
            "holonomy_ok": self.check_holonomy(proposal_id),
            "holonomy_error": self.get_holonomy_error(proposal_id),
            "committed": proposal_id in self._committed,
            "nodes": list(votes.keys()),
        }

    def is_byzantine_fault(self, proposal_id: str, node_id: str,
                           threshold: float = 0.5) -> bool:
        """Detect if a node's vote is Byzantine (inconsistent with cycle)."""
        if proposal_id not in self._proposals:
            return False
        if node_id not in self._proposals[proposal_id]:
            return False

        # Check if removing this vote improves holonomy
        full_error = self.get_holonomy_error(proposal_id)
        
        # Temporarily remove vote
        removed_vote = self._proposals[proposal_id].pop(node_id)
        without_error = self.get_holonomy_error(proposal_id)
        self._proposals[proposal_id][node_id] = removed_vote

        # If removing significantly improves holonomy, it's likely Byzantine
        return without_error < full_error - threshold
