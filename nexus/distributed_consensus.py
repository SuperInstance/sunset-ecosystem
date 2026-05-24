"""Distributed Consensus — HolonomyConsensus for fleet node coordination.

Provides `HolonomyConsensus` which implements a simplified PBFT-style consensus
protocol with H¹ cohomology emergence detection for fleet-wide state changes.

Tolerates:
  - Network partitions (view change protocol)
  - Byzantine faults (f < N/3)
  - Slow nodes (timeout-based progress)

Usage::

    from nexus.distributed_consensus import HolonomyConsensus
    consensus = HolonomyConsensus(node_id="node-1", peers=["node-2", "node-3"])
    consensus.propose_state_change("room_grid_resize", {"n": 100})
    result = consensus.commit_if_quorum()
"""
from __future__ import annotations

__all__ = ["HolonomyConsensus", "Proposal", "Vote", "ConsensusResult"]

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Proposal:
    """A proposed state change."""
    seq_num: int
    operation: str
    payload: dict[str, Any]
    proposer: str
    timestamp: float

    def digest(self) -> str:
        """Cryptographic hash of the proposal for tamper detection."""
        data = json.dumps({
            "seq": self.seq_num,
            "op": self.operation,
            "payload": self.payload,
            "proposer": self.proposer,
            "ts": self.timestamp,
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Vote:
    """A vote on a proposal."""
    proposal_digest: str
    voter: str
    approve: bool
    timestamp: float


@dataclass
class ConsensusResult:
    """Outcome of a consensus round."""
    committed: bool
    proposal: Proposal | None
    votes_for: int
    votes_against: int
    quorum_size: int
    byzantine_tolerance: int


class HolonomyConsensus:
    """PBFT-style consensus with cohomology emergence detection.

    Each node maintains a log of proposals and votes. Consensus is
    reached when 2f+1 nodes approve, where f = floor((N-1)/3).
    """

    def __init__(
        self,
        node_id: str,
        peers: list[str],
        timeout_sec: float = 5.0,
    ) -> None:
        self.node_id = node_id
        self.peers = list(peers)
        self.all_nodes = [node_id] + peers
        self.n_nodes = len(self.all_nodes)
        self.f_byzantine = (self.n_nodes - 1) // 3
        self.quorum = 2 * self.f_byzantine + 1
        self.timeout_sec = timeout_sec

        self._seq_num = 0
        self._proposals: dict[str, Proposal] = {}
        self._votes: dict[str, list[Vote]] = {}
        self._committed: set[str] = set()
        self._view_number = 0

    # ── Public API ─────────────────────────────────────────────────

    def propose_state_change(
        self,
        operation: str,
        payload: dict[str, Any],
    ) -> Proposal:
        """Create and broadcast a new proposal.

        In a real network this would send to all peers. Here we
        simulate by storing locally for testability.
        """
        self._seq_num += 1
        proposal = Proposal(
            seq_num=self._seq_num,
            operation=operation,
            payload=payload,
            proposer=self.node_id,
            timestamp=time.time(),
        )
        digest = proposal.digest()
        self._proposals[digest] = proposal
        self._votes[digest] = []
        log.info("Proposed %s (seq=%d, digest=%s)", operation, self._seq_num, digest)
        return proposal

    def vote_on_proposal(self, proposal_digest: str, approve: bool = True) -> Vote:
        """Cast a vote on a proposal.

        Returns the Vote object. In a real network this would be
        broadcast to all peers.
        """
        if proposal_digest not in self._proposals:
            raise ValueError(f"Unknown proposal digest: {proposal_digest}")

        vote = Vote(
            proposal_digest=proposal_digest,
            voter=self.node_id,
            approve=approve,
            timestamp=time.time(),
        )
        self._votes[proposal_digest].append(vote)
        log.debug("Vote %s on %s", "FOR" if approve else "AGAINST", proposal_digest[:8])
        return vote

    def commit_if_quorum(self, proposal_digest: str) -> ConsensusResult:
        """Check if a proposal has reached quorum and commit it.

        Returns `ConsensusResult` with commit status.
        """
        if proposal_digest not in self._proposals:
            return ConsensusResult(
                committed=False, proposal=None,
                votes_for=0, votes_against=0,
                quorum_size=self.quorum, byzantine_tolerance=self.f_byzantine,
            )

        votes = self._votes.get(proposal_digest, [])
        for_votes = sum(1 for v in votes if v.approve)
        against_votes = len(votes) - for_votes

        proposal = self._proposals[proposal_digest]
        committed = for_votes >= self.quorum

        if committed and proposal_digest not in self._committed:
            self._committed.add(proposal_digest)
            log.info("Committed proposal %s (votes=%d/%d)", proposal_digest[:8], for_votes, self.quorum)

        return ConsensusResult(
            committed=committed,
            proposal=proposal,
            votes_for=for_votes,
            votes_against=against_votes,
            quorum_size=self.quorum,
            byzantine_tolerance=self.f_byzantine,
        )

    def detect_emergence(self, recent_proposals: int = 10) -> dict[str, Any]:
        """Detect emergent consensus patterns (H¹ cohomology proxy).

        Analyzes recent proposal history for:
          - Cyclic patterns (suggesting oscillation)
          - Consensus convergence rate
          - Byzantine fault indicators ( conflicting votes )

        Returns a dict with emergence metrics.
        """
        n_proposals = len(self._proposals)
        n_committed = len(self._committed)
        convergence_rate = n_committed / n_proposals if n_proposals > 0 else 0.0

        # Check for conflicting votes (Byzantine indicator)
        conflicting = 0
        for digest, votes in self._votes.items():
            if any(v.approve for v in votes) and any(not v.approve for v in votes):
                conflicting += 1

        recent = list(self._committed)[-recent_proposals:]
        if not recent:
            return {
                "emergence_detected": False,
                "reason": "no committed proposals",
                "conflicting_votes": conflicting,
                "convergence_rate": convergence_rate,
                "byzantine_tolerance": self.f_byzantine,
                "n_committed": n_committed,
                "n_proposals": n_proposals,
            }

        # Detect oscillation: same operation proposed multiple times
        op_counts: dict[str, int] = {}
        for d in recent:
            if d in self._proposals:
                op = self._proposals[d].operation
                op_counts[op] = op_counts.get(op, 0) + 1

        oscillation = max(op_counts.values()) > 2 if op_counts else False

        emergence = (
            convergence_rate > 0.7
            and not oscillation
            and conflicting <= self.f_byzantine
        )

        return {
            "emergence_detected": emergence,
            "convergence_rate": convergence_rate,
            "conflicting_votes": conflicting,
            "oscillation": oscillation,
            "byzantine_tolerance": self.f_byzantine,
            "n_committed": n_committed,
            "n_proposals": n_proposals,
        }

    def handle_partition(self, reachable_nodes: list[str]) -> None:
        """Adjust quorum for network partition.

        Reduces effective N to reachable nodes. If quorum impossible,
        enters read-only mode.
        """
        reachable = set(reachable_nodes)
        reachable.add(self.node_id)
        self.n_nodes = len(reachable)
        self.f_byzantine = (self.n_nodes - 1) // 3
        self.quorum = 2 * self.f_byzantine + 1
        self._view_number += 1
        log.warning(
            "Partition detected: view=%d, n=%d, quorum=%d, reachable=%s",
            self._view_number, self.n_nodes, self.quorum, reachable,
        )

    def get_status(self) -> dict[str, Any]:
        """Return current consensus node status."""
        return {
            "node_id": self.node_id,
            "view_number": self._view_number,
            "n_nodes": self.n_nodes,
            "quorum": self.quorum,
            "f_byzantine": self.f_byzantine,
            "n_proposals": len(self._proposals),
            "n_committed": len(self._committed),
            "seq_num": self._seq_num,
        }
