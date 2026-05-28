"""FleetBFT-QD integration into BreederDaemonV2 parent selection.

Wires ``FleetBreederConsensus`` (PBFT + MAP-Elites) into
the breeding loop so that every parent batch must reach 2f+1
consensus before breeding proceeds.
"""

from __future__ import annotations

__all__ = ["BreederBFTIntegration", "_parse_agent_id"]

import logging
from typing import Any, Optional

from swarm.fleet_bft_qd import (
    BFTPhase,
    FleetBreederConsensus,
    FleetBFTNetwork,
    PBFTMessage,
)

logger = logging.getLogger(__name__)


class BreederBFTIntegration:
    """PBFT consensus gate for BreederDaemonV2 parent selection.

    Args:
        consensus: ``FleetBreederConsensus`` instance (owns the BFT node + QD archive).
        network: ``FleetBFTNetwork`` in-memory router for PBFT messages.
        timeout_sec: Not used by the synchronous simulation path, but kept
            for API compatibility with future async transports.
    """

    def __init__(
        self,
        consensus: FleetBreederConsensus,
        network: FleetBFTNetwork,
        timeout_sec: float = 5.0,
    ) -> None:
        self.consensus = consensus
        self.network = network
        self.timeout_sec = timeout_sec
        self._pending: dict[int, PBFTMessage] = {}

    # ── public API ──────────────────────────────────────────

    def propose_parents(
        self,
        candidates: list[dict[str, Any]],
        batch_size: int = 4,
    ) -> Optional[list[tuple[int, int]]]:
        """Propose a parent batch and run full PBFT consensus.

        Steps:
            1. Primary creates PRE-PREPARE via ``propose_breeding_batch``.
            2. Broadcast PRE-PREPARE → collect PREPARE votes.
            3. Broadcast PREPARE → collect COMMIT votes.
            4. Broadcast COMMIT → collect REPLY votes.
            5. If replies ≥ quorum (2f+1), parse and return parent pairs.

        Returns:
            List of ``(parent_a, parent_b)`` integer pairs, or ``None``
            if this node is not primary or consensus failed.
        """
        msg = self.consensus.propose_breeding_batch(candidates, batch_size)
        if msg is None:
            logger.debug("Not primary — cannot propose breeding batch")
            return None

        batch_id = msg.payload.get("batch_id", msg.seq_num)
        self._pending[batch_id] = msg

        ok = self._run_consensus_round(msg)
        if not ok:
            logger.warning(
                "PBFT consensus failed for batch %s (timeout/quorum not reached)",
                batch_id,
            )
            self.abort_parents(msg)
            return None

        # Consensus reached — execute breeding log + return parsed pairs
        self.commit_parents(msg)
        return self._parse_parent_pairs(msg)

    def commit_parents(self, msg: PBFTMessage) -> dict[str, Any]:
        """Execute the breeding batch once consensus is reached.

        Removes the proposal from the pending set and calls
        ``FleetBreederConsensus.execute_breeding``.
        """
        batch_id = msg.payload.get("batch_id", msg.seq_num)
        self._pending.pop(batch_id, None)
        return self.consensus.execute_breeding(msg.payload)

    def abort_parents(self, msg: PBFTMessage) -> None:
        """Abort a pending proposal and clean up pending state."""
        batch_id = msg.payload.get("batch_id", msg.seq_num)
        self._pending.pop(batch_id, None)
        logger.info("Aborted breeding proposal batch %s", batch_id)

    # ── internal helpers ────────────────────────────────────

    def _run_consensus_round(self, pre_prepare: PBFTMessage) -> bool:
        """Simulate the three PBFT broadcast phases through ``network``.

        Returns ``True`` when reply count ≥ quorum (2f+1).
        """
        # Phase 2: PRE-PREPARE → PREPARE
        prepares = self.network.send(pre_prepare)
        if not prepares:
            return False

        # Phase 3: PREPARE → COMMIT
        commits: list[PBFTMessage] = []
        for p in prepares:
            commits.extend(self.network.send(p))

        # Phase 4: COMMIT → REPLY
        replies: list[PBFTMessage] = []
        for c in commits:
            replies.extend(self.network.send(c))

        quorum = self.consensus.bft.quorum
        reached = len(replies) >= quorum
        logger.debug(
            "PBFT round: %d prepares → %d commits → %d replies (quorum=%d) %s",
            len(prepares),
            len(commits),
            len(replies),
            quorum,
            "✓" if reached else "✗",
        )
        return reached

    @staticmethod
    def _parse_parent_pairs(msg: PBFTMessage) -> list[tuple[int, int]]:
        """Convert payload ``parent_ids`` into integer (a, b) tuples."""
        raw_ids = msg.payload.get("parent_ids", [])
        pairs: list[tuple[int, int]] = []
        i = 0
        while i < len(raw_ids):
            a = _parse_agent_id(raw_ids[i])
            b = _parse_agent_id(raw_ids[i + 1]) if i + 1 < len(raw_ids) else None
            if a is not None and b is not None:
                pairs.append((a, b))
            elif a is not None:
                # Asexual fallback — pair with itself
                pairs.append((a, a))
            i += 2
        return pairs


def _parse_agent_id(raw: Any) -> Optional[int]:
    """Parse ``'agent_123'`` or ``123`` → ``123``."""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        if raw.startswith("agent_"):
            try:
                return int(raw.split("_")[1])
            except (IndexError, ValueError):
                return None
        try:
            return int(raw)
        except ValueError:
            return None
    return None
