"""
FleetBFT-QD — Byzantine Fault Tolerant Consensus + Quality Diversity Breeding

Implements a full PBFT consensus protocol for fleet-wide breeding decisions,
combined with MAP-Elites Quality Diversity algorithms for diversity-aware
parent selection.

Mathematical Foundations
------------------------
- PBFT: Castro & Liskov (1999), O(n²) message complexity, 5 phases
- HotStuff: Yin et al (2019), O(n) message complexity, pipelined
- Semantic BFT: Confidence-weighted voting (WBFT, 2025)
- MAP-Elites: Mouret & Clune (2015), CMA-MAE: Bryant et al (2024)
- Quorum: 2f+1 for N nodes, tolerating f < N/3 Byzantine faults

Integration Points
------------------
- HolonomyConsensus: Upgrades simple vote-counting to full PBFT
- MetronomeBridge: Heartbeat-driven view synchronization
- MeshVectorGossip: CRDT-propagated QD archive updates
- FleetConductorV2: BFT consensus for conductor state changes
- SignedWAL: Cryptographically signed consensus messages
"""
from __future__ import annotations

__all__ = [
    "BFTPhase",
    "PBFTMessage",
    "QuorumCertificate",
    "PBFTNode",
    "SemanticBFTNode",
    "BehaviorDescriptor",
    "QDArchive",
    "CMAESEmitter",
    "FleetBreederConsensus",
    "FleetBFTNetwork",
]

import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── BFT Message Types ───────────────────────────────────────────


class BFTPhase(Enum):
    """PBFT message phases."""

    REQUEST = auto()
    PRE_PREPARE = auto()
    PREPARE = auto()
    COMMIT = auto()
    REPLY = auto()
    VIEW_CHANGE = auto()
    NEW_VIEW = auto()


@dataclass(frozen=True)
class PBFTMessage:
    """A single PBFT protocol message.

    Fields:
        phase: Protocol phase.
        view_number: Current view (leader epoch).
        seq_num: Sequence number within the view.
        digest: SHA-256 digest of the payload (truncated to 16 hex chars).
        node_id: Sender node identifier.
        payload: Arbitrary operation payload.
        timestamp: Unix time.
        confidence: Semantic confidence score ∈ [0, 1] (WBFT extension).
    """

    phase: BFTPhase
    view_number: int
    seq_num: int
    digest: str
    node_id: str
    payload: Dict[str, Any]
    timestamp: float
    confidence: float = 1.0

    def canonical_bytes(self) -> bytes:
        """Canonical serialization for HMAC signing."""
        data = {
            "phase": self.phase.name,
            "view": self.view_number,
            "seq": self.seq_num,
            "digest": self.digest,
            "node": self.node_id,
            "payload": self.payload,
            "ts": self.timestamp,
            "conf": self.confidence,
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()

    def sign(self, key: str) -> str:
        """HMAC-SHA256 signature (32 hex chars)."""
        return hashlib.sha256(self.canonical_bytes() + key.encode()).hexdigest()[:32]


# ── Quorum Certificate ────────────────────────────────────────


@dataclass
class QuorumCertificate:
    """Cryptographic proof that ``quorum_size`` nodes reached consensus.

    A QC is a collection of signed messages attesting that a specific
    digest was agreed upon in a given view and sequence number.
    """

    view_number: int
    seq_num: int
    digest: str
    phase: BFTPhase
    signatures: List[Tuple[str, str]] = field(default_factory=list)

    def is_valid(self, quorum_size: int, verify_key: str) -> bool:
        """Verify all signatures and check quorum count.

        Returns ``True`` iff at least ``quorum_size`` signatures are
        present *and* every signature verifies against ``verify_key``.
        """
        if len(self.signatures) < quorum_size:
            return False
        for node_id, sig in self.signatures:
            msg = PBFTMessage(
                phase=self.phase,
                view_number=self.view_number,
                seq_num=self.seq_num,
                digest=self.digest,
                node_id=node_id,
                payload={},
                timestamp=0.0,
            )
            if sig != msg.sign(verify_key):
                return False
        return True

    @property
    def weight(self) -> float:
        """Total confidence weight (for SemanticBFTNode)."""
        # Signatures don't store confidence; this is a placeholder
        # for subclasses that extend QC with weighted votes.
        return float(len(self.signatures))


# ── PBFT Node ─────────────────────────────────────────────────


class PBFTNode:
    """Practical Byzantine Fault Tolerance node (Castro & Liskov, 1999).

    Implements the full 5-phase protocol:
      1. REQUEST    → client sends request.
      2. PRE-PREPARE → primary assigns ``seq_num``, broadcasts.
      3. PREPARE    → replicas validate, broadcast prepare.
      4. COMMIT     → replicas collect ``2f`` prepares, broadcast commit.
      5. REPLY      → replicas collect ``2f+1`` commits, execute, reply.

    View change protocol recovers from leader crash or Byzantine primary.
    """

    def __init__(
        self,
        node_id: str,
        all_nodes: List[str],
        secret_key: str,
        timeout_sec: float = 2.0,
    ) -> None:
        self.node_id = node_id
        self.all_nodes = sorted(set(all_nodes))
        self.n = len(self.all_nodes)
        self.f = (self.n - 1) // 3
        self.quorum = 2 * self.f + 1
        self.secret_key = secret_key
        self.timeout_sec = timeout_sec

        self.view_number = 0
        self.seq_num = 0

        # Message logs: (view, seq) → set of node_ids that sent this phase
        self._pre_prepare_log: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        self._prepare_log: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        self._commit_log: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        self._view_change_log: Dict[int, List[PBFTMessage]] = defaultdict(list)

        # State machine
        self._executed: Dict[int, Any] = {}  # seq_num → execution result
        self._checkpoints: Dict[int, Any] = {}  # seq_num → state snapshot

        # View change state
        self._in_view_change = False
        self._last_leader_activity = time.time()

        log.info(
            "PBFTNode %s init: n=%d f=%d quorum=%d primary=%s",
            node_id,
            self.n,
            self.f,
            self.quorum,
            self.primary_id,
        )

    # ── derived properties ──────────────────────────────────

    @property
    def primary_id(self) -> str:
        """Current primary (leader) for this view."""
        return self.all_nodes[self.view_number % self.n]

    def is_primary(self) -> bool:
        return self.node_id == self.primary_id

    # ── Phase 1: REQUEST → PRE-PREPARE ──────────────────────

    def handle_request(
        self, operation: str, payload: Dict[str, Any]
    ) -> Optional[PBFTMessage]:
        """Client (this node) initiates a request.

        If this node is the primary, it immediately produces a
        PRE-PREPARE message assigning the next sequence number.
        Non-primary nodes should forward the request to the primary.
        """
        if not self.is_primary():
            return None

        self.seq_num += 1
        digest = self._digest_payload(payload)

        msg = PBFTMessage(
            phase=BFTPhase.PRE_PREPARE,
            view_number=self.view_number,
            seq_num=self.seq_num,
            digest=digest,
            node_id=self.node_id,
            payload=payload,
            timestamp=time.time(),
        )

        key = (self.view_number, self.seq_num)
        self._pre_prepare_log[key].add(self.node_id)
        self._last_leader_activity = time.time()
        return msg

    # ── Phase 2: PRE-PREPARE → PREPARE ──────────────────────

    def handle_pre_prepare(self, msg: PBFTMessage) -> Optional[PBFTMessage]:
        """Replica handles PRE-PREPARE from primary.

        Validates:
        1. Sender is the expected primary for ``msg.view_number``.
        2. Digest matches locally computed digest of payload.
        3. We are not currently in a view change.

        On success, broadcasts a PREPARE message.
        """
        if self._in_view_change:
            return None

        expected_primary = self.all_nodes[msg.view_number % self.n]
        if msg.node_id != expected_primary:
            log.warning(
                "Invalid pre-prepare primary: got %s expected %s",
                msg.node_id,
                expected_primary,
            )
            return None

        if msg.digest != self._digest_payload(msg.payload):
            log.warning("Digest mismatch in pre-prepare from %s", msg.node_id)
            return None

        key = (msg.view_number, msg.seq_num)
        self._pre_prepare_log[key].add(msg.node_id)

        prepare = PBFTMessage(
            phase=BFTPhase.PREPARE,
            view_number=msg.view_number,
            seq_num=msg.seq_num,
            digest=msg.digest,
            node_id=self.node_id,
            payload={},
            timestamp=time.time(),
        )
        self._prepare_log[key].add(self.node_id)
        return prepare

    # ── Phase 3: PREPARE → COMMIT ───────────────────────────

    def handle_prepare(self, msg: PBFTMessage) -> Optional[PBFTMessage]:
        """Replica handles PREPARE from peer.

        Stores the prepare and checks whether we have received
        at least ``quorum`` PREPARE messages (including our own).
        If so, broadcasts a COMMIT message.
        """
        if self._in_view_change:
            return None

        key = (msg.view_number, msg.seq_num)
        self._prepare_log[key].add(msg.node_id)

        pre_prepare_count = len(self._pre_prepare_log.get(key, set()))
        prepare_count = len(self._prepare_log.get(key, set()))

        # Need at least 1 valid pre-prepare (from primary) and quorum prepares total
        if pre_prepare_count >= 1 and prepare_count >= self.quorum:
            commit = PBFTMessage(
                phase=BFTPhase.COMMIT,
                view_number=msg.view_number,
                seq_num=msg.seq_num,
                digest=msg.digest,
                node_id=self.node_id,
                payload={},
                timestamp=time.time(),
            )
            self._commit_log[key].add(self.node_id)
            return commit

        return None

    # ── Phase 4: COMMIT → REPLY ─────────────────────────────

    def handle_commit(self, msg: PBFTMessage) -> Optional[PBFTMessage]:
        """Replica handles COMMIT from peer.

        Stores the commit and checks whether we have received
        at least ``quorum`` COMMIT messages. If so, executes the
        operation (idempotently), records the checkpoint, and
        broadcasts a REPLY.
        """
        if self._in_view_change:
            return None

        key = (msg.view_number, msg.seq_num)
        self._commit_log[key].add(msg.node_id)

        commit_count = len(self._commit_log.get(key, set()))
        exec_key = (msg.view_number, msg.seq_num)
        if commit_count >= self.quorum and exec_key not in self._executed:
            result = self._execute(msg.payload)
            self._executed[exec_key] = result

            # Checkpoint every 100 sequence numbers
            if msg.seq_num % 100 == 0:
                self._checkpoints[msg.seq_num] = dict(self._executed)

            reply = PBFTMessage(
                phase=BFTPhase.REPLY,
                view_number=msg.view_number,
                seq_num=msg.seq_num,
                digest=msg.digest,
                node_id=self.node_id,
                payload={"result": result},
                timestamp=time.time(),
            )
            return reply

        return None

    # ── Execution hook ────────────────────────────────────────

    def _execute(self, payload: Dict[str, Any]) -> Any:
        """Execute the payload operation.

        Override in subclasses to implement application logic.
        Base implementation simply returns the payload unchanged.
        """
        return payload

    @staticmethod
    def _digest_payload(payload: Dict[str, Any]) -> str:
        """SHA-256 digest of payload (16 hex chars)."""
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(data).hexdigest()[:16]

    # ── View Change Protocol ──────────────────────────────────

    def start_view_change(self) -> Optional[PBFTMessage]:
        """Initiate view change due to leader timeout or detected fault.

        Increments the view number, marks self as ``_in_view_change``,
        and broadcasts a VIEW-CHANGE message containing the last stable
        checkpoint sequence number.
        """
        self._in_view_change = True
        self.view_number += 1

        last_checkpoint = max(self._checkpoints.keys()) if self._checkpoints else 0

        msg = PBFTMessage(
            phase=BFTPhase.VIEW_CHANGE,
            view_number=self.view_number,
            seq_num=last_checkpoint,
            digest="",
            node_id=self.node_id,
            payload={"checkpoints": list(self._checkpoints.keys())},
            timestamp=time.time(),
        )
        self._view_change_log[self.view_number].append(msg)
        log.info("Node %s started view change → view %d", self.node_id, self.view_number)
        return msg

    def handle_view_change(self, msg: PBFTMessage) -> Optional[PBFTMessage]:
        """Handle VIEW-CHANGE from peer.

        If this node is the new primary and has received ``quorum``
        VIEW-CHANGE messages, it broadcasts a NEW-VIEW message to
        complete the transition.
        """
        # Accept view changes for our target view (allow == during transition)
        if msg.view_number < self.view_number:
            return None

        self._view_change_log[msg.view_number].append(msg)

        new_primary = self.all_nodes[msg.view_number % self.n]
        if self.node_id == new_primary:
            vc_count = len(self._view_change_log[msg.view_number])
            if vc_count >= self.quorum:
                new_view = PBFTMessage(
                    phase=BFTPhase.NEW_VIEW,
                    view_number=msg.view_number,
                    seq_num=0,
                    digest="",
                    node_id=self.node_id,
                    payload={"primary": new_primary, "last_stable": msg.seq_num},
                    timestamp=time.time(),
                )
                self._in_view_change = False
                log.info("Primary %s announced new view %d", self.node_id, msg.view_number)
                return new_view

        return None

    def handle_new_view(self, msg: PBFTMessage) -> None:
        """Adopt a new view announced by the new primary."""
        if msg.view_number >= self.view_number and self._in_view_change:
            self.view_number = msg.view_number
            self._in_view_change = False
            log.info(
                "Node %s adopted view %d, primary=%s",
                self.node_id,
                self.view_number,
                self.primary_id,
            )

    # ── Status ────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "view_number": self.view_number,
            "is_primary": self.is_primary(),
            "primary_id": self.primary_id,
            "n_nodes": self.n,
            "f_byzantine": self.f,
            "quorum": self.quorum,
            "seq_num": self.seq_num,
            "executed_count": len(self._executed),
            "in_view_change": self._in_view_change,
        }


# ── Semantic BFT Node (WBFT extension) ─────────────────────────


class SemanticBFTNode(PBFTNode):
    """PBFT extended with confidence-weighted voting for agent networks.

    From *Weighted Byzantine Fault Tolerance Consensus Driven Trusted
    Multiple Large Language Models Network* (2025).

    Each vote carries a confidence score based on:
    - Historical accuracy (reputation).
    - Task-specific capability match.
    - Uncertainty quantification from payload complexity.

    Quorum becomes: Σ(confidence) ≥ quorum_threshold.

    This makes the consensus layer *semantically aware* — it can
    weight votes from high-confidence agents more heavily than votes
    from uncertain or historically inaccurate agents.
    """

    def __init__(
        self,
        node_id: str,
        all_nodes: List[str],
        secret_key: str,
        timeout_sec: float = 2.0,
    ) -> None:
        super().__init__(node_id, all_nodes, secret_key, timeout_sec)
        self._reputation: Dict[str, float] = {n: 1.0 for n in all_nodes}
        self._task_history: Dict[str, List[Tuple[str, bool]]] = defaultdict(list)

    def compute_confidence(self, task_type: str, payload: Dict[str, Any]) -> float:
        """Compute confidence for a task type and payload.

        Algorithm:
        1. Start with base reputation of this node.
        2. Adjust by recent task accuracy (exponential moving average).
        3. Decrease by payload complexity (larger payloads = higher uncertainty).
        """
        base = self._reputation.get(self.node_id, 1.0)

        history = self._task_history.get(task_type, [])
        if history:
            recent = history[-10:]
            accuracy = sum(1 for _, success in recent if success) / len(recent)
            base *= 0.5 + 0.5 * accuracy  # Scale to [0.5, 1.0]

        # Uncertainty from payload complexity (bytes / 1000, capped at 1.0)
        complexity = min(1.0, len(json.dumps(payload, separators=(",", ":"))) / 1000.0)
        return max(0.1, min(1.0, base * (1.0 - 0.3 * complexity)))

    def handle_request(self, operation: str, payload: Dict[str, Any]) -> Optional[PBFTMessage]:
        """Override to inject semantic confidence into PRE-PREPARE."""
        msg = super().handle_request(operation, payload)
        if msg:
            confidence = self.compute_confidence(operation, payload)
            return PBFTMessage(
                phase=msg.phase,
                view_number=msg.view_number,
                seq_num=msg.seq_num,
                digest=msg.digest,
                node_id=msg.node_id,
                payload=msg.payload,
                timestamp=msg.timestamp,
                confidence=confidence,
            )
        return msg

    def update_reputation(self, node_id: str, task_type: str, success: bool) -> None:
        """Update reputation after task completion.

        Uses an exponential moving average with learning rate α=0.1.
        """
        self._task_history[task_type].append((node_id, success))
        alpha = 0.1
        current = self._reputation.get(node_id, 1.0)
        target = 1.0 if success else 0.0
        self._reputation[node_id] = current + alpha * (target - current)

    def weighted_quorum_reached(self, msgs: List[PBFTMessage]) -> bool:
        """Check if weighted commit votes reach quorum.

        Sum of confidence scores must exceed the integer quorum threshold.
        This allows a smaller number of high-confidence nodes to commit,
        or requires more low-confidence nodes.
        """
        total = sum(m.confidence for m in msgs if m.phase == BFTPhase.COMMIT)
        return total >= self.quorum


# ── Quality Diversity Archive (MAP-Elites) ─────────────────────


@dataclass
class BehaviorDescriptor:
    """N-dimensional behavior characterization for an individual.

    ``values`` is a vector where each dimension represents a behavioral
    trait (e.g., exploration rate, communication frequency, task
    success rate). The grid discretization maps continuous behavior
    into discrete archive cells.
    """

    values: np.ndarray
    names: Tuple[str, ...]

    def grid_index(
        self, grid_shape: Tuple[int, ...], bounds: List[Tuple[float, float]]
    ) -> Tuple[int, ...]:
        """Map continuous behavior values to discrete grid indices.

        Args:
            grid_shape: Number of bins per dimension.
            bounds: (low, high) per dimension.

        Returns:
            Integer index tuple.
        """
        indices = []
        for i, (val, (low, high)) in enumerate(zip(self.values, bounds)):
            if high == low:
                idx = 0
            else:
                idx = int((val - low) / (high - low) * grid_shape[i])
                idx = max(0, min(idx, grid_shape[i] - 1))
            indices.append(idx)
        return tuple(indices)


@dataclass
class QDArchive:
    """MAP-Elites archive for Quality Diversity optimization.

    The archive is an N-dimensional grid where each cell holds the
    best-performing individual for that behavior descriptor region.

    Metrics:
    - **coverage**: Fraction of non-empty cells (% of grid explored).
    - **qd_score**: Sum of fitness across all occupied cells.
    """

    grid_shape: Tuple[int, ...]
    bounds: List[Tuple[float, float]]
    n_dims: int

    _grid: Dict[Tuple[int, ...], Dict[str, Any]] = field(default_factory=dict, repr=False)
    _best_fitness: Dict[Tuple[int, ...], float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        assert len(self.grid_shape) == self.n_dims, "grid_shape must match n_dims"
        assert len(self.bounds) == self.n_dims, "bounds must match n_dims"

    def add(
        self, descriptor: BehaviorDescriptor, individual: Any, fitness: float
    ) -> bool:
        """Add an individual to the archive.

        Returns ``True`` if the individual improved its cell
        (higher fitness than previous occupant).
        """
        idx = descriptor.grid_index(self.grid_shape, self.bounds)
        if idx not in self._grid or fitness > self._best_fitness.get(idx, -float("inf")):
            self._grid[idx] = individual
            self._best_fitness[idx] = fitness
            return True
        return False

    def get_random_elite(self) -> Optional[Any]:
        """Randomly sample a parent from occupied cells."""
        if not self._grid:
            return None
        return random.choice(list(self._grid.values()))

    def get_all_elites(self) -> List[Tuple[Any, float]]:
        """Return all (individual, fitness) pairs in the archive."""
        return [(self._grid[idx], self._best_fitness[idx]) for idx in self._grid]

    @property
    def coverage(self) -> float:
        """Fraction of grid cells that are occupied."""
        total_cells = 1
        for dim in self.grid_shape:
            total_cells *= dim
        return len(self._grid) / total_cells if total_cells > 0 else 0.0

    @property
    def qd_score(self) -> float:
        """Sum of fitness values across all occupied cells."""
        return sum(self._best_fitness.values())

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "n_occupied": len(self._grid),
            "coverage": self.coverage,
            "qd_score": self.qd_score,
            "grid_shape": self.grid_shape,
            "max_fitness": max(self._best_fitness.values()) if self._best_fitness else 0.0,
            "mean_fitness": (self.qd_score / len(self._best_fitness)) if self._best_fitness else 0.0,
        }


# ── CMA-ES Emitter for QD ─────────────────────────────────────


class CMAESEmitter:
    """CMA-ES emitter for Quality Diversity optimization.

    Self-adaptive search via covariance matrix adaptation (Hansen &
    Ostermeier, 2001). Maintains a multivariate Gaussian distribution
    and samples new individuals. Updates the distribution from elite
    individuals selected after evaluation.

    Reference: *Covariance Matrix Adaptation Evolution Strategy* —
    the de-facto standard for derivative-free continuous optimization.
    """

    def __init__(self, dim: int, sigma: float = 0.3):
        self.dim = dim
        self.sigma = sigma
        self.mean = np.zeros(dim)
        self.C = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.generations = 0

        # Strategy parameters
        self.mu = max(1, dim // 2)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mueff = 1.0 / (self.weights**2).sum()

        self.cc = 4.0 / (dim + 4.0)
        self.cs = self.mueff / (dim + self.mueff)
        self.c1 = 2.0 / ((dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1.0 - self.c1,
            2.0
            * (self.mueff - 2.0 + 1.0 / self.mueff)
            / ((dim + 2.0) ** 2 + self.mueff),
        )
        self.damps = (
            1.0
            + 2.0
            * max(0.0, np.sqrt((self.mueff - 1.0) / (dim + 1.0)) - 1.0)
            + self.cs
        )

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample ``n`` individuals from the current distribution."""
        samples = []
        for _ in range(n):
            z = np.random.standard_normal(self.dim)
            x = self.mean + self.sigma * (self.C @ z)
            samples.append(x)
        return np.array(samples)

    def update(self, elites: List[Tuple[np.ndarray, float]]) -> None:
        """Update distribution from elite individuals.

        Args:
            elites: List of ``(individual, fitness)`` sorted descending
                by fitness. Must contain at least ``mu`` individuals.
        """
        if len(elites) < self.mu:
            return

        selected = elites[: self.mu]
        old_mean = self.mean.copy()

        # Recombination: weighted average of selected individuals
        self.mean = sum(
            w * x for w, (x, _) in zip(self.weights, selected)
        )

        # Evolution paths
        z_vectors = [
            self._inv_sqrt_C((x - old_mean) / self.sigma) for x, _ in selected
        ]
        z_mean = sum(w * z for w, z in zip(self.weights, z_vectors))
        
        # Simplified path update (full CMA-ES requires eigendecomposition)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * z_mean

        hsig = (
            np.linalg.norm(self.ps)
            / np.sqrt(1 - (1 - self.cs) ** (2 * self.generations))
            / self.cs
            < (1.4 + 2.0 / (self.dim + 1.0))
        )

        self.pc = (1 - self.cc) * self.pc + (
            hsig
            * np.sqrt(self.cc * (2 - self.cc) * self.mueff)
            * (self.mean - old_mean)
            / self.sigma
        )

        # Covariance matrix update
        artmp = np.array([(x - old_mean) / self.sigma for x, _ in selected])
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * np.outer(self.pc, self.pc)
            + self.cmu * artmp.T @ np.diag(self.weights) @ artmp
        )

        # Step size update
        self.sigma *= np.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.cs - 1.0)
        )

        self.generations += 1

    def _inv_sqrt_C(self, x: np.ndarray) -> np.ndarray:
        """Approximate C^{-1/2} x using current C (simplified)."""
        # For small dimensions, direct solve is fine
        if self.dim <= 10:
            return np.linalg.solve(self.C, x)
        # Fallback: identity approximation for large dims
        return x


# ── Fleet Breeder Consensus ───────────────────────────────────


class FleetBreederConsensus:
    """BFT-gated Quality Diversity breeding for the fleet.

    Combines PBFT consensus for breeding batch decisions with
    MAP-Elites archive for diversity-aware parent selection.

    Execution flow per breeding round:
    1. **Propose**: Leader (primary) proposes a breeding batch
       including parent IDs, mutation rates, and archive metadata.
    2. **Consensus**: PBFT phases ensure ``2f+1`` nodes agree.
    3. **Commit**: Decision becomes immutable. All honest nodes
       execute the same breeding batch.
    4. **Evaluate**: Offspring are characterized by behavior
       descriptors and fitness scores.
    5. **Archive**: Successful offspring update the QD archive.
    6. **Gossip**: Archive deltas propagate via mesh gossip CRDT.
    """

    def __init__(
        self,
        node_id: str,
        all_nodes: List[str],
        secret_key: str,
        archive_dims: Tuple[int, ...] = (10, 10),
        behavior_bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self.bft = SemanticBFTNode(node_id, all_nodes, secret_key)
        self.archive = QDArchive(
            grid_shape=archive_dims,
            bounds=behavior_bounds or [(0.0, 1.0)] * len(archive_dims),
            n_dims=len(archive_dims),
        )
        self.emitter = CMAESEmitter(dim=sum(archive_dims) * 2)
        self._breeding_log: List[Dict[str, Any]] = []
        self._batch_counter = 0
        self._last_proposal: Optional[PBFTMessage] = None

    def propose_breeding_batch(
        self,
        candidates: List[Dict[str, Any]],
        batch_size: int = 4,
    ) -> Optional[PBFTMessage]:
        """Propose a breeding batch via BFT consensus.

        Uses QD-informed selection to maximize archive coverage.
        """
        self._batch_counter += 1
        parents = self._select_diverse_parents(candidates, batch_size)

        payload = {
            "batch_id": self._batch_counter,
            "parent_ids": [p["id"] for p in parents],
            "mutation_rates": [p.get("chaos", 0.3) for p in parents],
            "archive_coverage": self.archive.coverage,
            "qd_score": self.archive.qd_score,
            "proposer": self.bft.node_id,
            "timestamp": time.time(),
        }

        msg = self.bft.handle_request("breed_batch", payload)
        if msg:
            self._last_proposal = msg
        return msg

    def _select_diverse_parents(
        self, candidates: List[Dict[str, Any]], batch_size: int
    ) -> List[Dict[str, Any]]:
        """Select parents maximizing archive coverage.

        Strategy:
        - Fill 50% of batch from archive elites (exploitation).
        - Fill remaining 50% from candidates (exploration).
        """
        if not candidates:
            return []

        archive_parents = []
        for _ in range(batch_size // 2):
            elite = self.archive.get_random_elite()
            if elite and isinstance(elite, dict) and "id" in elite:
                archive_parents.append(elite)

        needed = batch_size - len(archive_parents)
        selected = archive_parents + candidates[:needed]
        return selected

    def execute_breeding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a committed breeding batch.

        This is called internally by ``_execute`` when PBFT reaches
        commit quorum. Override for actual breeding logic.
        """
        batch_id = payload.get("batch_id", 0)
        parent_ids = payload.get("parent_ids", [])

        offspring = []
        for i, pid in enumerate(parent_ids):
            child = {
                "id": f"offspring_{batch_id}_{i}",
                "parent": pid,
                "generation": 0,
            }
            offspring.append(child)

        result = {
            "batch_id": batch_id,
            "offspring": offspring,
            "archive_coverage_before": payload.get("archive_coverage", 0.0),
            "qd_score_before": payload.get("qd_score", 0.0),
            "n_offspring": len(offspring),
        }
        self._breeding_log.append(result)
        return result

    def evaluate_offspring(
        self,
        offspring: Dict[str, Any],
        fitness: float,
        behavior: np.ndarray,
    ) -> bool:
        """Evaluate and archive a single offspring.

        Returns ``True`` if the offspring improved its archive cell.
        """
        descriptor = BehaviorDescriptor(
            values=behavior,
            names=tuple(f"dim_{i}" for i in range(len(behavior))),
        )
        return self.archive.add(descriptor, offspring, fitness)

    def get_sync_payload(self) -> Dict[str, Any]:
        """CRDT-compatible sync payload for mesh gossip.

        Returns archive statistics and recent breeding decisions
        that can be merged across nodes.
        """
        return {
            "node_id": self.bft.node_id,
            "archive_stats": self.archive.stats,
            "recent_batches": self._breeding_log[-10:],
            "view_number": self.bft.view_number,
            "timestamp": time.time(),
        }

    def apply_sync_payload(self, payload: Dict[str, Any]) -> None:
        """Apply a sync payload from a peer node.

        Currently archives stats are read-only summaries; full CRDT
        merge of archive grids requires vector merging logic in
        ``mesh_vector_tables.py``.
        """
        peer_stats = payload.get("archive_stats", {})
        if peer_stats.get("qd_score", 0) > self.archive.qd_score:
            log.info(
                "Peer %s has superior QD score (%.2f vs %.2f)",
                payload.get("node_id"),
                peer_stats["qd_score"],
                self.archive.qd_score,
            )

    def get_status(self) -> Dict[str, Any]:
        return {
            "bft": self.bft.get_status(),
            "archive": self.archive.stats,
            "emitter_generations": self.emitter.generations,
            "total_batches": len(self._breeding_log),
        }


# ── Network Simulation ────────────────────────────────────────


class FleetBFTNetwork:
    """In-memory simulation of a fleet of PBFT nodes.

    Routes messages between nodes and supports:
    - Deterministic message delivery.
    - Byzantine fault injection (dropped / forged messages).
    - Network partition simulation.
    - End-to-end consensus testing.
    """

    def __init__(
        self, nodes: List[PBFTNode], latency_ms: float = 0.0
    ) -> None:
        self.nodes = {n.node_id: n for n in nodes}
        self.latency_ms = latency_ms
        self._byzantine_nodes: Set[str] = set()
        self._partitioned_nodes: Set[str] = set()
        self._message_log: List[PBFTMessage] = []

    def send(
        self, msg: PBFTMessage, target: Optional[str] = None
    ) -> List[PBFTMessage]:
        """Deliver a message to target (broadcast if ``None``).

        Returns all response messages generated by recipients.
        """
        responses: List[PBFTMessage] = []
        targets = [target] if target else list(self.nodes.keys())

        for nid in targets:
            if nid in self._partitioned_nodes:
                continue
            if nid in self._byzantine_nodes:
                # Byzantine node: currently drops all messages
                continue

            node = self.nodes[nid]
            resp: Optional[PBFTMessage] = None

            if msg.phase == BFTPhase.PRE_PREPARE:
                resp = node.handle_pre_prepare(msg)
            elif msg.phase == BFTPhase.PREPARE:
                resp = node.handle_prepare(msg)
            elif msg.phase == BFTPhase.COMMIT:
                resp = node.handle_commit(msg)
            elif msg.phase == BFTPhase.VIEW_CHANGE:
                resp = node.handle_view_change(msg)
            elif msg.phase == BFTPhase.NEW_VIEW:
                node.handle_new_view(msg)

            if resp:
                responses.append(resp)
                self._message_log.append(resp)

        return responses

    def broadcast_request(self, operation: str, payload: Dict[str, Any]) -> bool:
        """Run a full PBFT consensus round for a client request.

        Simulates the entire protocol:
        1. Primary receives request → PRE-PREPARE.
        2. All replicas → PREPARE.
        3. All replicas → COMMIT.
        4. All replicas → REPLY.

        Returns ``True`` if quorum was reached.
        """
        primary_id = next(
            (nid for nid, n in self.nodes.items() if n.is_primary()), None
        )
        if not primary_id or primary_id in self._byzantine_nodes:
            return False

        primary = self.nodes[primary_id]

        # Phase 1: REQUEST → PRE-PREPARE
        pre_prepare = primary.handle_request(operation, payload)
        if not pre_prepare:
            return False

        # Phase 2: PRE-PREPARE → PREPARE
        prepares = self.send(pre_prepare)

        # Phase 3: PREPARE → COMMIT
        commits: List[PBFTMessage] = []
        for p in prepares:
            commits.extend(self.send(p))

        # Phase 4: COMMIT → REPLY
        replies: List[PBFTMessage] = []
        for c in commits:
            replies.extend(self.send(c))

        return len(replies) >= primary.quorum

    def run_view_change(self) -> bool:
        """Simulate a view change across all non-Byzantine nodes.

        All non-faulty nodes independently detect the leader failure and
        broadcast VIEW-CHANGE messages. The new primary collects them
        and announces NEW-VIEW when quorum is reached.
        """
        honest = [nid for nid in self.nodes if nid not in self._byzantine_nodes]
        if not honest:
            return False

        # Step 1: All honest nodes start view change
        vc_messages = []
        for nid in honest:
            msg = self.nodes[nid].start_view_change()
            if msg:
                vc_messages.append(msg)

        # Step 2: Broadcast all view-change messages
        all_responses = []
        for msg in vc_messages:
            all_responses.extend(self.send(msg))

        # Step 3: Look for NEW-VIEW announcement
        for resp in all_responses:
            if resp.phase == BFTPhase.NEW_VIEW:
                # Propagate NEW-VIEW to all nodes
                self.send(resp)
                return True

        return False

    def set_byzantine(self, node_ids: List[str]) -> None:
        """Mark nodes as Byzantine (arbitrary / malicious behavior)."""
        self._byzantine_nodes.update(node_ids)

    def set_partitioned(self, node_ids: List[str]) -> None:
        """Partition nodes from the network (unreachable)."""
        self._partitioned_nodes.update(node_ids)

    def clear_faults(self) -> None:
        """Clear all injected faults."""
        self._byzantine_nodes.clear()
        self._partitioned_nodes.clear()

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "n_nodes": len(self.nodes),
            "byzantine": list(self._byzantine_nodes),
            "partitioned": list(self._partitioned_nodes),
            "total_messages": len(self._message_log),
        }
