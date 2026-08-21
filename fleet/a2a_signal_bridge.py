"""fleet/a2a_signal_bridge.py — FLUX-A2A Signal Protocol bridge.

Brings the flux-a2a-signal patterns into sunset-ecosystem:
- ConfidenceScore — first-class uncertainty with arithmetic
- SignalMessage — envelope with $schema versioning
- AgentPosition — multi-dimensional opinion space
- Branch — parallel exploration with merge strategies
- Fork — agent inheritance with state control
- ConsensusDetector — 6 consensus types with convergence tracking
- CoIterate — shared program traversal
- Discuss — structured agent discourse (debate, brainstorm, review, negotiate)
- Reflect — meta-cognition and self-assessment

Usage:
    from fleet.a2a_signal_bridge import ConfidenceScore, SignalMessage, AgentPosition
    from fleet.a2a_signal_bridge import Branch, Fork, ConsensusDetector, Discussion, SelfAssessor

    # Confidence arithmetic
    c1 = ConfidenceScore(0.8)
    c2 = ConfidenceScore(0.6)
    combined = c1.combine_min(c2)  # 0.6
    weighted = c1.combine_weighted(c2, 0.7, 0.3)  # 0.74

    # Signal message
    msg = SignalMessage(
        sender="Oracle1",
        recipient="kimi1",
        body={"task": "build bridge"},
        schema="https://flux.a2a/signal/v1",
    )

    # Consensus detection
    positions = [
        AgentPosition("Oracle1", [0.8, 0.2]),
        AgentPosition("kimi1", [0.75, 0.25]),
        AgentPosition("JC1", [0.1, 0.9]),
    ]
    detector = ConsensusDetector()
    result = detector.detect(positions, threshold=0.7)
    # result.type == "majority" (Oracle1 and kimi1 agree)

    # Structured discussion
    d = Discussion(topic="deploy v2", mode=DiscourseMode.DEBATE)
    d.add_round([
        Turn("Oracle1", AgentPosition("Oracle1", [0.8, 0.2]), ConfidenceScore(0.9)),
        Turn("kimi1", AgentPosition("kimi1", [0.75, 0.25]), ConfidenceScore(0.8)),
    ])
    result = d.consensus()

    # Self-assessment
    assessor = SelfAssessor(target_confidence=0.8)
    reflection = assessor.assess("kimi1", ConfidenceScore(0.6), {"strategy": "default"})
    # reflection.adjustments == ["gather_more_evidence", "seek_review"]
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Optional
from datetime import datetime, timezone


# ─────────────────────────────────────────────────────────────
# ConfidenceScore — first-class uncertainty
# ─────────────────────────────────────────────────────────────


@dataclass
class ConfidenceScore:
    """Confidence in [0.0, 1.0]. Every result carries one."""

    value: float

    def __post_init__(self):
        self.value = max(0.0, min(1.0, float(self.value)))

    def combine_min(self, other: ConfidenceScore) -> ConfidenceScore:
        """Propagation: uncertain input makes output uncertain."""
        return ConfidenceScore(min(self.value, other.value))

    def combine_weighted(
        self, other: ConfidenceScore, weight_self: float, weight_other: float
    ) -> ConfidenceScore:
        """Weighted average — branch merging."""
        total = weight_self + weight_other
        if total == 0:
            return ConfidenceScore(0.0)
        return ConfidenceScore(
            (weight_self * self.value + weight_other * other.value) / total
        )

    def combine_geometric(self, others: list[ConfidenceScore]) -> ConfidenceScore:
        """Geometric mean — co-iteration consensus."""
        scores = [self.value] + [o.value for o in others]
        product = 1.0
        for s in scores:
            if s <= 0:
                return ConfidenceScore(0.0)
            product *= s
        return ConfidenceScore(product ** (1.0 / len(scores)))

    def __bool__(self) -> bool:
        return self.value > 0.5

    def __repr__(self) -> str:
        return f"Confidence({self.value:.2f})"


# ─────────────────────────────────────────────────────────────
# SignalMessage — envelope with schema versioning
# ─────────────────────────────────────────────────────────────


@dataclass
class SignalMessage:
    """A2A signal message envelope."""

    sender: str
    recipient: str
    body: dict[str, Any] = field(default_factory=dict)
    schema: str = "https://flux.a2a/signal/v1"
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    meta: dict[str, Any] = field(default_factory=dict)
    confidence: ConfidenceScore = field(default_factory=lambda: ConfidenceScore(1.0))

    def to_dict(self) -> dict:
        return {
            "$schema": self.schema,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "recipient": self.recipient,
            "body": self.body,
            "confidence": self.confidence.value,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SignalMessage:
        """Parse from dict. Unknown fields go into meta."""
        known = {
            "sender",
            "recipient",
            "body",
            "message_id",
            "timestamp",
            "schema",
            "confidence",
            "meta",
        }
        meta = dict(d.get("meta", {}))
        for k, v in d.items():
            if k not in known and not k.startswith("$"):
                meta[k] = v
        return cls(
            sender=d.get("sender", ""),
            recipient=d.get("recipient", ""),
            body=d.get("body", {}),
            schema=d.get("$schema", d.get("schema", "https://flux.a2a/signal/v1")),
            message_id=d.get("message_id", ""),
            timestamp=d.get("timestamp", ""),
            meta=meta,
            confidence=ConfidenceScore(d.get("confidence", 1.0)),
        )


# ─────────────────────────────────────────────────────────────
# AgentPosition — multi-dimensional opinion space
# ─────────────────────────────────────────────────────────────


@dataclass
class AgentPosition:
    """An agent's stance in a multi-dimensional opinion space."""

    agent_id: str
    vector: list[float]
    confidence: ConfidenceScore = field(default_factory=lambda: ConfidenceScore(1.0))

    def cosine_similarity(self, other: AgentPosition) -> float:
        """Cosine similarity between two positions."""
        a, b = self.vector, other.vector
        if len(a) != len(b):
            raise ValueError("Vectors must have same dimension")
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def euclidean_distance(self, other: AgentPosition) -> float:
        """Euclidean distance between two positions."""
        a, b = self.vector, other.vector
        if len(a) != len(b):
            raise ValueError("Vectors must have same dimension")
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ─────────────────────────────────────────────────────────────
# Consensus Types
# ─────────────────────────────────────────────────────────────


class ConsensusType(Enum):
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    SUPERMAJORITY = "supermajority"
    CONVERGENCE = "convergence"
    COMPROMISE = "compromise"
    STALEMATE = "stalemate"


class ResolutionType(Enum):
    REBRANCH = "rebranch"
    ESCALATE = "escalate"
    VOTE = "vote"
    COMPROMISE_FIND = "compromise_find"
    DEFER = "defer"
    SPLIT_DIFFERENCE = "split_difference"
    RANDOM_ARBITRATOR = "random_arbitrator"


@dataclass
class ConsensusResult:
    """Result of consensus detection."""

    type: ConsensusType
    confidence: ConfidenceScore
    positions: list[AgentPosition]
    winning_position: Optional[AgentPosition] = None
    resolution: Optional[ResolutionType] = None
    explanation: str = ""
    agreement_count: int = 0
    total_count: int = 0

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "confidence": self.confidence.value,
            "winning_agent": self.winning_position.agent_id
            if self.winning_position
            else None,
            "resolution": self.resolution.value if self.resolution else None,
            "explanation": self.explanation,
            "agreement_count": self.agreement_count,
            "total_count": self.total_count,
        }


# ─────────────────────────────────────────────────────────────
# ConsensusDetector — detect when agents agree
# ─────────────────────────────────────────────────────────────


class ConsensusDetector:
    """
    Detect consensus among agents in a multi-dimensional opinion space.

    Consensus is a spectrum: stalemate → converging → converged → unanimous.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.history: list[list[AgentPosition]] = []

    def detect(
        self,
        positions: list[AgentPosition],
        threshold: Optional[float] = None,
    ) -> ConsensusResult:
        """Detect consensus type among positions."""
        thr = threshold if threshold is not None else self.similarity_threshold
        self.history.append(positions)

        n = len(positions)
        if n == 0:
            return ConsensusResult(
                type=ConsensusType.STALEMATE,
                confidence=ConfidenceScore(0.0),
                positions=positions,
                explanation="No agents to compare",
            )
        if n == 1:
            return ConsensusResult(
                type=ConsensusType.UNANIMOUS,
                confidence=positions[0].confidence,
                positions=positions,
                winning_position=positions[0],
                agreement_count=1,
                total_count=1,
                explanation="Single agent — unanimous by default",
            )

        # Build similarity matrix
        clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                sim = positions[i].cosine_similarity(positions[j])
                if sim >= thr:
                    # Merge clusters
                    ci, cj = None, None
                    for cid, members in clusters.items():
                        if i in members:
                            ci = cid
                        if j in members:
                            cj = cid
                    if ci is not None and cj is not None and ci != cj:
                        clusters[ci].extend(clusters[cj])
                        del clusters[cj]

        # Find largest cluster
        largest = max(clusters.values(), key=len)
        agreement_count = len(largest)
        total_count = n

        # Determine consensus type
        if agreement_count == n:
            ctype = ConsensusType.UNANIMOUS
            explanation = f"All {n} agents agree"
        elif agreement_count / n > (2 / 3):
            ctype = ConsensusType.SUPERMAJORITY
            explanation = f"{agreement_count}/{n} agents agree (supermajority)"
        elif agreement_count / n > 0.5:
            ctype = ConsensusType.MAJORITY
            explanation = f"{agreement_count}/{n} agents agree (majority)"
        elif agreement_count >= 2:
            ctype = ConsensusType.CONVERGENCE
            explanation = f"{agreement_count}/{n} agents converging"
        else:
            ctype = ConsensusType.STALEMATE
            explanation = f"No agreement — {n} divergent positions"

        winning = positions[largest[0]] if largest else None
        confidence = ConfidenceScore(agreement_count / n)

        # Detect resolution for stalemate
        resolution = None
        if ctype == ConsensusType.STALEMATE:
            resolution = ResolutionType.VOTE
        elif ctype == ConsensusType.CONVERGENCE:
            resolution = ResolutionType.COMPROMISE_FIND

        return ConsensusResult(
            type=ctype,
            confidence=confidence,
            positions=positions,
            winning_position=winning,
            resolution=resolution,
            explanation=explanation,
            agreement_count=agreement_count,
            total_count=total_count,
        )

    def convergence_trend(self) -> str:
        """Analyze trend over history."""
        if len(self.history) < 2:
            return "stable"
        recent = self.history[-3:]
        avg_agreement = []
        for batch in recent:
            if len(batch) <= 1:
                avg_agreement.append(1.0)
                continue
            agreements = 0
            pairs = 0
            for i in range(len(batch)):
                for j in range(i + 1, len(batch)):
                    if (
                        batch[i].cosine_similarity(batch[j])
                        >= self.similarity_threshold
                    ):
                        agreements += 1
                    pairs += 1
            avg_agreement.append(agreements / pairs if pairs > 0 else 0.0)
        if len(avg_agreement) < 2:
            return "stable"
        if avg_agreement[-1] > avg_agreement[0] + 0.1:
            return "converging"
        if avg_agreement[-1] < avg_agreement[0] - 0.1:
            return "diverging"
        return "stable"


# ─────────────────────────────────────────────────────────────
# Branch — parallel exploration
# ─────────────────────────────────────────────────────────────


class BranchStrategy(Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    COMPETITIVE = "competitive"


class MergeStrategyType(Enum):
    CONSENSUS = "consensus"
    VOTE = "vote"
    BEST = "best"
    ALL = "all"
    WEIGHTED_CONFIDENCE = "weighted_confidence"
    FIRST_COMPLETE = "first_complete"


@dataclass
class Branch:
    """Parallel exploration with configurable merge."""

    label: str
    body: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: ConfidenceScore = field(default_factory=lambda: ConfidenceScore(1.0))
    result: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "weight": self.weight,
            "confidence": self.confidence.value,
            "body": self.body,
            "result": self.result,
        }


@dataclass
class BranchPoint:
    """A point where execution splits into parallel branches."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    branches: list[Branch] = field(default_factory=list)
    strategy: BranchStrategy = BranchStrategy.PARALLEL
    merge_type: MergeStrategyType = MergeStrategyType.CONSENSUS
    status: str = "pending"

    def add_branch(
        self, label: str, body: dict[str, Any], weight: float = 1.0
    ) -> Branch:
        b = Branch(label=label, body=body, weight=weight)
        self.branches.append(b)
        return b

    def set_results(
        self, label: str, result: dict[str, Any], confidence: float
    ) -> None:
        for b in self.branches:
            if b.label == label:
                b.result = result
                b.confidence = ConfidenceScore(confidence)
                break

    def merge(self) -> dict[str, Any]:
        """Merge branch results according to merge_type."""
        completed = [b for b in self.branches if b.result is not None]
        if not completed:
            return {}

        if self.merge_type == MergeStrategyType.BEST:
            best = max(completed, key=lambda b: b.confidence.value)
            return {
                "winner": best.label,
                "result": best.result,
                "confidence": best.confidence.value,
            }

        if self.merge_type == MergeStrategyType.VOTE:
            counts: dict[str, int] = {}
            for b in completed:
                key = str(b.result)
                counts[key] = counts.get(key, 0) + 1
            winner = max(counts, key=counts.get)
            return {"winner": winner, "votes": counts, "branches": len(completed)}

        if self.merge_type == MergeStrategyType.WEIGHTED_CONFIDENCE:
            total_weight = sum(b.weight * b.confidence.value for b in completed)
            if total_weight == 0:
                return {}
            weighted_result = {}
            for b in completed:
                w = b.weight * b.confidence.value / total_weight
                for k, v in b.result.items():
                    if isinstance(v, (int, float)):
                        weighted_result[k] = weighted_result.get(k, 0.0) + v * w
            return {"weighted_result": weighted_result, "total_weight": total_weight}

        # Default: ALL — return all results
        return {
            "results": {b.label: b.result for b in completed},
            "confidences": {b.label: b.confidence.value for b in completed},
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "strategy": self.strategy.value,
            "merge_type": self.merge_type.value,
            "status": self.status,
            "branches": [b.to_dict() for b in self.branches],
        }


# ─────────────────────────────────────────────────────────────
# Fork — agent inheritance with state control
# ─────────────────────────────────────────────────────────────


class ForkOnComplete(Enum):
    COLLECT = "collect"
    DISCARD = "discard"
    SIGNAL = "signal"
    MERGE = "merge"


class ForkConflictMode(Enum):
    PARENT_WINS = "parent_wins"
    CHILD_WINS = "child_wins"
    NEGOTIATE = "negotiate"


@dataclass
class Fork:
    """Agent inheritance with fine-grained state control."""

    parent_id: str
    child_id: str
    inherited_state: dict[str, Any] = field(default_factory=dict)
    child_state: dict[str, Any] = field(default_factory=dict)
    on_complete: ForkOnComplete = ForkOnComplete.COLLECT
    conflict_mode: ForkConflictMode = ForkConflictMode.NEGOTIATE
    result: Optional[dict[str, Any]] = None
    confidence: ConfidenceScore = field(default_factory=lambda: ConfidenceScore(1.0))

    def merge(self) -> dict[str, Any]:
        """Merge child result into parent state."""
        if self.result is None:
            return dict(self.inherited_state)
        if self.conflict_mode == ForkConflictMode.PARENT_WINS:
            merged = dict(self.inherited_state)
            merged.update(self.child_state)
            return merged
        if self.conflict_mode == ForkConflictMode.CHILD_WINS:
            merged = dict(self.inherited_state)
            merged.update(self.child_state)
            merged.update(self.result)
            return merged
        # NEGOTIATE: confidence-weighted merge between inherited and result
        merged = dict(self.inherited_state)
        merged.update(self.child_state)
        for k, v in self.result.items():
            if (
                k in self.inherited_state
                and isinstance(v, (int, float))
                and isinstance(self.inherited_state[k], (int, float))
            ):
                w = self.confidence.value
                merged[k] = self.inherited_state[k] * (1 - w) + v * w
            else:
                merged[k] = v
        return merged

    def to_dict(self) -> dict:
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "on_complete": self.on_complete.value,
            "conflict_mode": self.conflict_mode.value,
            "result": self.result,
            "confidence": self.confidence.value,
        }


# ─────────────────────────────────────────────────────────────
# CoIterate — shared program traversal
# ─────────────────────────────────────────────────────────────


class SharedStateMode(Enum):
    CONFLICT = "conflict"
    MERGE = "merge"
    PARTITIONED = "partitioned"
    ISOLATED = "isolated"


class CoIterateMergeType(Enum):
    SEQUENTIAL_CONSENSUS = "sequential_consensus"
    PARALLEL_MERGE = "parallel_merge"
    MAJORITY_VOTE = "majority_vote"
    TRUST_WEIGHTED = "trust_weighted"


@dataclass
class Cursor:
    """Position in a shared program."""

    agent_id: str
    position: int = 0
    modifications: int = 0
    blocked: bool = False

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "position": self.position,
            "modifications": self.modifications,
            "blocked": self.blocked,
        }


@dataclass
class SharedProgram:
    """A program that multiple agents can traverse simultaneously."""

    name: str
    body: list[dict[str, Any]] = field(default_factory=list)
    cursors: list[Cursor] = field(default_factory=list)
    state_mode: SharedStateMode = SharedStateMode.MERGE

    def add_cursor(self, agent_id: str) -> Cursor:
        c = Cursor(agent_id=agent_id)
        self.cursors.append(c)
        return c

    def advance(self, agent_id: str) -> bool:
        for c in self.cursors:
            if c.agent_id == agent_id:
                c.position += 1
                return c.position < len(self.body)
        return False

    def modify(self, agent_id: str, position: int, value: Any) -> bool:
        for c in self.cursors:
            if c.agent_id == agent_id:
                if self.state_mode == SharedStateMode.ISOLATED:
                    c.modifications += 1
                    return True
                if self.state_mode == SharedStateMode.CONFLICT:
                    # Check if another agent modified this position
                    for other in self.cursors:
                        if other.agent_id != agent_id and other.modifications > 0:
                            c.blocked = True
                            return False
                c.modifications += 1
                if position < len(self.body):
                    self.body[position] = value
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "length": len(self.body),
            "cursors": [c.to_dict() for c in self.cursors],
            "state_mode": self.state_mode.value,
        }


# ─────────────────────────────────────────────────────────────
# Discuss — structured agent discourse
# ─────────────────────────────────────────────────────────────


class DiscourseMode(Enum):
    DEBATE = "debate"
    BRAINSTORM = "brainstorm"
    REVIEW = "review"
    NEGOTIATE = "negotiate"


@dataclass
class Turn:
    """A single turn in a structured discussion."""

    agent_id: str
    position: AgentPosition
    confidence: ConfidenceScore
    text: str = ""
    round_num: int = 0

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "position": self.position.vector,
            "confidence": self.confidence.value,
            "text": self.text,
            "round": self.round_num,
        }


@dataclass
class Discussion:
    """Structured multi-agent discourse with a topic and rounds."""

    topic: str
    mode: DiscourseMode = DiscourseMode.DEBATE
    rounds: list[list[Turn]] = field(default_factory=list)
    max_rounds: int = 3
    consensus_threshold: float = 0.7

    def add_round(self, turns: list[Turn]) -> None:
        for t in turns:
            t.round_num = len(self.rounds) + 1
        self.rounds.append(turns)

    def consensus(self) -> ConsensusResult:
        """Detect consensus across all turns in the discussion."""
        if not self.rounds:
            return ConsensusResult(
                type=ConsensusType.STALEMATE,
                confidence=ConfidenceScore(0.0),
                positions=[],
                explanation="No discussion rounds",
            )
        # Use latest round positions for consensus
        latest = self.rounds[-1]
        positions = [t.position for t in latest]
        detector = ConsensusDetector(similarity_threshold=self.consensus_threshold)
        return detector.detect(positions)

    def convergence_trend(self) -> str:
        """Track how positions evolved across rounds."""
        if len(self.rounds) < 2:
            return "stable"
        detector = ConsensusDetector(similarity_threshold=self.consensus_threshold)
        for r in self.rounds:
            positions = [t.position for t in r]
            detector.detect(positions)
        return detector.convergence_trend()

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "mode": self.mode.value,
            "rounds": [[t.to_dict() for t in r] for r in self.rounds],
            "max_rounds": self.max_rounds,
            "consensus": self.consensus().to_dict(),
            "trend": self.convergence_trend(),
        }


# ─────────────────────────────────────────────────────────────
# Reflect — meta-cognition and self-assessment
# ─────────────────────────────────────────────────────────────


@dataclass
class Reflection:
    """Meta-cognitive assessment of an agent's own performance."""

    agent_id: str
    self_confidence: ConfidenceScore
    strategy: str = ""
    assessment: str = ""
    adjustments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "self_confidence": self.self_confidence.value,
            "strategy": self.strategy,
            "assessment": self.assessment,
            "adjustments": self.adjustments,
        }


class SelfAssessor:
    """Agent self-assessment with strategy adjustment recommendations."""

    def __init__(self, target_confidence: float = 0.8):
        self.target_confidence = target_confidence
        self.history: list[Reflection] = []

    def assess(
        self, agent_id: str, confidence: ConfidenceScore, context: dict[str, Any]
    ) -> Reflection:
        """Assess an agent and suggest adjustments."""
        adjustments: list[str] = []
        if confidence.value < self.target_confidence * 0.5:
            assessment = "critical"
            adjustments.append("escalate_to_trusted_peer")
            adjustments.append("reduce_scope")
        elif confidence.value < self.target_confidence:
            assessment = "below_target"
            adjustments.append("gather_more_evidence")
            adjustments.append("seek_review")
        elif confidence.value > 0.95:
            assessment = "overconfident"
            adjustments.append("red_team_check")
        else:
            assessment = "healthy"

        strategy = context.get("strategy", "default")
        if assessment == "critical" and strategy != "conservative":
            adjustments.append("switch_to_conservative")
        elif assessment == "overconfident" and strategy != "aggressive":
            adjustments.append("stress_test_assumptions")

        reflection = Reflection(
            agent_id=agent_id,
            self_confidence=confidence,
            strategy=strategy,
            assessment=assessment,
            adjustments=adjustments,
        )
        self.history.append(reflection)
        return reflection

    def trend(self) -> str:
        """Confidence trend over assessment history."""
        if len(self.history) < 2:
            return "stable"
        values = [r.self_confidence.value for r in self.history]
        first_half = sum(values[: len(values) // 2]) / max(len(values) // 2, 1)
        second_half = sum(values[len(values) // 2 :]) / max(
            len(values) - len(values) // 2, 1
        )
        if second_half > first_half + 0.1:
            return "improving"
        elif second_half < first_half - 0.1:
            return "declining"
        return "stable"

    def to_dict(self) -> dict:
        return {
            "target_confidence": self.target_confidence,
            "trend": self.trend(),
            "history": [r.to_dict() for r in self.history[-10:]],  # last 10
        }
