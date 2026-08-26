"""TernaryTypes — Ternary logic framework for fleet signal classification.

An emergent application inspired by Market Manifold's ternary insight:
Every continuous signal in the fleet can be classified into three states:
  -1 (Negative / Reduce / Critical)
   0 (Neutral / Hold / Warning)
  +1 (Positive / Accumulate / Healthy)

This module provides ternary logic operations, signal classification,
vector operations, and consensus mechanisms for distributed fleet decisions.

Usage
-----
    from fleet.ternary_types import TernaryValue, TernaryVector, TernaryMap

    # Classify a signal
    val = TernaryMap.classify(0.75, threshold=0.5)  # +1

    # Combine signals
    combined = TernaryValue.majority([+1, +1, 0, -1])  # +1

    # Vector operations
    vec = TernaryVector([+1, 0, -1, +1])
    assert vec.hamming_weight() == 2
    assert vec.density() == 0.5

Integration Points
------------------
- FleetMonitor: health status → ternary classification
- BreedOptimizer: offspring quality → ternary risk level
- VectorSwarm: distributed ternary consensus
- CognitiveCache: query confidence → ternary prediction
"""

from __future__ import annotations

__all__ = [
    "TernaryValue",
    "TernaryVector",
    "TernaryMap",
    "TernaryConsensus",
    "TernaryOperator",
]

from dataclasses import dataclass, field
from typing import Any, Callable


class TernaryValue:
    """A single ternary value: -1, 0, or +1.

    Ternary logic extends boolean logic with three states:
    - NEG (-1): False, negative, critical, reduce
    - ZERO (0): Unknown, neutral, warning, hold
    - POS (+1): True, positive, healthy, accumulate
    """

    NEG: int = -1
    ZERO: int = 0
    POS: int = +1
    VALID: set[int] = {-1, 0, +1}

    @classmethod
    def from_float(cls, value: float, threshold: float = 0.0) -> int:
        """Convert a float to ternary value.

        Parameters
        ----------
        value : float
            Input value.
        threshold : float
            Dead zone around zero. Values in [-threshold, +threshold]
            map to ZERO.

        Returns
        -------
        int
            -1, 0, or +1.
        """
        if value < -threshold:
            return cls.NEG
        if value > threshold:
            return cls.POS
        return cls.ZERO

    @classmethod
    def from_bool(cls, value: bool) -> int:
        """Convert boolean to ternary."""
        return cls.POS if value else cls.NEG

    @classmethod
    def not_(cls, a: int) -> int:
        """Ternary NOT: negates the sign, preserves zero."""
        cls._validate(a)
        return -a if a != 0 else 0

    @classmethod
    def and_(cls, a: int, b: int) -> int:
        """Ternary AND (minimum)."""
        cls._validate(a)
        cls._validate(b)
        return min(a, b)

    @classmethod
    def or_(cls, a: int, b: int) -> int:
        """Ternary OR (maximum)."""
        cls._validate(a)
        cls._validate(b)
        return max(a, b)

    @classmethod
    def xor_(cls, a: int, b: int) -> int:
        """Ternary XOR: strict inequality.

        0 acts as pass-through (XOR with 0 returns the other value).
        """
        cls._validate(a)
        cls._validate(b)
        if a == b:
            return 0
        if a == 0:
            return b
        if b == 0:
            return a
        return max(abs(a), abs(b)) * (1 if a > b else -1)

    @classmethod
    def majority(cls, values: list[int]) -> int:
        """Majority vote among ternary values.

        Returns the most common non-zero value, or ZERO if tied.
        """
        for v in values:
            cls._validate(v)
        pos_count = sum(1 for v in values if v == cls.POS)
        neg_count = sum(1 for v in values if v == cls.NEG)
        if pos_count > neg_count:
            return cls.POS
        if neg_count > pos_count:
            return cls.NEG
        return cls.ZERO

    @classmethod
    def consensus(cls, values: list[int], threshold: float = 0.6) -> int:
        """Consensus vote: requires threshold fraction agreement.

        Parameters
        ----------
        values : list[int]
            Ternary values.
        threshold : float
            Fraction required for consensus (0.5 = simple majority).

        Returns
        -------
        int
            +1 or -1 if consensus reached, ZERO otherwise.
        """
        if not values:
            return cls.ZERO
        for v in values:
            cls._validate(v)
        pos_count = sum(1 for v in values if v == cls.POS)
        neg_count = sum(1 for v in values if v == cls.NEG)
        total = len(values)
        if pos_count / total >= threshold:
            return cls.POS
        if neg_count / total >= threshold:
            return cls.NEG
        return cls.ZERO

    @classmethod
    def _validate(cls, value: int) -> None:
        if value not in cls.VALID:
            raise ValueError(f"Invalid ternary value: {value}. Must be -1, 0, or +1.")

    @classmethod
    def to_string(cls, value: int) -> str:
        """Convert ternary value to human-readable string."""
        cls._validate(value)
        mapping = {cls.NEG: "NEG", cls.ZERO: "ZERO", cls.POS: "POS"}
        return mapping[value]

    @classmethod
    def to_emoji(cls, value: int) -> str:
        """Convert ternary value to emoji."""
        cls._validate(value)
        mapping = {cls.NEG: "🔴", cls.ZERO: "🟡", cls.POS: "🟢"}
        return mapping[value]


@dataclass
class TernaryVector:
    """A vector of ternary values."""

    values: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        for v in self.values:
            TernaryValue._validate(v)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, idx: int) -> int:
        return self.values[idx]

    def __iter__(self):
        return iter(self.values)

    def hamming_weight(self) -> int:
        """Count non-zero elements."""
        return sum(1 for v in self.values if v != 0)

    def density(self) -> float:
        """Fraction of non-zero elements."""
        if not self.values:
            return 0.0
        return self.hamming_weight() / len(self.values)

    def balance(self) -> float:
        """Net balance: (pos - neg) / total. Range [-1, 1]."""
        if not self.values:
            return 0.0
        pos = sum(1 for v in self.values if v == TernaryValue.POS)
        neg = sum(1 for v in self.values if v == TernaryValue.NEG)
        return (pos - neg) / len(self.values)

    def entropy(self) -> float:
        """Shannon entropy of the ternary distribution."""
        if not self.values:
            return 0.0
        total = len(self.values)
        pos = sum(1 for v in self.values if v == TernaryValue.POS) / total
        neg = sum(1 for v in self.values if v == TernaryValue.NEG) / total
        zero = sum(1 for v in self.values if v == TernaryValue.ZERO) / total
        import math

        entropy = 0.0
        for p in [pos, neg, zero]:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def and_with(self, other: TernaryVector) -> TernaryVector:
        """Element-wise AND with another vector."""
        min_len = min(len(self.values), len(other.values))
        result = [
            TernaryValue.and_(self.values[i], other.values[i]) for i in range(min_len)
        ]
        return TernaryVector(result)

    def or_with(self, other: TernaryVector) -> TernaryVector:
        """Element-wise OR with another vector."""
        min_len = min(len(self.values), len(other.values))
        result = [
            TernaryValue.or_(self.values[i], other.values[i]) for i in range(min_len)
        ]
        return TernaryVector(result)

    def not_(self) -> TernaryVector:
        """Element-wise NOT."""
        return TernaryVector([TernaryValue.not_(v) for v in self.values])

    def majority(self) -> int:
        """Majority vote across all elements."""
        return TernaryValue.majority(self.values)

    def consensus(self, threshold: float = 0.6) -> int:
        """Consensus vote across all elements."""
        return TernaryValue.consensus(self.values, threshold)

    def to_string(self) -> str:
        """Convert to string representation."""
        return "".join(TernaryValue.to_string(v)[0] for v in self.values)

    @classmethod
    def from_floats(cls, floats: list[float], threshold: float = 0.0) -> TernaryVector:
        """Create a TernaryVector from float values."""
        return TernaryVector([TernaryValue.from_float(f, threshold) for f in floats])

    @classmethod
    def from_bools(cls, bools: list[bool]) -> TernaryVector:
        """Create a TernaryVector from boolean values."""
        return TernaryVector([TernaryValue.from_bool(b) for b in bools])


class TernaryMap:
    """Map continuous signals to ternary classification."""

    @classmethod
    def classify(cls, value: float, threshold: float = 0.0) -> int:
        """Classify a single float value."""
        return TernaryValue.from_float(value, threshold)

    @classmethod
    def classify_with_zscore(
        cls,
        value: float,
        mean: float,
        std: float,
        threshold: float = 1.0,
    ) -> int:
        """Classify using z-score: value = (x - mean) / std.

        Parameters
        ----------
        value : float
            Raw value.
        mean : float
            Distribution mean.
        std : float
            Distribution standard deviation.
        threshold : float
            Z-score threshold for classification.

        Returns
        -------
        int
            +1 if z-score > threshold, -1 if z-score < -threshold, 0 otherwise.
        """
        if std == 0:
            return TernaryValue.ZERO
        zscore = (value - mean) / std
        return TernaryValue.from_float(zscore, threshold)

    @classmethod
    def classify_vector(
        cls,
        values: list[float],
        threshold: float = 0.0,
    ) -> TernaryVector:
        """Classify a vector of floats."""
        return TernaryVector.from_floats(values, threshold)

    @classmethod
    def classify_percentile(
        cls,
        value: float,
        percentile_25: float,
        percentile_75: float,
    ) -> int:
        """Classify using percentile thresholds.

        -1 if value < p25, +1 if value > p75, 0 otherwise.
        """
        if value < percentile_25:
            return TernaryValue.NEG
        if value > percentile_75:
            return TernaryValue.POS
        return TernaryValue.ZERO

    @classmethod
    def window_classify(
        cls,
        values: list[float],
        window_size: int = 5,
        threshold: float = 0.0,
    ) -> TernaryVector:
        """Classify using rolling window averages."""
        if len(values) < window_size:
            return TernaryVector([])
        result = []
        for i in range(len(values) - window_size + 1):
            window = values[i : i + window_size]
            avg = sum(window) / len(window)
            result.append(TernaryValue.from_float(avg, threshold))
        return TernaryVector(result)


class TernaryConsensus:
    """Distributed consensus using ternary voting."""

    @classmethod
    def fleet_vote(
        cls,
        votes: dict[str, int],
        threshold: float = 0.6,
    ) -> dict[str, Any]:
        """Aggregate votes from multiple fleet nodes.

        Parameters
        ----------
        votes : dict[str, int]
            Node ID → ternary vote mapping.
        threshold : float
            Consensus threshold.

        Returns
        -------
        dict
            Result with consensus, confidence, and dissenters.
        """
        if not votes:
            return {"consensus": 0, "confidence": 0.0, "dissenters": []}

        values = list(votes.values())
        for v in values:
            TernaryValue._validate(v)

        consensus = TernaryValue.consensus(values, threshold)
        pos = sum(1 for v in values if v == TernaryValue.POS)
        neg = sum(1 for v in values if v == TernaryValue.NEG)
        zero = sum(1 for v in values if v == TernaryValue.ZERO)
        total = len(values)

        # Confidence = fraction of non-zero votes that agree with consensus
        if consensus == TernaryValue.POS:
            agreeing = pos
            non_zero = pos + neg
        elif consensus == TernaryValue.NEG:
            agreeing = neg
            non_zero = pos + neg
        else:
            agreeing = zero
            non_zero = total

        confidence = agreeing / non_zero if non_zero > 0 else 0.0

        # Dissenters = nodes that voted differently
        dissenters = [
            node for node, vote in votes.items() if vote != consensus and vote != 0
        ]

        return {
            "consensus": consensus,
            "confidence": confidence,
            "dissenters": dissenters,
            "pos_count": pos,
            "neg_count": neg,
            "zero_count": zero,
            "total": total,
        }

    @classmethod
    def weighted_vote(
        cls,
        votes: dict[str, tuple[int, float]],
        threshold: float = 0.6,
    ) -> dict[str, Any]:
        """Weighted consensus vote.

        Parameters
        ----------
        votes : dict[str, tuple[int, float]]
            Node ID → (vote, weight) mapping.
        threshold : float
            Consensus threshold (applied to total weight).

        Returns
        -------
        dict
            Result with weighted consensus.
        """
        if not votes:
            return {"consensus": 0, "confidence": 0.0, "dissenters": []}

        pos_weight = sum(w for v, w in votes.values() if v == TernaryValue.POS)
        neg_weight = sum(w for v, w in votes.values() if v == TernaryValue.NEG)
        zero_weight = sum(w for v, w in votes.values() if v == TernaryValue.ZERO)
        total_weight = pos_weight + neg_weight + zero_weight

        if total_weight == 0:
            return {"consensus": 0, "confidence": 0.0, "dissenters": []}

        if pos_weight / total_weight >= threshold:
            consensus = TernaryValue.POS
        elif neg_weight / total_weight >= threshold:
            consensus = TernaryValue.NEG
        else:
            consensus = TernaryValue.ZERO

        confidence = max(pos_weight, neg_weight) / total_weight
        dissenters = [
            node for node, (vote, _) in votes.items() if vote != consensus and vote != 0
        ]

        return {
            "consensus": consensus,
            "confidence": confidence,
            "dissenters": dissenters,
            "pos_weight": pos_weight,
            "neg_weight": neg_weight,
            "zero_weight": zero_weight,
            "total_weight": total_weight,
        }


class TernaryOperator:
    """Higher-order ternary operators."""

    @classmethod
    def if_then_else(cls, condition: int, true_val: int, false_val: int) -> int:
        """Ternary if-then-else.

        If condition is POS, return true_val.
        If condition is NEG, return false_val.
        If condition is ZERO, return the more conservative (min) of the two.
        """
        TernaryValue._validate(condition)
        TernaryValue._validate(true_val)
        TernaryValue._validate(false_val)
        if condition == TernaryValue.POS:
            return true_val
        if condition == TernaryValue.NEG:
            return false_val
        return TernaryValue.and_(true_val, false_val)

    @classmethod
    def clamp(cls, value: int, min_val: int, max_val: int) -> int:
        """Clamp a ternary value between min and max."""
        TernaryValue._validate(value)
        TernaryValue._validate(min_val)
        TernaryValue._validate(max_val)
        return max(min_val, min(value, max_val))

    @classmethod
    def switch(cls, value: int, cases: dict[int, Any]) -> Any:
        """Switch on ternary value."""
        TernaryValue._validate(value)
        return cases.get(value, cases.get(0, None))

    @classmethod
    def cascade(cls, values: list[int], default: int = 0) -> int:
        """Cascade: return first non-zero value, or default."""
        for v in values:
            TernaryValue._validate(v)
            if v != 0:
                return v
        return default
