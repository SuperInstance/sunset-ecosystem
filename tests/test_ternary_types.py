"""Tests for TernaryTypes — ternary logic framework.

Reference: fleet/ternary_types.py
"""

from __future__ import annotations

import math

import pytest

from fleet.ternary_types import (
    TernaryConsensus,
    TernaryMap,
    TernaryOperator,
    TernaryValue,
    TernaryVector,
)


class TestTernaryValue:
    def test_from_float_positive(self) -> None:
        assert TernaryValue.from_float(0.5) == +1
        assert TernaryValue.from_float(1.0) == +1

    def test_from_float_negative(self) -> None:
        assert TernaryValue.from_float(-0.5) == -1
        assert TernaryValue.from_float(-1.0) == -1

    def test_from_float_zero(self) -> None:
        assert TernaryValue.from_float(0.0) == 0
        assert TernaryValue.from_float(0.1, threshold=0.2) == 0

    def test_from_float_threshold(self) -> None:
        assert TernaryValue.from_float(0.3, threshold=0.5) == 0
        assert TernaryValue.from_float(0.6, threshold=0.5) == +1
        assert TernaryValue.from_float(-0.6, threshold=0.5) == -1

    def test_from_bool(self) -> None:
        assert TernaryValue.from_bool(True) == +1
        assert TernaryValue.from_bool(False) == -1

    def test_not(self) -> None:
        assert TernaryValue.not_(+1) == -1
        assert TernaryValue.not_(-1) == +1
        assert TernaryValue.not_(0) == 0

    def test_and(self) -> None:
        assert TernaryValue.and_(+1, +1) == +1
        assert TernaryValue.and_(+1, 0) == 0
        assert TernaryValue.and_(+1, -1) == -1
        assert TernaryValue.and_(-1, 0) == -1
        assert TernaryValue.and_(0, 0) == 0

    def test_or(self) -> None:
        assert TernaryValue.or_(+1, +1) == +1
        assert TernaryValue.or_(+1, 0) == +1
        assert TernaryValue.or_(+1, -1) == +1
        assert TernaryValue.or_(-1, 0) == 0
        assert TernaryValue.or_(-1, -1) == -1

    def test_xor(self) -> None:
        assert TernaryValue.xor_(+1, +1) == 0
        assert TernaryValue.xor_(+1, -1) == +1
        assert TernaryValue.xor_(-1, +1) == -1
        assert TernaryValue.xor_(0, +1) == +1  # 0 acts as pass-through
        assert TernaryValue.xor_(0, -1) == -1
        assert TernaryValue.xor_(0, 0) == 0

    def test_majority(self) -> None:
        assert TernaryValue.majority([+1, +1, 0]) == +1
        assert TernaryValue.majority([-1, -1, 0]) == -1
        assert TernaryValue.majority([+1, -1, 0]) == 0
        assert TernaryValue.majority([+1, +1, -1, -1]) == 0

    def test_consensus(self) -> None:
        assert TernaryValue.consensus([+1, +1, +1], threshold=0.6) == +1
        assert TernaryValue.consensus([-1, -1, -1], threshold=0.6) == -1
        # 2/3 = 66.7% >= 60% threshold → consensus +1
        assert TernaryValue.consensus([+1, +1, 0], threshold=0.6) == +1
        assert (
            TernaryValue.consensus([+1, +1, +1, -1], threshold=0.75) == +1
        )  # 3/4 = 75%

    def test_consensus_empty(self) -> None:
        assert TernaryValue.consensus([], threshold=0.6) == 0

    def test_validate_invalid(self) -> None:
        with pytest.raises(ValueError):
            TernaryValue._validate(2)
        with pytest.raises(ValueError):
            TernaryValue._validate(-2)

    def test_to_string(self) -> None:
        assert TernaryValue.to_string(+1) == "POS"
        assert TernaryValue.to_string(0) == "ZERO"
        assert TernaryValue.to_string(-1) == "NEG"

    def test_to_emoji(self) -> None:
        assert TernaryValue.to_emoji(+1) == "🟢"
        assert TernaryValue.to_emoji(0) == "🟡"
        assert TernaryValue.to_emoji(-1) == "🔴"


class TestTernaryVector:
    def test_hamming_weight(self) -> None:
        vec = TernaryVector([+1, 0, -1, +1, 0])
        assert vec.hamming_weight() == 3

    def test_density(self) -> None:
        vec = TernaryVector([+1, 0, -1, +1, 0])
        assert vec.density() == 0.6

    def test_density_empty(self) -> None:
        vec = TernaryVector([])
        assert vec.density() == 0.0

    def test_balance(self) -> None:
        vec = TernaryVector([+1, +1, -1, 0])
        assert vec.balance() == 0.25  # (2 - 1) / 4 = 0.25

    def test_balance_empty(self) -> None:
        vec = TernaryVector([])
        assert vec.balance() == 0.0

    def test_entropy(self) -> None:
        vec = TernaryVector([+1, -1, 0])
        assert vec.entropy() > 0.0
        # Uniform distribution has max entropy = log2(3) ≈ 1.585
        assert vec.entropy() <= math.log2(3) + 0.01

    def test_entropy_empty(self) -> None:
        vec = TernaryVector([])
        assert vec.entropy() == 0.0

    def test_and_with(self) -> None:
        a = TernaryVector([+1, +1, -1])
        b = TernaryVector([+1, 0, +1])
        result = a.and_with(b)
        assert result.values == [+1, 0, -1]

    def test_or_with(self) -> None:
        a = TernaryVector([+1, 0, -1])
        b = TernaryVector([0, +1, +1])
        result = a.or_with(b)
        assert result.values == [+1, +1, +1]

    def test_not_(self) -> None:
        vec = TernaryVector([+1, 0, -1])
        result = vec.not_()
        assert result.values == [-1, 0, +1]

    def test_majority(self) -> None:
        vec = TernaryVector([+1, +1, -1, 0])
        assert vec.majority() == +1

    def test_consensus(self) -> None:
        vec = TernaryVector([+1, +1, +1, -1])
        assert vec.consensus(threshold=0.6) == +1
        assert vec.consensus(threshold=0.75) == +1  # 3/4 = 75%

    def test_to_string(self) -> None:
        vec = TernaryVector([+1, 0, -1])
        assert vec.to_string() == "PZN"

    def test_from_floats(self) -> None:
        vec = TernaryVector.from_floats([0.5, -0.5, 0.1], threshold=0.0)
        assert vec.values == [+1, -1, +1]

    def test_from_bools(self) -> None:
        vec = TernaryVector.from_bools([True, False, True])
        assert vec.values == [+1, -1, +1]

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            TernaryVector([2])

    def test_len(self) -> None:
        vec = TernaryVector([+1, 0, -1])
        assert len(vec) == 3

    def test_getitem(self) -> None:
        vec = TernaryVector([+1, 0, -1])
        assert vec[0] == +1
        assert vec[1] == 0


class TestTernaryMap:
    def test_classify(self) -> None:
        assert TernaryMap.classify(0.5) == +1
        assert TernaryMap.classify(-0.5) == -1
        assert TernaryMap.classify(0.0) == 0

    def test_classify_with_zscore(self) -> None:
        assert TernaryMap.classify_with_zscore(1.5, 0.0, 1.0, threshold=1.0) == +1
        assert TernaryMap.classify_with_zscore(-1.5, 0.0, 1.0, threshold=1.0) == -1
        assert TernaryMap.classify_with_zscore(0.5, 0.0, 1.0, threshold=1.0) == 0

    def test_classify_with_zscore_zero_std(self) -> None:
        assert TernaryMap.classify_with_zscore(1.0, 0.0, 0.0) == 0

    def test_classify_vector(self) -> None:
        vec = TernaryMap.classify_vector([0.5, -0.5, 0.0])
        assert vec.values == [+1, -1, 0]

    def test_classify_percentile(self) -> None:
        assert TernaryMap.classify_percentile(0.1, 0.25, 0.75) == -1
        assert TernaryMap.classify_percentile(0.5, 0.25, 0.75) == 0
        assert TernaryMap.classify_percentile(0.9, 0.25, 0.75) == +1

    def test_window_classify(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        vec = TernaryMap.window_classify(values, window_size=3, threshold=0.0)
        assert len(vec) == 3
        # Window averages: [0.2, 0.3, 0.4] → all positive → [+1, +1, +1]
        assert vec.values == [+1, +1, +1]

    def test_window_classify_too_short(self) -> None:
        vec = TernaryMap.window_classify([0.1], window_size=5)
        assert len(vec) == 0


class TestTernaryConsensus:
    def test_fleet_vote_consensus(self) -> None:
        votes = {"node1": +1, "node2": +1, "node3": +1}
        result = TernaryConsensus.fleet_vote(votes, threshold=0.6)
        assert result["consensus"] == +1
        assert result["confidence"] == 1.0
        assert result["dissenters"] == []

    def test_fleet_vote_no_consensus(self) -> None:
        votes = {"node1": +1, "node2": -1, "node3": 0}
        result = TernaryConsensus.fleet_vote(votes, threshold=0.6)
        assert result["consensus"] == 0
        assert result["dissenters"] == ["node1", "node2"]

    def test_fleet_vote_empty(self) -> None:
        result = TernaryConsensus.fleet_vote({}, threshold=0.6)
        assert result["consensus"] == 0
        assert result["confidence"] == 0.0

    def test_fleet_vote_counts(self) -> None:
        votes = {"n1": +1, "n2": +1, "n3": -1, "n4": 0}
        result = TernaryConsensus.fleet_vote(votes)
        assert result["pos_count"] == 2
        assert result["neg_count"] == 1
        assert result["zero_count"] == 1
        assert result["total"] == 4

    def test_weighted_vote(self) -> None:
        votes = {"n1": (+1, 2.0), "n2": (+1, 1.0), "n3": (-1, 1.0)}
        result = TernaryConsensus.weighted_vote(votes, threshold=0.6)
        assert result["consensus"] == +1
        assert result["confidence"] == 0.75  # 3.0 / 4.0
        assert result["pos_weight"] == 3.0

    def test_weighted_vote_no_consensus(self) -> None:
        votes = {"n1": (+1, 1.0), "n2": (-1, 1.0), "n3": (0, 1.0)}
        result = TernaryConsensus.weighted_vote(votes, threshold=0.6)
        assert result["consensus"] == 0

    def test_weighted_vote_empty(self) -> None:
        result = TernaryConsensus.weighted_vote({})
        assert result["consensus"] == 0
        assert result["confidence"] == 0.0


class TestTernaryOperator:
    def test_if_then_else_pos(self) -> None:
        assert TernaryOperator.if_then_else(+1, +1, -1) == +1

    def test_if_then_else_neg(self) -> None:
        assert TernaryOperator.if_then_else(-1, +1, -1) == -1

    def test_if_then_else_zero(self) -> None:
        # Conservative: min of both
        assert TernaryOperator.if_then_else(0, +1, -1) == -1
        assert TernaryOperator.if_then_else(0, +1, +1) == +1

    def test_clamp(self) -> None:
        assert TernaryOperator.clamp(+1, -1, 0) == 0
        assert TernaryOperator.clamp(-1, 0, +1) == 0
        assert TernaryOperator.clamp(0, -1, +1) == 0

    def test_switch(self) -> None:
        result = TernaryOperator.switch(
            +1, {+1: "positive", -1: "negative", 0: "neutral"}
        )
        assert result == "positive"

    def test_switch_default(self) -> None:
        result = TernaryOperator.switch(+1, {-1: "negative"})
        assert result is None

    def test_cascade(self) -> None:
        assert TernaryOperator.cascade([0, 0, +1, -1]) == +1
        assert TernaryOperator.cascade([0, 0, -1, +1]) == -1
        assert TernaryOperator.cascade([0, 0, 0]) == 0

    def test_cascade_default(self) -> None:
        assert TernaryOperator.cascade([0, 0], default=-1) == -1


class TestIntegration:
    def test_vector_from_floats_then_consensus(self) -> None:
        floats = [0.5, -0.3, 0.8, -0.1, 0.2]
        vec = TernaryVector.from_floats(floats, threshold=0.0)
        consensus = vec.consensus(threshold=0.5)
        assert consensus in [-1, 0, +1]

    def test_zscore_classification(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        result = TernaryMap.classify_with_zscore(0.5, mean, std, threshold=1.0)
        assert result in [-1, 0, +1]

    def test_ternary_chain(self) -> None:
        # Chain: classify -> vector -> majority -> consensus
        floats = [0.5, 0.6, -0.7, 0.1, -0.2]
        vec = TernaryMap.classify_vector(floats)
        maj = vec.majority()
        con = vec.consensus(threshold=0.5)
        assert maj in [-1, 0, +1]
        assert con in [-1, 0, +1]

    def test_fleet_consensus_with_zeros(self) -> None:
        votes = {"n1": +1, "n2": 0, "n3": +1, "n4": 0}
        result = TernaryConsensus.fleet_vote(votes, threshold=0.5)
        assert result["consensus"] == +1
        # 2 pos out of 2 non-zero = 1.0 confidence
        assert result["confidence"] == 1.0

    def test_window_then_majority(self) -> None:
        values = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]
        vec = TernaryMap.window_classify(values, window_size=2, threshold=0.0)
        if len(vec) > 0:
            maj = vec.majority()
            assert maj in [-1, 0, +1]
