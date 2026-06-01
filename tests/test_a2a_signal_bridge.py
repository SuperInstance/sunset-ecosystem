"""tests/test_a2a_signal_bridge.py — FLUX-A2A Signal Protocol bridge tests."""

import pytest
import math
from fleet.a2a_signal_bridge import (
    ConfidenceScore,
    SignalMessage,
    AgentPosition,
    ConsensusDetector,
    ConsensusType,
    ResolutionType,
    Branch,
    BranchPoint,
    BranchStrategy,
    MergeStrategyType,
    Fork,
    ForkOnComplete,
    ForkConflictMode,
    SharedProgram,
    SharedStateMode,
)


class TestConfidenceScore:
    def test_clamping(self):
        assert ConfidenceScore(1.5).value == 1.0
        assert ConfidenceScore(-0.3).value == 0.0
        assert ConfidenceScore(0.8).value == 0.8

    def test_combine_min(self):
        c1 = ConfidenceScore(0.8)
        c2 = ConfidenceScore(0.6)
        assert c1.combine_min(c2).value == 0.6

    def test_combine_weighted(self):
        c1 = ConfidenceScore(0.8)
        c2 = ConfidenceScore(0.6)
        result = c1.combine_weighted(c2, 0.7, 0.3)
        expected = (0.7 * 0.8 + 0.3 * 0.6) / 1.0
        assert abs(result.value - expected) < 0.001

    def test_combine_geometric(self):
        c1 = ConfidenceScore(0.8)
        c2 = ConfidenceScore(0.6)
        c3 = ConfidenceScore(0.9)
        result = c1.combine_geometric([c2, c3])
        expected = (0.8 * 0.6 * 0.9) ** (1.0 / 3)
        assert abs(result.value - expected) < 0.001

    def test_combine_geometric_zero(self):
        c1 = ConfidenceScore(0.8)
        c2 = ConfidenceScore(0.0)
        assert c1.combine_geometric([c2]).value == 0.0

    def test_bool(self):
        assert bool(ConfidenceScore(0.6))
        assert not bool(ConfidenceScore(0.4))
        assert not bool(ConfidenceScore(0.5))

    def test_repr(self):
        assert "Confidence(0.80)" == repr(ConfidenceScore(0.8))


class TestSignalMessage:
    def test_basic(self):
        msg = SignalMessage(sender="Oracle1", recipient="kimi1", body={"task": "build"})
        assert msg.sender == "Oracle1"
        assert msg.recipient == "kimi1"
        assert msg.body == {"task": "build"}
        assert msg.schema == "https://flux.a2a/signal/v1"
        assert len(msg.message_id) == 8
        assert msg.timestamp

    def test_to_dict(self):
        msg = SignalMessage(sender="A", recipient="B", body={"x": 1}, confidence=ConfidenceScore(0.9))
        d = msg.to_dict()
        assert d["sender"] == "A"
        assert d["confidence"] == 0.9
        assert d["$schema"] == "https://flux.a2a/signal/v1"

    def test_from_dict(self):
        d = {
            "sender": "A",
            "recipient": "B",
            "body": {"x": 1},
            "confidence": 0.8,
            "extra_field": "preserved",
        }
        msg = SignalMessage.from_dict(d)
        assert msg.sender == "A"
        assert msg.confidence.value == 0.8
        assert msg.meta.get("extra_field") == "preserved"

    def test_from_dict_schema(self):
        d = {"$schema": "https://flux.a2a/signal/v2", "sender": "A", "recipient": "B"}
        msg = SignalMessage.from_dict(d)
        assert msg.schema == "https://flux.a2a/signal/v2"


class TestAgentPosition:
    def test_cosine_similarity(self):
        a = AgentPosition("A", [1.0, 0.0])
        b = AgentPosition("B", [1.0, 0.0])
        assert a.cosine_similarity(b) == 1.0

    def test_cosine_orthogonal(self):
        a = AgentPosition("A", [1.0, 0.0])
        b = AgentPosition("B", [0.0, 1.0])
        assert a.cosine_similarity(b) == 0.0

    def test_cosine_opposite(self):
        a = AgentPosition("A", [1.0, 0.0])
        b = AgentPosition("B", [-1.0, 0.0])
        assert a.cosine_similarity(b) == -1.0

    def test_euclidean_distance(self):
        a = AgentPosition("A", [0.0, 0.0])
        b = AgentPosition("B", [3.0, 4.0])
        assert a.euclidean_distance(b) == 5.0

    def test_mismatched_dimensions(self):
        a = AgentPosition("A", [1.0, 0.0])
        b = AgentPosition("B", [1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            a.cosine_similarity(b)


class TestConsensusDetector:
    def test_unanimous(self):
        detector = ConsensusDetector()
        positions = [
            AgentPosition("A", [0.8, 0.2]),
            AgentPosition("B", [0.79, 0.21]),
        ]
        result = detector.detect(positions, threshold=0.7)
        assert result.type == ConsensusType.UNANIMOUS
        assert result.agreement_count == 2
        assert result.total_count == 2

    def test_majority(self):
        detector = ConsensusDetector()
        positions = [
            AgentPosition("A", [0.8, 0.2]),
            AgentPosition("B", [0.75, 0.25]),
            AgentPosition("C", [0.1, 0.9]),
        ]
        result = detector.detect(positions, threshold=0.7)
        assert result.type == ConsensusType.MAJORITY
        assert result.agreement_count == 2
        assert result.total_count == 3

    def test_stalemate(self):
        detector = ConsensusDetector()
        positions = [
            AgentPosition("A", [1.0, 0.0]),
            AgentPosition("B", [0.0, 1.0]),
            AgentPosition("C", [-1.0, 0.0]),
        ]
        result = detector.detect(positions, threshold=0.7)
        assert result.type == ConsensusType.STALEMATE
        assert result.resolution == ResolutionType.VOTE

    def test_single_agent(self):
        detector = ConsensusDetector()
        positions = [AgentPosition("A", [1.0, 0.0])]
        result = detector.detect(positions)
        assert result.type == ConsensusType.UNANIMOUS
        assert result.explanation == "Single agent — unanimous by default"

    def test_empty(self):
        detector = ConsensusDetector()
        result = detector.detect([])
        assert result.type == ConsensusType.STALEMATE
        assert result.confidence.value == 0.0

    def test_supermajority(self):
        detector = ConsensusDetector()
        positions = [
            AgentPosition("A", [0.8, 0.2]),
            AgentPosition("B", [0.79, 0.21]),
            AgentPosition("C", [0.1, 0.9]),
            AgentPosition("D", [0.78, 0.22]),
        ]
        result = detector.detect(positions, threshold=0.7)
        assert result.type == ConsensusType.SUPERMAJORITY
        assert result.agreement_count == 3
        assert result.total_count == 4

    def test_convergence_trend(self):
        detector = ConsensusDetector(similarity_threshold=0.7)
        # First batch: 3 agents, 2 agree (0.67)
        detector.detect([
            AgentPosition("A", [0.8, 0.2]),
            AgentPosition("B", [0.75, 0.25]),
            AgentPosition("C", [0.1, 0.9]),
        ])
        # Second batch: 3 agents, 3 agree (1.0)
        detector.detect([
            AgentPosition("A", [0.8, 0.2]),
            AgentPosition("B", [0.79, 0.21]),
            AgentPosition("C", [0.78, 0.22]),
        ])
        trend = detector.convergence_trend()
        assert trend == "converging"

    def test_diverging_trend(self):
        detector = ConsensusDetector(similarity_threshold=0.7)
        detector.detect([
            AgentPosition("A", [0.8, 0.2]),
            AgentPosition("B", [0.79, 0.21]),
            AgentPosition("C", [0.78, 0.22]),
        ])
        detector.detect([
            AgentPosition("A", [0.8, 0.2]),
            AgentPosition("B", [0.1, 0.9]),
            AgentPosition("C", [0.2, 0.8]),
        ])
        trend = detector.convergence_trend()
        assert trend == "diverging"

    def test_to_dict(self):
        detector = ConsensusDetector()
        positions = [
            AgentPosition("A", [0.8, 0.2]),
            AgentPosition("B", [0.79, 0.21]),
        ]
        result = detector.detect(positions)
        d = result.to_dict()
        assert d["type"] == "unanimous"
        assert d["agreement_count"] == 2
        assert d["total_count"] == 2


class TestBranch:
    def test_add_branch(self):
        bp = BranchPoint()
        b = bp.add_branch("path_a", {"x": 1}, weight=0.5)
        assert b.label == "path_a"
        assert b.weight == 0.5
        assert len(bp.branches) == 1

    def test_merge_best(self):
        bp = BranchPoint(merge_type=MergeStrategyType.BEST)
        bp.add_branch("a", {"x": 1})
        bp.add_branch("b", {"x": 2})
        bp.set_results("a", {"x": 10}, 0.8)
        bp.set_results("b", {"x": 20}, 0.9)
        result = bp.merge()
        assert result["winner"] == "b"
        assert result["confidence"] == 0.9

    def test_merge_vote(self):
        bp = BranchPoint(merge_type=MergeStrategyType.VOTE)
        bp.add_branch("a", {"x": 1})
        bp.add_branch("b", {"x": 1})
        bp.add_branch("c", {"x": 2})
        bp.set_results("a", {"x": 1}, 0.8)
        bp.set_results("b", {"x": 1}, 0.7)
        bp.set_results("c", {"x": 2}, 0.9)
        result = bp.merge()
        assert result["winner"] == "{'x': 1}"
        assert result["votes"]["{'x': 1}"] == 2

    def test_merge_weighted(self):
        bp = BranchPoint(merge_type=MergeStrategyType.WEIGHTED_CONFIDENCE)
        bp.add_branch("a", {"x": 1}, weight=1.0)
        bp.add_branch("b", {"x": 2}, weight=1.0)
        bp.set_results("a", {"x": 10}, 0.8)
        bp.set_results("b", {"x": 20}, 0.6)
        result = bp.merge()
        weighted = result["weighted_result"]
        total = result["total_weight"]
        # a: 1.0 * 0.8 = 0.8, b: 1.0 * 0.6 = 0.6, total = 1.4
        # x: 10 * 0.8/1.4 + 20 * 0.6/1.4 = 5.71 + 8.57 = 14.28
        assert abs(weighted["x"] - (10 * 0.8 / 1.4 + 20 * 0.6 / 1.4)) < 0.1
        assert total == 1.4

    def test_merge_all(self):
        bp = BranchPoint(merge_type=MergeStrategyType.ALL)
        bp.add_branch("a", {"x": 1})
        bp.set_results("a", {"x": 10}, 0.8)
        result = bp.merge()
        assert result["results"]["a"] == {"x": 10}
        assert result["confidences"]["a"] == 0.8

    def test_to_dict(self):
        bp = BranchPoint(strategy=BranchStrategy.PARALLEL)
        bp.add_branch("a", {"x": 1})
        d = bp.to_dict()
        assert d["strategy"] == "parallel"
        assert len(d["branches"]) == 1


class TestFork:
    def test_merge_parent_wins(self):
        f = Fork(
            parent_id="P", child_id="C",
            inherited_state={"x": 10}, child_state={"x": 20},
            conflict_mode=ForkConflictMode.PARENT_WINS,
            result={"x": 30},
        )
        merged = f.merge()
        assert merged["x"] == 20  # child_state wins over inherited

    def test_merge_child_wins(self):
        f = Fork(
            parent_id="P", child_id="C",
            inherited_state={"x": 10}, child_state={"x": 20},
            conflict_mode=ForkConflictMode.CHILD_WINS,
            result={"x": 30},
        )
        merged = f.merge()
        assert merged["x"] == 30  # result wins

    def test_merge_negotiate(self):
        f = Fork(
            parent_id="P", child_id="C",
            inherited_state={"x": 10}, child_state={"x": 20},
            conflict_mode=ForkConflictMode.NEGOTIATE,
            result={"x": 30},
            confidence=ConfidenceScore(0.5),
        )
        merged = f.merge()
        # x: 10 * 0.5 + 30 * 0.5 = 20
        assert merged["x"] == 20.0

    def test_no_result(self):
        f = Fork(parent_id="P", child_id="C", inherited_state={"x": 10})
        merged = f.merge()
        assert merged["x"] == 10

    def test_to_dict(self):
        f = Fork(parent_id="P", child_id="C", confidence=ConfidenceScore(0.8))
        d = f.to_dict()
        assert d["parent_id"] == "P"
        assert d["confidence"] == 0.8


class TestSharedProgram:
    def test_add_cursor(self):
        sp = SharedProgram("test", body=[{}, {}, {}])
        c = sp.add_cursor("A")
        assert c.agent_id == "A"
        assert c.position == 0
        assert len(sp.cursors) == 1

    def test_advance(self):
        sp = SharedProgram("test", body=[{}, {}, {}])
        sp.add_cursor("A")
        assert sp.advance("A")
        assert sp.advance("A")
        assert not sp.advance("A")  # past end

    def test_modify_isolated(self):
        sp = SharedProgram("test", body=[{"x": 1}, {"x": 2}])
        sp.add_cursor("A")
        assert sp.modify("A", 0, {"x": 10})
        assert sp.cursors[0].modifications == 1

    def test_modify_conflict(self):
        sp = SharedProgram("test", body=[{"x": 1}, {"x": 2}], state_mode=SharedStateMode.CONFLICT)
        sp.add_cursor("A")
        sp.add_cursor("B")
        sp.modify("A", 0, {"x": 10})
        # B tries to modify — blocked because A already modified
        assert not sp.modify("B", 0, {"x": 20})
        assert sp.cursors[1].blocked

    def test_modify_merge(self):
        sp = SharedProgram("test", body=[{"x": 1}, {"x": 2}], state_mode=SharedStateMode.MERGE)
        sp.add_cursor("A")
        sp.add_cursor("B")
        assert sp.modify("A", 0, {"x": 10})
        assert sp.modify("B", 0, {"x": 20})
        assert sp.body[0] == {"x": 20}

    def test_to_dict(self):
        sp = SharedProgram("test", body=[{}, {}])
        sp.add_cursor("A")
        d = sp.to_dict()
        assert d["name"] == "test"
        assert d["length"] == 2
        assert len(d["cursors"]) == 1
