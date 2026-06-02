"""Tests for fleet.a2a_signal_bridge Discuss and Reflect primitives."""

from fleet.a2a_signal_bridge import (
    AgentPosition,
    ConfidenceScore,
    ConsensusType,
    DiscourseMode,
    Discussion,
    ResolutionType,
    SelfAssessor,
    Turn,
)


class TestDiscussion:
    """Structured multi-agent discourse."""

    def test_empty_discussion_no_consensus(self):
        d = Discussion(topic="test", mode=DiscourseMode.DEBATE)
        result = d.consensus()
        assert result.type == ConsensusType.STALEMATE
        assert result.confidence.value == 0.0

    def test_debate_reaches_majority(self):
        d = Discussion(topic="deploy v2", mode=DiscourseMode.DEBATE)
        d.add_round([
            Turn("A", AgentPosition("A", [0.8, 0.2]), ConfidenceScore(0.9), "yes"),
            Turn("B", AgentPosition("B", [0.75, 0.25]), ConfidenceScore(0.8), "yes"),
            Turn("C", AgentPosition("C", [0.1, 0.9]), ConfidenceScore(0.7), "no"),
        ])
        result = d.consensus()
        assert result.type == ConsensusType.MAJORITY
        assert result.agreement_count == 2
        assert result.total_count == 3

    def test_brainstorm_unanimous(self):
        d = Discussion(topic="name the fleet", mode=DiscourseMode.BRAINSTORM)
        d.add_round([
            Turn("A", AgentPosition("A", [1.0, 0.0]), ConfidenceScore(0.9)),
            Turn("B", AgentPosition("B", [0.95, 0.05]), ConfidenceScore(0.8)),
        ])
        result = d.consensus()
        assert result.type == ConsensusType.UNANIMOUS

    def test_convergence_trend(self):
        d = Discussion(topic="refactor", mode=DiscourseMode.REVIEW, max_rounds=3)
        # Round 1: split
        d.add_round([
            Turn("A", AgentPosition("A", [0.9, 0.1]), ConfidenceScore(0.8)),
            Turn("B", AgentPosition("B", [0.1, 0.9]), ConfidenceScore(0.8)),
        ])
        # Round 2: converging
        d.add_round([
            Turn("A", AgentPosition("A", [0.7, 0.3]), ConfidenceScore(0.8)),
            Turn("B", AgentPosition("B", [0.6, 0.4]), ConfidenceScore(0.8)),
        ])
        trend = d.convergence_trend()
        assert trend in ("converging", "stable")

    def test_to_dict(self):
        d = Discussion(topic="x", mode=DiscourseMode.NEGOTIATE)
        d.add_round([Turn("A", AgentPosition("A", [1.0, 0.0]), ConfidenceScore(0.9))])
        doc = d.to_dict()
        assert doc["topic"] == "x"
        assert doc["mode"] == "negotiate"
        assert len(doc["rounds"]) == 1
        assert "consensus" in doc
        assert "trend" in doc


class TestSelfAssessor:
    """Agent meta-cognition and self-assessment."""

    def test_critical_confidence(self):
        a = SelfAssessor(target_confidence=0.8)
        r = a.assess("kimi1", ConfidenceScore(0.2), {"strategy": "default"})
        assert r.assessment == "critical"
        assert "escalate_to_trusted_peer" in r.adjustments
        assert "reduce_scope" in r.adjustments
        assert "switch_to_conservative" in r.adjustments

    def test_below_target(self):
        a = SelfAssessor(target_confidence=0.8)
        r = a.assess("kimi1", ConfidenceScore(0.6), {"strategy": "default"})
        assert r.assessment == "below_target"
        assert "gather_more_evidence" in r.adjustments

    def test_healthy(self):
        a = SelfAssessor(target_confidence=0.8)
        r = a.assess("kimi1", ConfidenceScore(0.85), {"strategy": "default"})
        assert r.assessment == "healthy"
        assert len(r.adjustments) == 0

    def test_overconfident(self):
        a = SelfAssessor(target_confidence=0.8)
        r = a.assess("kimi1", ConfidenceScore(0.99), {"strategy": "aggressive"})
        assert r.assessment == "overconfident"
        assert "red_team_check" in r.adjustments

    def test_trend_improving(self):
        a = SelfAssessor(target_confidence=0.8)
        a.assess("A", ConfidenceScore(0.3), {})
        a.assess("A", ConfidenceScore(0.4), {})
        a.assess("A", ConfidenceScore(0.6), {})
        a.assess("A", ConfidenceScore(0.9), {})
        assert a.trend() == "improving"

    def test_trend_declining(self):
        a = SelfAssessor(target_confidence=0.8)
        a.assess("A", ConfidenceScore(0.9), {})
        a.assess("A", ConfidenceScore(0.8), {})
        a.assess("A", ConfidenceScore(0.5), {})
        a.assess("A", ConfidenceScore(0.3), {})
        assert a.trend() == "declining"

    def test_trend_stable(self):
        a = SelfAssessor(target_confidence=0.8)
        a.assess("A", ConfidenceScore(0.7), {})
        a.assess("A", ConfidenceScore(0.72), {})
        assert a.trend() == "stable"

    def test_to_dict(self):
        a = SelfAssessor(target_confidence=0.8)
        a.assess("A", ConfidenceScore(0.7), {})
        doc = a.to_dict()
        assert doc["target_confidence"] == 0.8
        assert "trend" in doc
        assert len(doc["history"]) == 1
