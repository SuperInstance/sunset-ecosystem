"""Tests for pathos modules: interaction_log, need_tracker, moment_scorer, trinity_connection."""

from __future__ import annotations

import time

import pytest

from pathos.interaction_log import InteractionLog, InteractionRecord, InteractionSummary
from pathos.need_tracker import NeedTracker, NeedState, Urgency, Satisfaction
from pathos.moment_scorer import MomentScorer, MomentScore
from pathos.trinity_connection import TrinityConnection, TrinityScore


class TestInteractionRecord:
    def test_latency(self):
        r = InteractionRecord(
            query="test",
            timestamp=100.0,
            response_timestamp=102.5,
        )
        assert r.latency == pytest.approx(2.5)

    def test_latency_none(self):
        r = InteractionRecord(query="test")
        assert r.latency is None

    def test_repr(self):
        r = InteractionRecord(query="short", resolved=True)
        assert "resolved=True" in repr(r)


class TestInteractionLog:
    def test_start_and_resolve(self):
        log = InteractionLog()
        rec = log.start_interaction("how do I deploy?")
        log.record_response(rec, "use deploy.sh")
        log.resolve(rec, time_to_solution=5.0)
        assert rec.resolved
        assert rec.time_to_solution == 5.0

    def test_follow_up(self):
        log = InteractionLog()
        rec = log.start_interaction("deploy failing")
        log.mark_follow_up(rec)
        assert rec.needed_follow_up

    def test_summarize_empty(self):
        log = InteractionLog()
        summary = log.summarize()
        assert summary.total_interactions == 0

    def test_summarize_with_data(self):
        log = InteractionLog()
        rec = log.start_interaction("deploy staging server")
        log.record_response(rec, "deployed")
        log.resolve(rec, time_to_solution=10.0)
        summary = log.summarize()
        assert summary.total_interactions == 1
        assert summary.resolved_count == 1
        assert summary.avg_time_to_solution == pytest.approx(10.0)

    def test_recurring_needs_detection(self):
        log = InteractionLog()
        for _ in range(3):
            rec = log.start_interaction("deploy the staging server")
            log.record_response(rec, "done")
            log.resolve(rec)
        summary = log.summarize()
        assert len(summary.recurring_needs) > 0

    def test_chronic_frustrations(self):
        log = InteractionLog()
        for _ in range(3):
            rec = log.start_interaction("deployment failing repeatedly")
            # Don't resolve these
        summary = log.summarize()
        # Should detect chronic frustrations
        assert summary.total_interactions == 3
        assert summary.resolved_count == 0

    def test_detect_patterns(self):
        log = InteractionLog()
        rec = log.start_interaction("test query")
        log.record_response(rec, "response")
        log.resolve(rec, time_to_solution=200.0)  # slow
        patterns = log.detect_patterns()
        assert "slow_resolutions" in patterns

    def test_records_property(self):
        log = InteractionLog()
        log.start_interaction("a")
        log.start_interaction("b")
        assert len(log.records) == 2

    def test_len(self):
        log = InteractionLog()
        log.start_interaction("a")
        log.start_interaction("b")
        assert len(log) == 2

    def test_repr(self):
        log = InteractionLog()
        r = repr(log)
        assert "records=0" in r


class TestNeedTracker:
    def test_initial_state(self):
        tracker = NeedTracker()
        state = tracker.snapshot()
        assert state.urgency == Urgency.LOW
        assert state.frustration == 0.0
        assert state.satisfaction == Satisfaction.UNKNOWN

    def test_positive_signal(self):
        tracker = NeedTracker()
        tracker.record_query("thanks that works great")
        state = tracker.snapshot()
        assert state.satisfaction == Satisfaction.POSITIVE

    def test_negative_signal(self):
        tracker = NeedTracker()
        tracker.record_query("still not working, wrong answer")
        state = tracker.snapshot()
        assert state.satisfaction == Satisfaction.NEGATIVE
        assert state.frustration > 0.0

    def test_repeated_query_escalates(self):
        tracker = NeedTracker()
        tracker.record_query("deploy the server to staging")
        tracker.record_query("deploy the server to staging")  # repeat
        state = tracker.snapshot()
        assert state.frustration > 0.0

    def test_record_response_satisfaction(self):
        tracker = NeedTracker()
        tracker.record_response("done, deployed successfully")
        state = tracker.snapshot()
        assert state.satisfaction == Satisfaction.POSITIVE

    def test_abandonment(self):
        tracker = NeedTracker()
        tracker.record_abandonment()
        state = tracker.snapshot()
        assert state.satisfaction == Satisfaction.NEGATIVE
        assert state.frustration > 0.0

    def test_record_satisfaction(self):
        tracker = NeedTracker()
        tracker.record_satisfaction("awesome thanks")
        state = tracker.snapshot()
        assert state.satisfaction == Satisfaction.POSITIVE

    def test_urgency_critical(self):
        tracker = NeedTracker()
        for _ in range(10):
            tracker.record_query("still not working wrong error again")
        state = tracker.snapshot()
        assert state.urgency in (Urgency.HIGH, Urgency.CRITICAL)

    def test_repr(self):
        tracker = NeedTracker()
        r = repr(tracker)
        assert "queries=0" in r


class TestNeedState:
    def test_repr(self):
        state = NeedState(task="deploy", urgency=Urgency.HIGH, frustration=0.5)
        r = repr(state)
        assert "deploy" in r
        assert "high" in r

    def test_similarity(self):
        assert NeedTracker._is_similar("deploy the server", "deploy the server")
        assert not NeedTracker._is_similar("deploy the server", "eat lunch")


class TestMomentScorer:
    def _make_need_state(self, **kwargs):
        defaults = dict(
            task="test",
            urgency=Urgency.MEDIUM,
            frustration=0.1,
            satisfaction=Satisfaction.NEUTRAL,
            wait_time_seconds=5.0,
        )
        defaults.update(kwargs)
        return NeedState(**defaults)

    def test_instant_response(self):
        scorer = MomentScorer()
        state = self._make_need_state()
        score = scorer.score(state, resolved=True, latency_s=0.5)
        assert score.moment_score >= 0.4
        assert "instant response" in score.reason

    def test_slow_response(self):
        scorer = MomentScorer()
        state = self._make_need_state()
        score = scorer.score(state, resolved=False, latency_s=120.0)
        assert score.moment_score < 0.3

    def test_invisibility_bonus(self):
        scorer = MomentScorer()
        state = self._make_need_state()
        visible = scorer.score(state, resolved=True, latency_s=1.0, human_aware_of_agent=True)
        invisible = scorer.score(state, resolved=True, latency_s=1.0, human_aware_of_agent=False)
        assert invisible.invisibility_bonus > visible.invisibility_bonus

    def test_unresolved(self):
        scorer = MomentScorer()
        state = self._make_need_state()
        score = scorer.score(state, resolved=False, latency_s=2.0)
        assert score.resolution_score == 0.0

    def test_frustration_penalty(self):
        scorer = MomentScorer()
        calm = self._make_need_state(frustration=0.0)
        angry = self._make_need_state(frustration=0.8)
        calm_score = scorer.score(calm, resolved=True, latency_s=1.0)
        angry_score = scorer.score(angry, resolved=True, latency_s=1.0)
        assert calm_score.moment_score > angry_score.moment_score

    def test_score_interactions(self):
        scorer = MomentScorer()
        state = self._make_need_state()
        score = scorer.score_interactions(
            need_state=state,
            total_interactions=10,
            resolved_count=8,
            avg_latency_s=3.0,
        )
        assert isinstance(score, MomentScore)
        assert score.moment_score > 0.0

    def test_repr(self):
        scorer = MomentScorer()
        assert "MomentScorer" in repr(scorer)


class TestTrinityConnection:
    def test_perfect_agent(self):
        tc = TrinityConnection()
        score = tc.score(
            human_facing=True,
            resolves_needs=True,
            output_used_directly=True,
            requires_human_intervention=False,
            automates_away_work=True,
        )
        assert score.composite > 0.7
        assert score.solves_human_problems > 0.8

    def test_useless_agent(self):
        tc = TrinityConnection()
        score = tc.score(
            human_facing=False,
            resolves_needs=False,
            output_used_directly=False,
            requires_human_intervention=True,
            adds_context_switches=True,
        )
        assert score.composite < 0.5

    def test_score_from_history(self):
        tc = TrinityConnection()
        score = tc.score_from_history(
            total_interactions=20,
            resolved_count=18,
            follow_up_count=2,
            human_initiated_count=15,
            avg_latency_s=3.0,
        )
        assert score.composite > 0.5

    def test_score_from_history_no_interactions(self):
        tc = TrinityConnection()
        score = tc.score_from_history(
            total_interactions=0,
            resolved_count=0,
            follow_up_count=0,
            human_initiated_count=0,
        )
        assert "no interactions" in score.reason

    def test_high_error_rate(self):
        tc = TrinityConnection()
        score = tc.score(error_rate=0.5, resolves_needs=True)
        assert score.solves_human_problems < 0.8

    def test_repr(self):
        tc = TrinityConnection()
        assert "TrinityConnection" in repr(tc)
