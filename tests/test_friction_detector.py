"""Tests for fleet/friction_detector.py — Automated UX research and friction detection.

Covers the full SDA loop:
  SENSE → DECIDE → ACT → VALIDATE

Includes behavior observation, friction clustering, map generation, fix
suggestion, severity scoring, trend detection, event emission, academy
integration, and edge cases.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from fleet.friction_detector import (
    BehaviorSample,
    FixSuggestionEngine,
    FixValidator,
    FrictionCategory,
    FrictionDetector,
    FrictionMap,
    FrictionPoint,
    Severity,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def detector():
    """Fresh FrictionDetector with default thresholds."""
    return FrictionDetector(node_id="alpha")


@pytest.fixture
def academy_bridge():
    """Mock academy bridge with 10 pre-populated cohort findings."""
    bridge = MagicMock()
    bridge.get_friction_points.return_value = [
        {
            "agent": "greenhorn",
            "finding": "boot_camp_path_discrepancy",
            "severity": "high",
        },
        {"agent": "greenhorn", "finding": "plato_identity_crisis", "severity": "high"},
        {
            "agent": "junior_dev",
            "finding": "room_creation_impossible",
            "severity": "medium",
        },
        {"agent": "junior_dev", "finding": "no_build_schema", "severity": "medium"},
        {
            "agent": "architect",
            "finding": "zero_authentication",
            "severity": "critical",
        },
        {
            "agent": "architect",
            "finding": "tile_count_discrepancy",
            "severity": "medium",
        },
        {"agent": "human_proxy", "finding": "no_web_ui", "severity": "high"},
        {"agent": "task_agent", "finding": "dual_submit_endpoints", "severity": "low"},
        {"agent": "captain", "finding": "no_broadcast_endpoints", "severity": "high"},
        {"agent": "captain", "finding": "no_global_fleet_map", "severity": "high"},
    ]
    return bridge


# ═══════════════════════════════════════════════════════════════
# 1. Behavior observation
# ═══════════════════════════════════════════════════════════════


class TestBehaviorObservation:
    def test_sense_creates_sample(self, detector):
        sample = detector.sense(endpoint="/api/tiles", status_code=200, latency=0.1)
        assert isinstance(sample, BehaviorSample)
        assert sample.endpoint == "/api/tiles"
        assert sample.status_code == 200
        assert sample.latency == 0.1
        assert sample.retries == 0

    def test_sense_with_retries(self, detector):
        detector.sense(endpoint="/api/tiles", status_code=401, latency=0.2, retries=3)
        detector.sense(endpoint="/api/tiles", status_code=401, latency=0.3, retries=2)
        assert len(detector._samples) == 2

    def test_sense_timeout_logging(self, detector):
        detector.sense(endpoint="/api/rooms", status_code=200, latency=5.0, retries=0)
        point = detector._analyze_sample(detector._samples[0])
        assert point is not None
        assert point.category == FrictionCategory.PERFORMANCE
        assert point.evidence in ("high_latency", "timeout")

    def test_retry_detection(self, detector):
        detector.sense(endpoint="/api/rooms", status_code=200, latency=0.5, retries=3)
        point = detector._analyze_sample(detector._samples[0])
        assert point is not None
        assert point.category == FrictionCategory.API
        assert point.evidence == "retry_storm"

    def test_error_counting_401(self, detector):
        for _ in range(5):
            detector.sense(
                endpoint="/api/auth", status_code=401, latency=0.1, retries=0
            )
        detector.tick()
        assert len(detector._friction_points) == 1
        key = list(detector._friction_points.keys())[0]
        assert detector._friction_points[key].sample_count == 5


# ═══════════════════════════════════════════════════════════════
# 2. Friction clustering
# ═══════════════════════════════════════════════════════════════


class TestFrictionClustering:
    def test_auth_cluster(self, detector):
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        detector.tick()
        points = list(detector._friction_points.values())
        assert any(p.category == FrictionCategory.AUTH for p in points)

    def test_ui_cluster_404(self, detector):
        detector.sense(endpoint="/dashboard", status_code=404, latency=0.1, retries=0)
        detector.tick()
        points = list(detector._friction_points.values())
        assert any(p.category == FrictionCategory.UI for p in points)

    def test_schema_cluster_400(self, detector):
        detector.sense(endpoint="/api/create", status_code=400, latency=0.1, retries=0)
        detector.tick()
        points = list(detector._friction_points.values())
        assert any(p.category == FrictionCategory.SCHEMA for p in points)

    def test_routing_cluster_503(self, detector):
        detector.sense(
            endpoint="/api/upstream", status_code=503, latency=0.1, retries=0
        )
        detector.tick()
        points = list(detector._friction_points.values())
        assert any(p.category == FrictionCategory.ROUTING for p in points)

    def test_performance_cluster(self, detector):
        detector.sense(endpoint="/api/slow", status_code=200, latency=3.0, retries=0)
        detector.tick()
        points = list(detector._friction_points.values())
        assert any(p.category == FrictionCategory.PERFORMANCE for p in points)

    def test_api_cluster_retry_storm(self, detector):
        detector.sense(
            endpoint="/api/unstable", status_code=200, latency=0.5, retries=5
        )
        detector.tick()
        points = list(detector._friction_points.values())
        assert any(p.category == FrictionCategory.API for p in points)

    def test_all_categories_present(self, detector):
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        detector.sense(endpoint="/dashboard", status_code=404, latency=0.1, retries=0)
        detector.sense(endpoint="/api/create", status_code=400, latency=0.1, retries=0)
        detector.sense(
            endpoint="/api/upstream", status_code=503, latency=0.1, retries=0
        )
        detector.sense(endpoint="/api/slow", status_code=200, latency=3.0, retries=0)
        detector.sense(
            endpoint="/api/unstable", status_code=500, latency=0.5, retries=0
        )
        detector.tick()
        categories = {p.category for p in detector._friction_points.values()}
        assert FrictionCategory.AUTH in categories
        assert FrictionCategory.UI in categories
        assert FrictionCategory.SCHEMA in categories
        assert FrictionCategory.ROUTING in categories
        assert FrictionCategory.PERFORMANCE in categories
        assert FrictionCategory.API in categories


# ═══════════════════════════════════════════════════════════════
# 3. Friction map generation
# ═══════════════════════════════════════════════════════════════


class TestFrictionMapGeneration:
    def test_map_has_points(self, detector):
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        fmap = detector.tick()
        assert isinstance(fmap, FrictionMap)
        assert len(fmap.points) == 1
        assert fmap.node_id == "alpha"

    def test_map_by_category(self, detector):
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        detector.sense(endpoint="/dashboard", status_code=404, latency=0.1, retries=0)
        fmap = detector.tick()
        auth_points = fmap.by_category(FrictionCategory.AUTH)
        assert len(auth_points) == 1
        ui_points = fmap.by_category(FrictionCategory.UI)
        assert len(ui_points) == 1

    def test_map_by_severity(self, detector):
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=3)
        fmap = detector.tick()
        critical = fmap.by_severity(Severity.CRITICAL)
        assert len(critical) == 1

    def test_map_counts(self, detector):
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=3)
        detector.sense(
            endpoint="/api/upstream", status_code=503, latency=0.1, retries=0
        )
        fmap = detector.tick()
        assert fmap.critical_count() == 1
        assert fmap.high_count() == 1
        assert fmap.trend_score() == 7  # CRITICAL(4) + HIGH(3)

    def test_map_to_dict(self, detector):
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=3)
        fmap = detector.tick()
        d = fmap.to_dict()
        assert d["node_id"] == "alpha"
        assert d["point_count"] == 1
        assert d["critical"] == 1
        assert d["points"][0]["category"] == "AUTH"


# ═══════════════════════════════════════════════════════════════
# 4. Fix suggestion
# ═══════════════════════════════════════════════════════════════


class TestFixSuggestion:
    def test_suggest_auth_401(self):
        fix = FixSuggestionEngine.suggest(FrictionCategory.AUTH, "401")
        assert "authentication" in fix.lower() or "token" in fix.lower()

    def test_suggest_ui_no_web_ui(self):
        fix = FixSuggestionEngine.suggest(FrictionCategory.UI, "no_web_ui")
        assert "web UI" in fix or "dashboard" in fix

    def test_suggest_schema_no_build_schema(self):
        fix = FixSuggestionEngine.suggest(FrictionCategory.SCHEMA, "no_build_schema")
        assert "schema" in fix.lower()

    def test_suggest_routing_503(self):
        fix = FixSuggestionEngine.suggest(FrictionCategory.ROUTING, "503")
        assert "circuit breaker" in fix.lower() or "retry" in fix.lower()

    def test_suggest_performance_timeout(self):
        fix = FixSuggestionEngine.suggest(FrictionCategory.PERFORMANCE, "timeout")
        assert "async" in fix.lower() or "optimize" in fix.lower()

    def test_suggest_api_broadcast(self):
        fix = FixSuggestionEngine.suggest(
            FrictionCategory.API, "no_broadcast_endpoints"
        )
        assert "WebSocket" in fix or "SSE" in fix

    def test_suggest_unknown_default(self):
        fix = FixSuggestionEngine.suggest(FrictionCategory.UNKNOWN, "anything")
        assert "investigate" in fix.lower()

    def test_act_populates_suggested_fix(self, detector):
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        points = detector.decide()
        detector.act(points)
        assert points[0].suggested_fix != ""


# ═══════════════════════════════════════════════════════════════
# 5. Fix validation
# ═══════════════════════════════════════════════════════════════


class TestFixValidation:
    def test_validate_good_fix(self, detector):
        point = FrictionPoint(
            category=FrictionCategory.AUTH,
            severity=Severity.HIGH,
            evidence="401",
            suggested_fix="Add authentication middleware.",
            endpoint="/api/auth",
        )
        validator = FixValidator()
        assert validator.validate_fix(point) is True

    def test_validate_bad_fix(self, detector):
        point = FrictionPoint(
            category=FrictionCategory.AUTH,
            severity=Severity.HIGH,
            evidence="401",
            suggested_fix="Do nothing and hope it goes away.",
            endpoint="/api/auth",
        )
        validator = FixValidator()
        assert validator.validate_fix(point) is False

    def test_validate_empty_fix(self, detector):
        point = FrictionPoint(
            category=FrictionCategory.AUTH,
            severity=Severity.HIGH,
            evidence="401",
            suggested_fix="",
            endpoint="/api/auth",
        )
        validator = FixValidator()
        assert validator.validate_fix(point) is False

    def test_replay_cohort_with_bridge(self, academy_bridge):
        validator = FixValidator(academy_bridge=academy_bridge)
        result = validator.replay_cohort("architect")
        assert result["success"] is True
        assert result["findings"] == 2
        assert result["resolved"] > 0

    def test_replay_cohort_no_bridge(self):
        validator = FixValidator(academy_bridge=None)
        result = validator.replay_cohort("architect")
        assert result["success"] is False
        assert "error" in result

    def test_replay_history(self, academy_bridge):
        validator = FixValidator(academy_bridge=academy_bridge)
        validator.replay_cohort("architect")
        validator.replay_cohort("captain")
        history = validator.get_replay_history()
        assert len(history) == 2
        assert history[0]["cohort"] == "architect"
        assert history[1]["cohort"] == "captain"


# ═══════════════════════════════════════════════════════════════
# 6. Severity scoring
# ═══════════════════════════════════════════════════════════════


class TestSeverityScoring:
    def test_401_no_retry_is_high(self, detector):
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        detector.tick()
        points = list(detector._friction_points.values())
        assert points[0].severity == Severity.HIGH

    def test_401_with_retries_is_critical(self, detector):
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=3)
        detector.tick()
        points = list(detector._friction_points.values())
        assert points[0].severity == Severity.CRITICAL

    def test_503_is_high(self, detector):
        detector.sense(
            endpoint="/api/upstream", status_code=503, latency=0.1, retries=0
        )
        detector.tick()
        points = list(detector._friction_points.values())
        assert points[0].severity == Severity.HIGH

    def test_502_is_medium(self, detector):
        detector.sense(
            endpoint="/api/upstream", status_code=502, latency=0.1, retries=0
        )
        detector.tick()
        points = list(detector._friction_points.values())
        assert points[0].severity == Severity.MEDIUM

    def test_high_latency_is_medium(self, detector):
        detector.sense(endpoint="/api/slow", status_code=200, latency=3.0, retries=0)
        detector.tick()
        points = list(detector._friction_points.values())
        assert points[0].severity == Severity.MEDIUM

    def test_timeout_is_high(self, detector):
        detector.sense(endpoint="/api/slow", status_code=200, latency=10.0, retries=0)
        detector.tick()
        points = list(detector._friction_points.values())
        assert points[0].severity == Severity.HIGH
        assert points[0].evidence == "timeout"

    def test_retry_storm_medium(self, detector):
        detector.sense(
            endpoint="/api/unstable", status_code=200, latency=0.5, retries=3
        )
        detector.tick()
        points = list(detector._friction_points.values())
        assert points[0].severity == Severity.MEDIUM

    def test_retry_storm_high(self, detector):
        detector.sense(
            endpoint="/api/unstable", status_code=200, latency=0.5, retries=6
        )
        detector.tick()
        points = list(detector._friction_points.values())
        assert points[0].severity == Severity.HIGH


# ═══════════════════════════════════════════════════════════════
# 7. Trend detection
# ═══════════════════════════════════════════════════════════════


class TestTrendDetection:
    def test_empty_trend_is_stable(self, detector):
        assert detector.get_trend() == "stable"

    def test_single_tick_stable(self, detector):
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        detector.tick()
        assert detector.get_trend() == "stable"

    def test_increasing_trend(self, detector):
        for i in range(5):
            detector.sense(
                endpoint=f"/api/auth{i}", status_code=401, latency=0.2, retries=3
            )
            detector.tick()
        assert detector.get_trend() == "increasing"

    def test_decreasing_trend(self, detector):
        # Seed with high friction
        for _ in range(5):
            detector.sense(
                endpoint="/api/auth", status_code=401, latency=0.2, retries=3
            )
            detector.tick()
        # Mark all as fixed
        for key in list(detector._friction_points.keys()):
            detector.apply_fix(key)
        detector.tick()  # Recalculate trend with fixed points
        # The trend is based on severity scores; with fixes applied, the score drops
        assert detector.get_trend() == "decreasing"

    def test_stable_trend(self, detector):
        for _ in range(5):
            detector.sense(
                endpoint="/api/auth", status_code=401, latency=0.2, retries=0
            )
            detector.tick()
        # Score stays the same each tick (same friction point, merged)
        assert detector.get_trend() == "stable"


# ═══════════════════════════════════════════════════════════════
# 8. Event emission
# ═══════════════════════════════════════════════════════════════


class TestEventEmission:
    def test_event_callback_fires(self, detector):
        events: List[FrictionPoint] = []
        detector.register_event_callback(lambda p: events.append(p))
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        detector.tick()
        assert len(events) == 1
        assert events[0].category == FrictionCategory.AUTH

    def test_multiple_callbacks(self, detector):
        events1: List[FrictionPoint] = []
        events2: List[FrictionPoint] = []
        detector.register_event_callback(lambda p: events1.append(p))
        detector.register_event_callback(lambda p: events2.append(p))
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        detector.tick()
        assert len(events1) == 1
        assert len(events2) == 1

    def test_callback_exception_isolated(self, detector):
        bad_called = [False]
        good_called = [False]

        def bad_cb(p):
            bad_called[0] = True
            raise RuntimeError("boom")

        def good_cb(p):
            good_called[0] = True

        detector.register_event_callback(bad_cb)
        detector.register_event_callback(good_cb)
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        detector.tick()
        assert bad_called[0] is True
        assert good_called[0] is True


# ═══════════════════════════════════════════════════════════════
# 9. Academy integration
# ═══════════════════════════════════════════════════════════════


class TestAcademyIntegration:
    def test_load_academy_findings(self, detector, academy_bridge):
        detector.academy_bridge = academy_bridge
        loaded = detector.load_academy_findings()
        assert len(loaded) == 10
        categories = {p.category for p in loaded}
        assert FrictionCategory.AUTH in categories
        assert FrictionCategory.UI in categories
        assert FrictionCategory.SCHEMA in categories
        assert FrictionCategory.API in categories

    def test_load_academy_no_bridge(self, detector):
        loaded = detector.load_academy_findings()
        assert loaded == []

    def test_compare_with_academy(self, detector, academy_bridge):
        detector.academy_bridge = academy_bridge
        # Seed live friction
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        detector.tick()
        result = detector.compare_with_academy()
        assert "live_only" in result
        assert "academy_only" in result
        assert "overlap" in result
        assert result["live_points"] >= 1
        assert result["academy_points"] == 10

    def test_academy_findings_merge(self, detector, academy_bridge):
        detector.academy_bridge = academy_bridge
        # Simulate a live finding that matches an academy finding
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        detector.tick()
        loaded = detector.load_academy_findings()
        # The zero_authentication finding merges with the live AUTH point
        # but the key differs (endpoint vs evidence), so both exist
        auth_points = [
            p
            for p in detector._friction_points.values()
            if p.category == FrictionCategory.AUTH
        ]
        assert len(auth_points) >= 1


# ═══════════════════════════════════════════════════════════════
# 10. Edge cases
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_no_friction(self, detector):
        detector.sense(endpoint="/api/health", status_code=200, latency=0.05, retries=0)
        fmap = detector.tick()
        assert len(fmap.points) == 0
        assert fmap.trend_score() == 0

    def test_all_friction(self, detector):
        statuses = [401, 404, 400, 502, 503, 500]
        for status in statuses:
            detector.sense(
                endpoint="/api/mixed", status_code=status, latency=0.1, retries=0
            )
        fmap = detector.tick()
        assert len(fmap.points) == 6
        assert fmap.trend_score() > 0

    def test_unknown_friction_type(self, detector):
        # 418 I'm a teapot — not explicitly mapped, falls to UNKNOWN catch-all
        detector.sense(endpoint="/api/teapot", status_code=418, latency=0.1, retries=0)
        fmap = detector.tick()
        points = list(detector._friction_points.values())
        assert len(points) == 1
        assert points[0].category == FrictionCategory.UNKNOWN

    def test_merge_same_friction(self, detector):
        for _ in range(10):
            detector.sense(
                endpoint="/api/auth", status_code=401, latency=0.2, retries=0
            )
        detector.tick()
        assert len(detector._friction_points) == 1
        point = list(detector._friction_points.values())[0]
        assert point.sample_count == 10

    def test_apply_fix_missing_key(self, detector):
        assert detector.apply_fix("AUTH:/api/auth:401") is False

    def test_apply_fix_success(self, detector):
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        detector.tick()
        key = list(detector._friction_points.keys())[0]
        assert detector.apply_fix(key) is True
        assert detector._friction_points[key].fixed is True

    def test_decide_clears_samples(self, detector):
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        detector.decide()
        assert len(detector._samples) == 0

    def test_validate_without_bridge(self, detector):
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        points = detector.decide()
        detector.act(points)
        results = detector.validate(points)
        assert all(v is False for v in results.values())

    def test_map_fixed_count(self, detector):
        detector.sense(endpoint="/api/auth", status_code=401, latency=0.2, retries=0)
        detector.tick()
        key = list(detector._friction_points.keys())[0]
        detector.apply_fix(key)
        fmap = detector.tick()
        assert fmap.fixed_count() == 1

    def test_empty_map_to_dict(self, detector):
        fmap = detector.tick()
        d = fmap.to_dict()
        assert d["point_count"] == 0
        assert d["trend_score"] == 0

    def test_friction_point_defaults(self):
        point = FrictionPoint(
            category=FrictionCategory.API,
            severity=Severity.MEDIUM,
            evidence="test",
            suggested_fix="fix it",
        )
        assert point.sample_count == 1
        assert point.fixed is False
        assert point.fix_timestamp is None
        assert point.endpoint is None
