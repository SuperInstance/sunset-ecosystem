"""Tests for work_dashboard.py — Fleet work status dashboard.

Run: python3 -m pytest tests/test_work_dashboard.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.work_dashboard import (
    FleetHealthLevel,
    FleetWorkDashboard,
    SubsystemStatus,
)


class TestFleetHealthLevel:
    def test_ordering(self):
        assert FleetHealthLevel.DOWN < FleetHealthLevel.CRITICAL
        assert FleetHealthLevel.CRITICAL < FleetHealthLevel.WARNING
        assert FleetHealthLevel.WARNING < FleetHealthLevel.DEGRADED
        assert FleetHealthLevel.DEGRADED < FleetHealthLevel.HEALTHY


class TestSubsystemStatus:
    def test_create(self):
        s = SubsystemStatus(name="breeding", healthy=True, metrics={"gen": 5})
        assert s.name == "breeding"
        assert s.healthy is True
        assert s.metrics["gen"] == 5


class TestFleetWorkDashboard:
    def test_create_empty(self):
        d = FleetWorkDashboard()
        snap = d.snapshot()
        assert "timestamp" in snap
        assert snap["subsystem_count"] == 0
        assert snap["health_level"] == FleetHealthLevel.DEGRADED.name

    def test_register_and_snapshot(self):
        d = FleetWorkDashboard()
        d.register("test_sys", lambda: {"value": 42})
        snap = d.snapshot()
        assert snap["subsystem_count"] == 1
        assert snap["subsystems"]["test_sys"]["metrics"]["value"] == 42
        assert snap["subsystems"]["test_sys"]["healthy"] is True

    def test_override(self):
        d = FleetWorkDashboard()
        d.register("test_sys", lambda: {"ok": True})
        d.set_override(SubsystemStatus(name="test_sys", healthy=False, error="down"))
        snap = d.snapshot()
        assert snap["subsystems"]["test_sys"]["healthy"] is False
        assert snap["subsystems"]["test_sys"]["error"] == "down"

    def test_health_level_healthy(self):
        d = FleetWorkDashboard()
        d.register("a", lambda: {})
        d.register("b", lambda: {})
        snap = d.snapshot()
        assert snap["health_level"] == FleetHealthLevel.HEALTHY.name
        assert snap["overall_healthy"] is True

    def test_health_level_degraded(self):
        d = FleetWorkDashboard()
        d.register("healthy", lambda: {})
        d.register("broken", lambda: (_ for _ in ()).throw(ValueError("boom")))
        snap = d.snapshot()
        # 1 healthy out of 2 = 50%
        assert snap["health_level"] in (
            FleetHealthLevel.WARNING.name,
            FleetHealthLevel.CRITICAL.name,
        )
        assert snap["overall_healthy"] is False

    def test_history(self):
        d = FleetWorkDashboard()
        d.register("sys", lambda: {"v": 1})
        d.snapshot()
        d.snapshot()
        assert len(d.history) == 2

    def test_trend(self):
        d = FleetWorkDashboard()
        d.register("sys", lambda: {"v": 1})
        d.snapshot()
        d.register("sys", lambda: {"v": 2})
        d.snapshot()
        trend = d.trend(["subsystems", "sys", "metrics", "v"])
        assert trend == [1, 2]

    def test_delta(self):
        d = FleetWorkDashboard()
        d.register("sys", lambda: {"v": 10})
        d.snapshot()
        d.register("sys", lambda: {"v": 15})
        d.snapshot()
        delta = d.delta(["subsystems", "sys", "metrics", "v"])
        assert delta == 5

    def test_alerts(self):
        d = FleetWorkDashboard()
        d.register("breeding", lambda: {"ok": True})
        d.set_override(SubsystemStatus(name="breeding", healthy=False, error="oom"))
        alerts = d.alerts()
        assert len(alerts) == 1
        assert alerts[0]["subsystem"] == "breeding"
        assert "oom" in alerts[0]["message"]

    def test_summary(self):
        d = FleetWorkDashboard()
        d.register("a", lambda: {})
        d.register("b", lambda: {})
        d.set_override(SubsystemStatus(name="b", healthy=False))
        s = d.summary()
        assert s["subsystems"] == 2
        assert s["healthy"] == 1
        assert s["alerts"] == 1

    def test_unregister(self):
        d = FleetWorkDashboard()
        d.register("sys", lambda: {})
        d.unregister("sys")
        snap = d.snapshot()
        assert snap["subsystem_count"] == 0

    def test_max_history(self):
        d = FleetWorkDashboard()
        d.register("sys", lambda: {})
        for _ in range(1010):
            d.snapshot()
        assert len(d.history) <= d._max_history

    def test_trend_none_for_missing_path(self):
        d = FleetWorkDashboard()
        d.register("sys", lambda: {})
        d.snapshot()
        trend = d.trend(["nonexistent", "path"])
        assert trend == [None]

    def test_delta_none_insufficient_history(self):
        d = FleetWorkDashboard()
        d.register("sys", lambda: {"v": 1})
        d.snapshot()
        delta = d.delta(["sys", "metrics", "v"])
        assert delta is None
