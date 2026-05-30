"""Tests for fleet/sim_real_degradation.py — SIM/REAL degradation stack."""
from __future__ import annotations

import time

import pytest

from fleet.sim_real_degradation import (
    DataSource,
    DegradationLevel,
    DegradationState,
    FleetDegradationMonitor,
    SimRealDegradationStack,
)


# ---------------------------------------------------------------------------
# 1. DataSource
# ---------------------------------------------------------------------------

class TestDataSource:
    def test_health_score_perfect(self):
        src = DataSource(name="sensor1", confidence=1.0, latency_ms=0.0)
        assert src.health_score() == pytest.approx(1.0)

    def test_health_score_stale(self):
        src = DataSource(name="sensor1", confidence=1.0)
        src.last_update = time.time() - 10.0  # 10 seconds ago
        assert src.health_score() == 0.0

    def test_health_score_zero_confidence(self):
        src = DataSource(name="sensor1", confidence=0.0, latency_ms=0.0)
        assert src.health_score() == 0.0

    def test_is_stale(self):
        src = DataSource(name="sensor1", stale_threshold_ms=1000.0)
        assert not src.is_stale()
        src.last_update = time.time() - 2.0
        assert src.is_stale()


# ---------------------------------------------------------------------------
# 2. SimRealDegradationStack — transitions
# ---------------------------------------------------------------------------

class TestSimRealDegradationStack:
    def test_starts_green(self):
        stack = SimRealDegradationStack("test")
        assert stack.level == DegradationLevel.GREEN
        assert stack.is_real is True
        assert stack.is_sim is False

    def test_degrade_to_yellow(self):
        stack = SimRealDegradationStack("test")
        stack.register_source("s1", confidence=0.5)
        stack.register_source("s2", confidence=0.5)
        # Health = 0.5 * 1.0 (latency 0) + 0.5 * 1.0 / 2 = 0.5, below 0.7
        level = stack.tick()
        assert level == DegradationLevel.YELLOW

    def test_degrade_to_red(self):
        stack = SimRealDegradationStack("test")
        stack.register_source("s1", confidence=0.1)
        stack.register_source("s2", confidence=0.1)
        level = stack.tick()
        assert level == DegradationLevel.RED
        assert stack.is_sim is True

    def test_sim_override(self):
        stack = SimRealDegradationStack("test")
        stack.register_source("s1", confidence=1.0)
        stack.force_sim(True)
        level = stack.tick()
        assert level == DegradationLevel.RED

    def test_select_value_green(self):
        stack = SimRealDegradationStack("test")
        stack.register_source("s1", confidence=1.0)
        stack.tick()
        assert stack.select_value(42, 0) == 42

    def test_select_value_red(self):
        stack = SimRealDegradationStack("test")
        stack.register_source("s1", confidence=0.1)
        stack.tick()
        assert stack.select_value(42, 0) == 0

    def test_select_value_yellow_blend(self):
        stack = SimRealDegradationStack("test")
        stack.register_source("s1", confidence=0.7)
        stack.register_source("s2", confidence=0.7)
        stack.tick()  # Health ≈ 0.7, boundary
        # Yellow level → blended value
        val = stack.select_value(100.0, 0.0)
        assert 0.0 <= val <= 100.0

    def test_select_value_non_numeric(self):
        stack = SimRealDegradationStack("test")
        stack.register_source("s1", confidence=0.1)
        stack.tick()
        # RED → sim value
        assert stack.select_value("real", "sim") == "sim"

    def test_transition_callback(self):
        stack = SimRealDegradationStack("test")
        transitions = []
        stack.on_transition(lambda old, new: transitions.append((old, new)))
        stack.register_source("s1", confidence=0.1)
        stack.tick()
        assert len(transitions) == 1
        assert transitions[0][0] == DegradationLevel.GREEN
        assert transitions[0][1] == DegradationLevel.RED

    def test_hysteresis_recovery(self):
        stack = SimRealDegradationStack("test")
        stack.register_source("s1", confidence=0.1)
        stack.tick()
        assert stack.level == DegradationLevel.RED

        # Improve health but not sustained
        stack.update_source("s1", latency_ms=0.0, confidence=1.0)
        # Need sustained period — single tick not enough
        stack.tick()
        # Still RED due to hysteresis
        assert stack.level == DegradationLevel.RED

    def test_sustained_recovery(self):
        stack = SimRealDegradationStack("test")
        stack.SUSTAINED_SECONDS = 0.1  # Short for testing
        stack.register_source("s1", confidence=0.1)
        stack.tick()
        assert stack.level == DegradationLevel.RED

        stack.update_source("s1", latency_ms=0.0, confidence=1.0)
        time.sleep(0.15)
        stack.tick()
        assert stack.level == DegradationLevel.GREEN

    def test_repr(self):
        stack = SimRealDegradationStack("test")
        stack.register_source("s1")
        r = repr(stack)
        assert "test" in r
        assert "GREEN" in r

    def test_update_source_not_found(self):
        stack = SimRealDegradationStack("test")
        with pytest.raises(KeyError):
            stack.update_source("missing", latency_ms=10.0)

    def test_degrade_to_red_skips_yellow(self):
        """Very low health should jump directly to RED."""
        stack = SimRealDegradationStack("test")
        stack.register_source("s1", confidence=0.0)
        stack.register_source("s2", confidence=0.0)
        level = stack.tick()
        assert level == DegradationLevel.RED

    def test_multiple_sources_mixed_health(self):
        """Some healthy, some unhealthy sources → weighted health."""
        stack = SimRealDegradationStack("test")
        stack.register_source("good", confidence=1.0)
        stack.register_source("bad", confidence=0.2)
        level = stack.tick()
        # Average health ≈ 0.6 → YELLOW
        assert level == DegradationLevel.YELLOW


# ---------------------------------------------------------------------------
# 3. FleetDegradationMonitor
# ---------------------------------------------------------------------------

class TestFleetDegradationMonitor:
    def test_register_subsystem(self):
        monitor = FleetDegradationMonitor()
        stack = monitor.register_subsystem("breeding")
        assert isinstance(stack, SimRealDegradationStack)
        assert "breeding" in monitor._stacks

    def test_tick_all(self):
        monitor = FleetDegradationMonitor()
        s1 = monitor.register_subsystem("breeding")
        s2 = monitor.register_subsystem("gossip")
        s1.register_source("src1", confidence=1.0)
        s2.register_source("src2", confidence=0.1)

        levels = monitor.tick_all()
        assert levels["breeding"] == DegradationLevel.GREEN
        assert levels["gossip"] == DegradationLevel.RED

    def test_fleet_health(self):
        monitor = FleetDegradationMonitor()
        s1 = monitor.register_subsystem("good")
        s2 = monitor.register_subsystem("bad")
        s1.register_source("src1", confidence=1.0)
        s2.register_source("src2", confidence=0.0)

        monitor.tick_all()
        health = monitor.fleet_health()
        assert health == pytest.approx(0.5, abs=0.01)

    def test_degraded_subsystems(self):
        monitor = FleetDegradationMonitor()
        s1 = monitor.register_subsystem("good")
        s2 = monitor.register_subsystem("bad")
        s1.register_source("src1", confidence=1.0)
        s2.register_source("src2", confidence=0.0)

        monitor.tick_all()
        degraded = monitor.degraded_subsystems()
        assert "bad" in degraded
        assert "good" not in degraded

    def test_empty_monitor_health(self):
        monitor = FleetDegradationMonitor()
        assert monitor.fleet_health() == 1.0

    def test_repr(self):
        monitor = FleetDegradationMonitor()
        monitor.register_subsystem("s1")
        assert "1 subsystems" in repr(monitor)


# ---------------------------------------------------------------------------
# 4. DegradationState
# ---------------------------------------------------------------------------

class TestDegradationState:
    def test_overall_health_empty(self):
        state = DegradationState(level=DegradationLevel.GREEN)
        assert state.overall_health() == 1.0

    def test_overall_health_mixed(self):
        state = DegradationState()
        state.sources["a"] = DataSource(name="a", confidence=1.0)
        state.sources["b"] = DataSource(name="b", confidence=0.0)
        assert state.overall_health() == pytest.approx(0.5, abs=0.01)

    def test_active_sources(self):
        state = DegradationState()
        state.sources["fresh"] = DataSource(name="fresh", last_update=time.time())
        state.sources["stale"] = DataSource(
            name="stale", last_update=time.time() - 10.0, stale_threshold_ms=1000.0
        )
        active = state.active_sources()
        assert "fresh" in active
        assert "stale" not in active
