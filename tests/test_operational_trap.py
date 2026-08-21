"""Tests for fleet/operational_trap.py.

Covers the trap base class, built-in implementations, registry, and dashboard.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from fleet.operational_trap import (
    AgentCrashTrap,
    FluxViolationTrap,
    OperationalTrap,
    TrapDashboard,
    TrapRegistry,
    TrapResult,
    TrapSeverity,
    ThermalTrap,
)
from swarm.flux_gating import FluxGatingChecker, FluxGatingConfig, FluxCheckResult
from swarm.thermal import DeviceType, ThermalBudget


# ── helpers ─────────────────────────────────────────────


class _DummyTrap(OperationalTrap):
    """Test-only trap that always returns a fixed result."""

    def __init__(self, result: TrapResult | None = None, **kwargs):
        super().__init__(name="dummy", **kwargs)
        self._fixed = result

    def check(self) -> TrapResult | None:
        return self._fixed


class _CountingTrap(OperationalTrap):
    """Trap that counts how many times it has been checked."""

    def __init__(self, fire_every: int = 1, **kwargs):
        super().__init__(name="counting", **kwargs)
        self._counter = 0
        self._fire_every = fire_every

    def check(self) -> TrapResult | None:
        self._counter += 1
        if self._counter % self._fire_every == 0:
            return TrapResult(
                condition="count",
                severity=TrapSeverity.INFO,
                message=f"fired {self._counter}",
            )
        return None


# ── base class ──────────────────────────────────────────


def test_base_check_raises_not_implemented():
    """Abstract check() must raise NotImplementedError."""

    class _ProxyTrap(OperationalTrap):
        def __init__(self):
            super().__init__(name="proxy")

        def check(self):
            return super().check()

    trap = _ProxyTrap()
    with pytest.raises(NotImplementedError):
        trap.check()


def test_escalation_routing_info():
    """INFO severity triggers logging but no callback by default."""
    calls = []

    class _CbTrap(_DummyTrap):
        def notify(self, result):
            calls.append(result.severity)
            super().notify(result)

    trap = _CbTrap(
        result=TrapResult(
            condition="x",
            severity=TrapSeverity.INFO,
            message="info msg",
        ),
        notify_channels=["log", "callback"],
    )
    trap.set_callback(lambda r: calls.append("callback"))
    trap.run()
    # INFO only logs; notify (and therefore callback) is not invoked
    assert calls == []


def test_escalation_routing_warning():
    """WARNING severity triggers callback channel."""
    calls = []

    class _CbTrap(_DummyTrap):
        def notify(self, result):
            calls.append(result.severity)
            super().notify(result)

    trap = _CbTrap(
        result=TrapResult(
            condition="x",
            severity=TrapSeverity.WARNING,
            message="warn msg",
        ),
        notify_channels=["log", "callback"],
    )
    trap.set_callback(lambda r: calls.append("callback"))
    trap.run()
    assert TrapSeverity.WARNING in calls
    assert "callback" in calls


def test_escalation_routing_critical():
    """CRITICAL severity triggers callback + A2A if configured."""
    calls = []

    class _CbTrap(_DummyTrap):
        def notify(self, result):
            calls.append(result.severity)
            super().notify(result)

    trap = _CbTrap(
        result=TrapResult(
            condition="x",
            severity=TrapSeverity.CRITICAL,
            message="crit msg",
        ),
        notify_channels=["log", "callback", "a2a"],
    )
    trap.set_callback(lambda r: calls.append("callback"))
    trap.set_a2a_callback(lambda r: calls.append("a2a"))
    trap.run()
    assert TrapSeverity.CRITICAL in calls
    assert "callback" in calls
    assert "a2a" in calls


# ── rate limiting ───────────────────────────────────────


def test_rate_limit_suppresses_duplicate():
    """Same key within interval must be suppressed."""
    trap = _DummyTrap(
        result=TrapResult(
            condition="dup",
            severity=TrapSeverity.WARNING,
            message="dup",
        ),
        rate_limit_interval=10.0,
    )
    r1 = trap.run()
    assert r1 is not None
    r2 = trap.run()
    assert r2 is None  # suppressed


def test_rate_limit_allows_after_interval(monkeypatch):
    """After the interval elapses, the same key fires again."""
    trap = _DummyTrap(
        result=TrapResult(
            condition="dup",
            severity=TrapSeverity.WARNING,
            message="dup",
        ),
        rate_limit_interval=1.0,
    )
    r1 = trap.run()
    assert r1 is not None

    # Fake time forward by 2 seconds
    fake = time.monotonic() + 2.0
    monkeypatch.setattr(time, "monotonic", lambda: fake)
    # Also need to advance the stored timestamp so rate_limit thinks interval passed
    with trap._lock:
        trap._rate_limit_store["dup"] = fake - 2.0

    r2 = trap.run()
    assert r2 is not None


def test_rate_limit_different_keys_not_suppressed():
    """Different condition keys should not interfere."""

    class _MultiKeyTrap(OperationalTrap):
        def __init__(self):
            super().__init__(name="multi", rate_limit_interval=10.0)
            self.idx = 0

        def check(self):
            self.idx += 1
            return TrapResult(
                condition=f"cond{self.idx}",
                severity=TrapSeverity.WARNING,
                message="m",
            )

    trap = _MultiKeyTrap()
    r1 = trap.run()
    r2 = trap.run()
    assert r1 is not None
    assert r2 is not None


# ── ThermalTrap ─────────────────────────────────────────


def test_thermal_trap_detects_overcommit():
    """ThermalTrap fires when current_agents exceeds max_agents."""
    budget = ThermalBudget({DeviceType.GPU: 2})
    budget.allocate("a1", DeviceType.GPU)
    budget.allocate("a2", DeviceType.GPU)
    # Force overcommit by bumping current_agents directly
    budget._devices[DeviceType.GPU].current_agents = 3

    trap = ThermalTrap(budget=budget, threshold=0.95)
    result = trap.check()
    assert result is not None
    assert result.condition == "thermal_overcommit"
    assert result.severity == TrapSeverity.CRITICAL
    assert "gpu" in result.message.lower()


def test_thermal_trap_respects_threshold():
    """ThermalTrap fires when utilization exceeds threshold even without overcommit."""
    budget = ThermalBudget({DeviceType.CPU: 10})
    for i in range(10):
        budget.allocate(f"a{i}", DeviceType.CPU)

    trap = ThermalTrap(budget=budget, threshold=0.8)
    result = trap.check()
    assert result is not None
    assert result.condition == "thermal_overcommit"
    assert result.severity == TrapSeverity.WARNING  # 100% util but not > max


def test_thermal_trap_no_fire_when_healthy():
    """Healthy budget produces no result."""
    budget = ThermalBudget({DeviceType.GPU: 5})
    budget.allocate("a1", DeviceType.GPU)

    trap = ThermalTrap(budget=budget, threshold=0.95)
    result = trap.check()
    assert result is None


# ── FluxViolationTrap ───────────────────────────────────


def test_flux_violation_trap_detects_breach():
    """FluxViolationTrap surfaces recent results with critical severity."""
    config = FluxGatingConfig()
    checker = FluxGatingChecker(config)

    def _get_results():
        return [FluxCheckResult(passed=False, score=0.9, violations={"bounds": 0.9})]

    trap = FluxViolationTrap(checker=checker, get_recent_results=_get_results)
    result = trap.check()
    assert result is not None
    assert result.condition == "flux_constraint_breach"
    assert result.severity == TrapSeverity.CRITICAL
    assert result.metadata["critical_count"] == 1


def test_flux_violation_trap_no_fire_when_clean():
    """Empty results list means no trap result."""
    config = FluxGatingConfig()
    checker = FluxGatingChecker(config)

    trap = FluxViolationTrap(checker=checker, get_recent_results=lambda: [])
    result = trap.check()
    assert result is None


def test_flux_violation_trap_warning_only():
    """Only warning-level violations produce WARNING severity."""
    config = FluxGatingConfig()
    checker = FluxGatingChecker(config)

    def _get_results():
        return [FluxCheckResult(passed=False, score=0.4, violations={"l2_norm": 0.4})]

    trap = FluxViolationTrap(checker=checker, get_recent_results=_get_results)
    result = trap.check()
    assert result is not None
    assert result.severity == TrapSeverity.WARNING
    assert result.metadata["warning_count"] == 1


# ── AgentCrashTrap ──────────────────────────────────────


def test_agent_crash_trap_detects_missing_process():
    """Missing PID for an expected agent triggers CRITICAL."""

    def _get_pids():
        return {"agent_1": 1234, "agent_2": None}

    trap = AgentCrashTrap(
        get_agent_pids=_get_pids,
        expected_agents=["agent_1", "agent_2"],
    )
    result = trap.check()
    assert result is not None
    assert result.condition == "agent_crash"
    assert result.severity == TrapSeverity.CRITICAL
    assert "agent_2" in result.metadata["missing_agents"]


def test_agent_crash_trap_detects_zero_pid():
    """PID of 0 is treated as dead."""

    def _get_pids():
        return {"agent_1": 0}

    trap = AgentCrashTrap(
        get_agent_pids=_get_pids,
        expected_agents=["agent_1"],
    )
    result = trap.check()
    assert result is not None
    assert result.condition == "agent_crash"


def test_agent_crash_trap_no_fire_when_all_healthy():
    """All expected agents have valid PIDs — no trap."""

    def _get_pids():
        return {"agent_1": 1234, "agent_2": 5678}

    trap = AgentCrashTrap(
        get_agent_pids=_get_pids,
        expected_agents=["agent_1", "agent_2"],
    )
    result = trap.check()
    assert result is None


def test_agent_crash_trap_unexpected_dead_agent():
    """An agent in the mapping with dead PID but not in expected list is still flagged."""

    def _get_pids():
        return {"orphan": None}

    trap = AgentCrashTrap(get_agent_pids=_get_pids)
    result = trap.check()
    assert result is not None
    assert "orphan" in result.metadata["missing_agents"]


# ── TrapRegistry ────────────────────────────────────────


def test_registry_runs_all_registered_traps():
    """run_all() executes every trap and returns fired results."""
    reg = TrapRegistry()
    t1 = _DummyTrap(
        result=TrapResult(
            condition="a",
            severity=TrapSeverity.INFO,
            message="a",
        )
    )
    t2 = _DummyTrap(result=None)
    reg.register(t1)
    reg.register(t2)

    results = reg.run_all()
    assert len(results) == 1
    assert results[0].condition == "a"


def test_registry_unregister():
    """Removing a trap stops it from running."""
    reg = TrapRegistry()
    t1 = _DummyTrap(
        result=TrapResult(
            condition="a",
            severity=TrapSeverity.INFO,
            message="a",
        )
    )
    reg.register(t1)
    reg.unregister(t1)
    results = reg.run_all()
    assert results == []


def test_registry_thread_safety():
    """Concurrent register / run_all should not crash."""
    reg = TrapRegistry()
    errors = []

    def adder():
        try:
            for _ in range(50):
                reg.register(_DummyTrap())
        except Exception as e:
            errors.append(e)

    def runner():
        try:
            for _ in range(50):
                reg.run_all()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=adder) for _ in range(3)] + [
        threading.Thread(target=runner) for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


# ── TrapDashboard ─────────────────────────────────────────


def test_dashboard_shows_correct_status():
    """Dashboard aggregates registry state into a flat snapshot."""
    reg = TrapRegistry()
    reg.register(
        _DummyTrap(
            result=TrapResult(
                condition="c1",
                severity=TrapSeverity.CRITICAL,
                message="crit",
            )
        )
    )
    reg.register(
        _DummyTrap(
            result=TrapResult(
                condition="w1",
                severity=TrapSeverity.WARNING,
                message="warn",
            )
        )
    )
    reg.run_all()

    dash = TrapDashboard(reg)
    status = dash.get_status()

    assert status["summary"]["trap_count"] == 2
    assert status["summary"]["critical_count"] == 1
    assert status["summary"]["warning_count"] == 1
    assert status["summary"]["info_count"] == 0
    assert len(status["alerts"]) == 2
    assert len(status["traps"]) == 2
    assert "timestamp" in status


def test_dashboard_empty_registry():
    """Dashboard with no traps reports zero counts."""
    reg = TrapRegistry()
    dash = TrapDashboard(reg)
    status = dash.get_status()
    assert status["summary"]["trap_count"] == 0
    assert status["summary"]["total_checks"] == 0
    assert status["alerts"] == []


# ── integration-style ─────────────────────────────────────


def test_full_pipeline_from_trap_to_dashboard():
    """End-to-end: registry → run → dashboard reflects state."""
    reg = TrapRegistry()
    reg.register(ThermalTrap(ThermalBudget({DeviceType.CPU: 1})))
    reg.register(AgentCrashTrap(get_agent_pids=lambda: {}))

    # Both should be silent with empty state
    assert reg.run_all() == []

    # Overcommit CPU
    budget = ThermalBudget({DeviceType.CPU: 1})
    budget.allocate("a1", DeviceType.CPU)
    budget.allocate("a2", DeviceType.CPU)
    reg2 = TrapRegistry()
    reg2.register(ThermalTrap(budget=budget))
    results = reg2.run_all()
    assert len(results) == 1

    dash = TrapDashboard(reg2)
    st = dash.get_status()
    assert st["summary"]["total_fired"] == 1
    assert st["alerts"][0]["condition"] == "thermal_overcommit"
