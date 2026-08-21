"""Tests for health_check_chain.py — Composite health check with dependencies.

Run: python3 -m pytest tests/test_health_check_chain.py -v --tb=short
"""

from __future__ import annotations

import time

import pytest

from fleet.health_check_chain import (
    ChainStatus,
    HealthCheckChain,
    HealthTier,
    ProbeResult,
)


class TestHealthCheckChain:
    def test_create_empty(self):
        chain = HealthCheckChain()
        status = chain.run()
        assert status.healthy is True
        assert status.probes == []

    def test_single_probe_pass(self):
        chain = HealthCheckChain()
        chain.add("db", lambda: (True, "OK"))
        status = chain.run()
        assert status.healthy is True
        assert len(status.probes) == 1
        assert status.probes[0].name == "db"
        assert status.probes[0].healthy is True

    def test_single_probe_fail(self):
        chain = HealthCheckChain()
        chain.add("db", lambda: (False, "timeout"), tier=HealthTier.CRITICAL)
        status = chain.run()
        assert status.healthy is False
        assert status.critical_healthy is False
        assert status.probes[0].message == "timeout"

    def test_warning_tier(self):
        chain = HealthCheckChain()
        chain.add("cache", lambda: (False, "slow"), tier=HealthTier.WARNING)
        status = chain.run()
        assert status.healthy is False  # warning fails = not fully healthy
        assert status.critical_healthy is True
        assert status.warning_healthy is False

    def test_info_tier_does_not_affect(self):
        chain = HealthCheckChain()
        chain.add("metric", lambda: (False, "stale"), tier=HealthTier.INFO)
        status = chain.run()
        # INFO failures shouldn't mark the chain as unhealthy
        assert status.critical_healthy is True
        assert status.warning_healthy is True

    def test_dependency_chain(self):
        chain = HealthCheckChain()
        chain.add("db", lambda: (True, "OK"))
        chain.add("cache", lambda: (True, "OK"), depends_on=["db"])
        status = chain.run()
        assert status.healthy is True
        assert len(status.probes) == 2

    def test_dependency_failure_blocks_dependent(self):
        chain = HealthCheckChain()
        chain.add("db", lambda: (False, "down"))
        chain.add("cache", lambda: (True, "OK"), depends_on=["db"])
        status = chain.run()
        assert status.healthy is False
        # cache probe may or may not run depending on ordering
        assert "db" in [p.name for p in status.probes]

    def test_multiple_probes_parallel(self):
        chain = HealthCheckChain()
        for i in range(5):
            chain.add(f"probe_{i}", lambda: (True, "ok"))
        status = chain.run()
        assert len(status.probes) == 5
        assert all(p.healthy for p in status.probes)

    def test_latency_tracking(self):
        chain = HealthCheckChain()
        chain.add("slow", lambda: time.sleep(0.05) or (True, "ok"))
        status = chain.run()
        assert status.latency_ms >= 40.0

    def test_remove_probe(self):
        chain = HealthCheckChain()
        chain.add("a", lambda: (True, "ok"))
        chain.add("b", lambda: (True, "ok"), depends_on=["a"])
        assert chain.remove("a") is True
        assert "a" not in chain.probe_names()
        # b's dependency on a should be removed
        status = chain.run()
        assert len(status.probes) == 1

    def test_remove_nonexistent(self):
        chain = HealthCheckChain()
        assert chain.remove("missing") is False

    def test_exception_in_checker(self):
        chain = HealthCheckChain()
        chain.add("bad", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        status = chain.run()
        probe = [p for p in status.probes if p.name == "bad"][0]
        assert probe.healthy is False
        assert "boom" in probe.message

    def test_report(self):
        chain = HealthCheckChain()
        chain.add("a", lambda: (True, "ok"))
        r = chain.report()
        assert r["probes"] == 1
        assert "a" in r["names"]

    def test_blockers_tracked(self):
        chain = HealthCheckChain()
        chain.add("db", lambda: (False, "down"))
        chain.add("app", lambda: (True, "ok"), depends_on=["db"])
        status = chain.run()
        assert "db" in status.blockers

    def test_deadlock_detection(self):
        chain = HealthCheckChain()
        chain.add("a", lambda: (True, "ok"), depends_on=["b"])
        chain.add("b", lambda: (True, "ok"), depends_on=["a"])
        status = chain.run()
        # Should detect deadlock
        assert not status.healthy
        assert len(status.probes) == 2

    def test_retries(self):
        attempts = [0]

        def flaky():
            attempts[0] += 1
            return (attempts[0] >= 2, "retry")

        chain = HealthCheckChain()
        chain.add("flaky", flaky, retries=2)
        status = chain.run()
        probe = [p for p in status.probes if p.name == "flaky"][0]
        assert probe.healthy is True
