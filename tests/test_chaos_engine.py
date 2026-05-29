"""Tests for chaos_engine.py — Chaos engineering fault injection.

Run: python3 -m pytest tests/test_chaos_engine.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.chaos_engine import ChaosEngine


class TestChaosEngine:
    def test_create(self):
        chaos = ChaosEngine()
        assert chaos.stats()["enabled"] is True

    def test_add_fault(self):
        chaos = ChaosEngine(seed=42)
        chaos.add_fault("delay", target="users", probability=0.5)
        assert len(chaos.faults_for_target("users")) == 1

    def test_remove_fault(self):
        chaos = ChaosEngine()
        chaos.add_fault("delay", target="users", probability=0.5)
        assert chaos.remove_fault("users", "delay") is True
        assert chaos.remove_fault("users", "missing") is False

    def test_should_trigger_disabled(self):
        chaos = ChaosEngine(enabled=False)
        chaos.add_fault("delay", target="users", probability=1.0)
        assert chaos.should_trigger("users") is False

    def test_should_trigger_high_prob(self):
        chaos = ChaosEngine(seed=42)
        chaos.add_fault("delay", target="users", probability=1.0)
        assert chaos.should_trigger("users") is True

    def test_should_trigger_zero_prob(self):
        chaos = ChaosEngine()
        chaos.add_fault("delay", target="users", probability=0.0)
        assert chaos.should_trigger("users") is False

    def test_get_triggered_fault(self):
        chaos = ChaosEngine(seed=42)
        chaos.add_fault("delay", target="users", probability=1.0, fault_type="latency", params={"delay_sec": 2})
        fault = chaos.get_triggered_fault("users")
        assert fault is not None
        assert fault["name"] == "delay"

    def test_no_fault_for_target(self):
        chaos = ChaosEngine()
        assert chaos.should_trigger("missing") is False
        assert chaos.get_triggered_fault("missing") is None

    def test_targets(self):
        chaos = ChaosEngine()
        chaos.add_fault("a", target="svc-1", probability=0.5)
        chaos.add_fault("b", target="svc-2", probability=0.5)
        assert sorted(chaos.targets()) == ["svc-1", "svc-2"]

    def test_stats(self):
        chaos = ChaosEngine()
        chaos.add_fault("a", target="svc-1", probability=0.5)
        stats = chaos.stats()
        assert stats["targets"] == 1
        assert stats["faults"] == 1
        assert stats["enabled"] is True

    def test_repr(self):
        chaos = ChaosEngine()
        assert "ChaosEngine" in repr(chaos)
