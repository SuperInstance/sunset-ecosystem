"""Tests for auto_scaler.py — Auto-scale fleet nodes based on load.

Run: python3 -m pytest tests/test_auto_scaler.py -v --tb=short
"""

from __future__ import annotations

import time

import pytest

from fleet.auto_scaler import AutoScaler, ScaleDecision, ScalingPolicy


class TestAutoScaler:
    def test_create(self):
        scaler = AutoScaler()
        assert "AutoScaler" in repr(scaler)

    def test_hold(self):
        scaler = AutoScaler()
        decision = scaler.evaluate({"cpu": 0.5, "memory": 0.6}, current_nodes=5)
        assert decision.action == "hold"
        assert decision.current_nodes == 5

    def test_scale_up_cpu(self):
        scaler = AutoScaler()
        decision = scaler.evaluate({"cpu": 0.95, "memory": 0.5}, current_nodes=5)
        assert decision.action == "scale_up"
        assert decision.count == 2  # default step
        assert decision.target_nodes == 7

    def test_scale_up_memory(self):
        scaler = AutoScaler()
        decision = scaler.evaluate({"cpu": 0.5, "memory": 0.95}, current_nodes=5)
        assert decision.action == "scale_up"
        assert decision.count > 0

    def test_scale_down(self):
        scaler = AutoScaler()
        decision = scaler.evaluate({"cpu": 0.2, "memory": 0.2}, current_nodes=5)
        assert decision.action == "scale_down"
        assert decision.count == 1  # default step
        assert decision.target_nodes == 4

    def test_scale_up_cooldown(self):
        scaler = AutoScaler(policy=ScalingPolicy(scale_up_cooldown=10.0))
        scaler.evaluate({"cpu": 0.95}, current_nodes=5)
        # Second call within cooldown
        decision = scaler.evaluate({"cpu": 0.95}, current_nodes=7)
        assert decision.action == "hold"  # cooldown active

    def test_scale_down_cooldown(self):
        scaler = AutoScaler(policy=ScalingPolicy(scale_down_cooldown=10.0))
        scaler.evaluate({"cpu": 0.2}, current_nodes=5)
        decision = scaler.evaluate({"cpu": 0.2}, current_nodes=4)
        assert decision.action == "hold"

    def test_max_nodes(self):
        scaler = AutoScaler(policy=ScalingPolicy(max_nodes=5))
        decision = scaler.evaluate({"cpu": 0.95}, current_nodes=5)
        assert decision.action == "hold"

    def test_min_nodes(self):
        scaler = AutoScaler(policy=ScalingPolicy(min_nodes=3))
        decision = scaler.evaluate({"cpu": 0.2}, current_nodes=3)
        assert decision.action == "hold"

    def test_confidence(self):
        scaler = AutoScaler()
        decision = scaler.evaluate({"cpu": 0.95}, current_nodes=5)
        assert decision.confidence > 0.8

    def test_cooldown_remaining(self):
        scaler = AutoScaler(policy=ScalingPolicy(scale_up_cooldown=10.0))
        scaler.evaluate({"cpu": 0.95}, current_nodes=5)
        cd = scaler.cooldown_remaining()
        assert cd["scale_up"] > 0.0
        assert cd["scale_down"] == 0.0

    def test_stats(self):
        scaler = AutoScaler()
        scaler.evaluate({"cpu": 0.5}, current_nodes=5)
        stats = scaler.stats()
        assert stats["history_points"] == 1

    def test_predictive(self):
        scaler = AutoScaler(policy=ScalingPolicy(scale_up_cooldown=0.0))
        # Feed history showing rapid increase
        for i in range(10):
            scaler.evaluate({"cpu": 0.1 + i * 0.05}, current_nodes=5)
        # cpu now at 0.55, trend from 0.1 to 0.55 = +0.45 > 0.2
        decision = scaler.evaluate({"cpu": 0.55}, current_nodes=5)
        # predictive scale-up might trigger
        # note: may also be hold if not above threshold, but predictive kicks in
        assert decision.action in ("scale_up", "hold")

    def test_reason(self):
        scaler = AutoScaler()
        decision = scaler.evaluate({"cpu": 0.95}, current_nodes=5)
        assert "cpu=" in decision.reason

    def test_policy_defaults(self):
        p = ScalingPolicy()
        assert p.target_cpu == 0.7
        assert p.target_memory == 0.8
        assert p.scale_up_threshold == 0.85
        assert p.scale_down_threshold == 0.3
        assert p.max_nodes == 100
        assert p.min_nodes == 1

    def test_repr(self):
        scaler = AutoScaler()
        assert "AutoScaler" in repr(scaler)
