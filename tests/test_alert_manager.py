"""Tests for alert_manager.py — Threshold-based alerts.

Run: python3 -m pytest tests/test_alert_manager.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.alert_manager import AlertManager


class TestAlertManager:
    def test_create(self):
        mgr = AlertManager()
        assert len(mgr.list_rules()) == 0

    def test_add_and_check(self):
        mgr = AlertManager()
        mgr.add_rule("cpu_high", threshold=90)
        assert mgr.check("cpu_high", 95) is True

    def test_check_no_alert(self):
        mgr = AlertManager()
        mgr.add_rule("cpu_high", threshold=90)
        assert mgr.check("cpu_high", 85) is False

    def test_cooldown_dedup(self):
        mgr = AlertManager(default_cooldown=0.5)
        mgr.add_rule("cpu_high", threshold=90)
        assert mgr.check("cpu_high", 95) is True
        assert mgr.check("cpu_high", 95) is False
        time.sleep(0.51)
        assert mgr.check("cpu_high", 95) is True

    def test_different_ops(self):
        mgr = AlertManager()
        mgr.add_rule("low_mem", threshold=10, op="lt")
        assert mgr.check("low_mem", 5) is True
        assert mgr.check("low_mem", 15) is False

        mgr.add_rule("exact", threshold=42, op="eq")
        assert mgr.check("exact", 42) is True
        assert mgr.check("exact", 43) is False

    def test_check_all(self):
        mgr = AlertManager()
        mgr.add_rule("a", threshold=10)
        mgr.add_rule("b", threshold=20)
        triggered = mgr.check_all({"a": 15, "b": 25})
        assert "a" in triggered
        assert "b" in triggered

    def test_remove_rule(self):
        mgr = AlertManager()
        mgr.add_rule("x", threshold=1)
        assert mgr.remove_rule("x") is True
        assert mgr.remove_rule("missing") is False

    def test_invalid_op(self):
        mgr = AlertManager()
        with pytest.raises(ValueError):
            mgr.add_rule("x", threshold=1, op="invalid")

    def test_alert_history(self):
        mgr = AlertManager()
        mgr.add_rule("cpu", threshold=90)
        mgr.check("cpu", 95)
        history = mgr.alert_history()
        assert len(history) == 1
        assert history[0]["rule"] == "cpu"

    def test_stats(self):
        mgr = AlertManager()
        mgr.add_rule("cpu", threshold=90)
        mgr.check("cpu", 95)
        stats = mgr.stats()
        assert stats["rules"] == 1
        assert stats["total_alerts"] == 1

    def test_clear_history(self):
        mgr = AlertManager()
        mgr.add_rule("x", threshold=1)
        mgr.check("x", 2)
        mgr.clear_history()
        assert len(mgr.alert_history()) == 0

    def test_repr(self):
        mgr = AlertManager()
        mgr.add_rule("x", threshold=1)
        assert "AlertManager" in repr(mgr)
