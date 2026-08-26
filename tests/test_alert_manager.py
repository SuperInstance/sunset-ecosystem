"""Tests for alert_manager.py — Alert routing, deduplication, suppression.

Run: python3 -m pytest tests/test_alert_manager.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.alert_manager import AlertManager


class TestAlertManager:
    def test_create(self):
        alerts = AlertManager()
        assert alerts.stats()["channels"] == 0

    def test_register_channel(self):
        alerts = AlertManager()
        received = []
        alerts.register_channel("slack", lambda msg: received.append(msg))
        assert "slack" in alerts.channels()

    def test_unregister_channel(self):
        alerts = AlertManager()
        alerts.register_channel("slack", lambda msg: None)
        assert alerts.unregister_channel("slack") is True
        assert alerts.unregister_channel("missing") is False

    def test_send(self):
        alerts = AlertManager()
        received = []
        alerts.register_channel("slack", lambda msg: received.append(msg))
        assert alerts.send("CPU high", severity="warning", channel="slack") is True
        assert len(received) == 1
        assert received[0]["message"] == "CPU high"
        assert received[0]["severity"] == "warning"

    def test_send_all_channels(self):
        alerts = AlertManager()
        count = [0]
        alerts.register_channel("a", lambda msg: count.__setitem__(0, count[0] + 1))
        alerts.register_channel("b", lambda msg: count.__setitem__(0, count[0] + 1))
        alerts.send("test")
        assert count[0] == 2

    def test_deduplication(self):
        alerts = AlertManager(suppression_sec=60)
        received = []
        alerts.register_channel("slack", lambda msg: received.append(msg))
        assert alerts.send("CPU high", severity="warning", channel="slack") is True
        assert alerts.send("CPU high", severity="warning", channel="slack") is False
        assert len(received) == 1

    def test_dedup_cleared(self):
        alerts = AlertManager(suppression_sec=0)
        received = []
        alerts.register_channel("slack", lambda msg: received.append(msg))
        assert alerts.send("CPU high", channel="slack") is True
        assert alerts.send("CPU high", channel="slack") is True
        assert len(received) == 2

    def test_clear_suppression(self):
        alerts = AlertManager(suppression_sec=60)
        alerts.send("CPU high", severity="warning")
        assert alerts.send("CPU high", severity="warning") is False
        alerts.clear_suppression("CPU high", severity="warning")
        assert alerts.send("CPU high", severity="warning") is True

    def test_clear_all_suppression(self):
        alerts = AlertManager(suppression_sec=60)
        alerts.send("a")
        alerts.send("b")
        assert alerts.stats()["active_suppressions"] == 2
        alerts.clear_all_suppression()
        assert alerts.stats()["active_suppressions"] == 0

    def test_is_suppressed(self):
        alerts = AlertManager(suppression_sec=60)
        alerts.send("CPU high", severity="warning")
        assert alerts.is_suppressed("CPU high", severity="warning") is True
        assert alerts.is_suppressed("Memory high", severity="warning") is False

    def test_send_no_channel(self):
        alerts = AlertManager()
        # No channels registered
        assert alerts.send("test") is True
        assert alerts.stats()["sent"] == 1

    def test_stats(self):
        alerts = AlertManager()
        alerts.send("a")
        alerts.send("a")  # dedup
        stats = alerts.stats()
        assert stats["sent"] == 1
        assert stats["suppressed"] == 1

    def test_repr(self):
        alerts = AlertManager()
        assert "AlertManager" in repr(alerts)
