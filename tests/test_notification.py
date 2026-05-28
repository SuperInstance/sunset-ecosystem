"""Tests for notification.py — Alert routing and notification delivery.

Run: python3 -m pytest tests/test_notification.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.notification import NotificationSystem, LogChannel, WebhookChannel, Alert


class TestNotificationSystem:
    def test_create(self):
        ns = NotificationSystem()
        assert ns.channels() == []

    def test_add_channel(self):
        ns = NotificationSystem()
        ns.add_channel("log", LogChannel())
        assert "log" in ns.channels()

    def test_notify(self):
        ns = NotificationSystem()
        webhook = WebhookChannel(url="https://example.com")
        ns.add_channel("webhook", webhook)
        ns.add_rule("warning", ["webhook"])
        sent = ns.notify("cpu_high", severity="warning", message="CPU at 95%")
        assert "webhook" in sent
        assert len(webhook.calls()) == 1

    def test_notify_no_match(self):
        ns = NotificationSystem()
        ns.add_rule("critical", ["log"])
        sent = ns.notify("cpu_high", severity="warning", message="CPU at 95%")
        assert sent == []

    def test_dedup(self):
        ns = NotificationSystem(dedup_window=1.0)
        ns.add_channel("log", LogChannel())
        ns.add_rule("warning", ["log"])
        sent1 = ns.notify("cpu_high", severity="warning", message="CPU at 95%")
        sent2 = ns.notify("cpu_high", severity="warning", message="CPU at 95%")
        assert len(sent1) == 1
        assert len(sent2) == 0  # deduplicated

    def test_dedup_force(self):
        ns = NotificationSystem(dedup_window=1.0)
        ns.add_channel("log", LogChannel())
        ns.add_rule("warning", ["log"])
        ns.notify("cpu_high", severity="warning", message="CPU at 95%")
        sent = ns.notify("cpu_high", severity="warning", message="CPU at 95%", force=True)
        assert len(sent) == 1

    def test_dedup_expires(self):
        ns = NotificationSystem(dedup_window=0.1)
        ns.add_channel("log", LogChannel())
        ns.add_rule("warning", ["log"])
        ns.notify("cpu_high", severity="warning", message="CPU at 95%")
        time.sleep(0.15)
        sent = ns.notify("cpu_high", severity="warning", message="CPU at 95%")
        assert len(sent) == 1

    def test_history(self):
        ns = NotificationSystem()
        ns.add_channel("log", LogChannel())
        ns.add_rule("info", ["log"])
        ns.notify("a", severity="info")
        ns.notify("b", severity="warning")
        ns.notify("c", severity="info")
        assert len(ns.history()) == 2
        assert len(ns.history(severity="info")) == 2

    def test_history_limit(self):
        ns = NotificationSystem()
        ns.add_channel("log", LogChannel())
        ns.add_rule("info", ["log"])
        for i in range(5):
            ns.notify(f"alert-{i}", severity="info")
        assert len(ns.history(limit=2)) == 2

    def test_stats(self):
        ns = NotificationSystem()
        ns.add_channel("log", LogChannel())
        ns.add_channel("webhook", WebhookChannel(url="..."))
        ns.add_rule("info", ["log"])
        stats = ns.stats()
        assert stats["channels"] == 2
        assert stats["rules"] == 1

    def test_repr(self):
        ns = NotificationSystem()
        assert "NotificationSystem" in repr(ns)


class TestLogChannel:
    def test_send(self):
        ch = LogChannel()
        alert = Alert(id="1", name="test", severity="warning", message="boom", timestamp=0.0)
        assert ch.send(alert) is True


class TestWebhookChannel:
    def test_send(self):
        ch = WebhookChannel(url="https://example.com")
        alert = Alert(id="1", name="test", severity="warning", message="boom", timestamp=0.0)
        assert ch.send(alert) is True
        assert len(ch.calls()) == 1
        assert ch.calls()[0].name == "test"
