import pytest
from fleet.notification_system import (
    Alert,
    AlertSeverity,
    LogChannel,
    NotificationSystem,
    WebhookChannel,
)


class TestAlert:
    def test_to_dict(self):
        a = Alert(
            alert_id="a1",
            title="Test",
            message="msg",
            severity=AlertSeverity.ERROR,
            source="svc",
            timestamp=0.0,
        )
        d = a.to_dict()
        assert d["title"] == "Test"
        assert d["severity"] == "error"
        assert d["acknowledged"] is False


class TestLogChannel:
    def test_send(self):
        ch = LogChannel()
        a = Alert("a1", "T", "M", AlertSeverity.INFO, "src", 0.0)
        assert ch.send(a) is True
        assert len(ch.get_entries()) == 1


class TestWebhookChannel:
    def test_send(self):
        ch = WebhookChannel("web", {"url": "http://example.com"})
        a = Alert("a1", "T", "M", AlertSeverity.INFO, "src", 0.0)
        assert ch.send(a) is True


class TestNotificationSystem:
    def test_init(self):
        ns = NotificationSystem()
        assert ns.fleet_node_id == "default"
        assert ns.channels == {}

    def test_add_channel(self):
        ns = NotificationSystem()
        ch = LogChannel()
        ns.add_channel(ch)
        assert "log" in ns.channels

    def test_send_no_rules(self):
        ns = NotificationSystem()
        ch = LogChannel()
        ns.add_channel(ch)
        alert = ns.send("title", "message", AlertSeverity.WARNING)
        assert alert.title == "title"
        assert alert.severity == AlertSeverity.WARNING
        assert len(ch.get_entries()) == 1

    def test_send_with_rule(self):
        ns = NotificationSystem()
        ch = LogChannel()
        ns.add_channel(ch)
        ns.add_rule(severity="error", channel_name="log")
        alert = ns.send("title", "message", AlertSeverity.ERROR)
        assert len(ch.get_entries()) == 1

    def test_send_rule_no_match(self):
        ns = NotificationSystem()
        ch = LogChannel()
        ns.add_channel(ch)
        ns.add_rule(severity="critical", channel_name="log")
        ns.send("title", "message", AlertSeverity.WARNING)
        # No rules match, no channels get it
        assert len(ch.get_entries()) == 0

    def test_acknowledge(self):
        ns = NotificationSystem()
        alert = ns.send("title", "message")
        assert ns.acknowledge(alert.alert_id) is True
        assert alert.acknowledged is True

    def test_acknowledge_missing(self):
        ns = NotificationSystem()
        assert ns.acknowledge("missing") is False

    def test_get_alerts(self):
        ns = NotificationSystem()
        ns.send("a", "msg", AlertSeverity.ERROR)
        ns.send("b", "msg", AlertSeverity.INFO)
        errors = ns.get_alerts(severity="error")
        assert len(errors) == 1
        assert errors[0].title == "a"

    def test_get_alerts_acknowledged(self):
        ns = NotificationSystem()
        alert = ns.send("a", "msg")
        ns.acknowledge(alert.alert_id)
        acked = ns.get_alerts(acknowledged=True)
        assert len(acked) == 1

    def test_get_stats(self):
        ns = NotificationSystem()
        ns.send("a", "msg", AlertSeverity.ERROR)
        ns.send("b", "msg", AlertSeverity.WARNING)
        stats = ns.get_stats()
        assert stats["total_alerts"] == 2
        assert stats["by_severity"]["error"] == 1

    def test_export_json(self):
        ns = NotificationSystem()
        ns.send("a", "msg")
        j = ns.export_json()
        assert "a" in j
        assert "stats" in j

    def test_to_dict(self):
        ns = NotificationSystem()
        ns.send("a", "msg")
        d = ns.to_dict()
        assert d["stats"]["total_alerts"] == 1
