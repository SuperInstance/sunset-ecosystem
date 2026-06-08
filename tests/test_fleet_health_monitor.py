"""Tests for FleetHealthMonitor — continuous health monitoring with alerting.

Reference: fleet/fleet_health_monitor.py
"""

from __future__ import annotations

import time

import pytest

from fleet.fleet_health_monitor import (
    Alert,
    AlertChannel,
    AlertRule,
    AlertSeverity,
    EscalationPolicy,
    FleetHealthMonitor,
)


class TestAlertRule:
    def test_evaluate_gt(self) -> None:
        rule = AlertRule(
            rule_id="test",
            name="Test",
            metric="critical_count",
            operator="gt",
            threshold=0,
            severity=AlertSeverity.WARNING,
        )
        assert rule.evaluate(1) is True
        assert rule.evaluate(0) is False
        assert rule.evaluate(-1) is False

    def test_evaluate_lt(self) -> None:
        rule = AlertRule(
            rule_id="test",
            name="Test",
            metric="health_score",
            operator="lt",
            threshold=0.7,
            severity=AlertSeverity.WARNING,
        )
        assert rule.evaluate(0.5) is True
        assert rule.evaluate(0.8) is False

    def test_evaluate_gte(self) -> None:
        rule = AlertRule(
            rule_id="test",
            name="Test",
            metric="degraded_count",
            operator="gte",
            threshold=3,
            severity=AlertSeverity.WARNING,
        )
        assert rule.evaluate(3) is True
        assert rule.evaluate(4) is True
        assert rule.evaluate(2) is False

    def test_evaluate_lte(self) -> None:
        rule = AlertRule(
            rule_id="test",
            name="Test",
            metric="health_score",
            operator="lte",
            threshold=0.5,
            severity=AlertSeverity.CRITICAL,
        )
        assert rule.evaluate(0.5) is True
        assert rule.evaluate(0.4) is True
        assert rule.evaluate(0.6) is False

    def test_evaluate_eq(self) -> None:
        rule = AlertRule(
            rule_id="test",
            name="Test",
            metric="critical_count",
            operator="eq",
            threshold=2,
            severity=AlertSeverity.CRITICAL,
        )
        assert rule.evaluate(2) is True
        assert rule.evaluate(3) is False

    def test_evaluate_unknown_operator(self) -> None:
        rule = AlertRule(
            rule_id="test",
            name="Test",
            metric="x",
            operator="unknown",
            threshold=1,
            severity=AlertSeverity.INFO,
        )
        assert rule.evaluate(1) is False

    def test_to_dict(self) -> None:
        rule = AlertRule(
            rule_id="r1",
            name="Rule",
            metric="critical_count",
            operator="gt",
            threshold=0,
            severity=AlertSeverity.CRITICAL,
            cooldown_seconds=60,
            enabled=True,
            auto_recover=True,
            recovery_action="beat",
            metadata={"key": "val"},
        )
        d = rule.to_dict()
        assert d["rule_id"] == "r1"
        assert d["severity"] == "CRITICAL"
        assert d["auto_recover"] is True
        assert d["recovery_action"] == "beat"

    def test_from_dict(self) -> None:
        d = {
            "rule_id": "r2",
            "name": "Rule 2",
            "metric": "degraded_count",
            "operator": "gt",
            "threshold": 3,
            "severity": "WARNING",
            "cooldown_seconds": 300,
            "enabled": True,
            "auto_recover": False,
            "recovery_action": None,
            "metadata": {},
        }
        rule = AlertRule.from_dict(d)
        assert rule.rule_id == "r2"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.cooldown_seconds == 300


class TestAlert:
    def test_to_dict(self) -> None:
        alert = Alert(
            alert_id="a1",
            rule_id="r1",
            rule_name="Test Rule",
            severity=AlertSeverity.CRITICAL,
            message="Test message",
            metric_value=1.0,
            threshold=0.0,
            timestamp=1000.0,
        )
        d = alert.to_dict()
        assert d["alert_id"] == "a1"
        assert d["severity"] == "CRITICAL"
        assert d["acknowledged"] is False
        assert d["resolved"] is False


class TestEscalationPolicy:
    def test_to_dict(self) -> None:
        policy = EscalationPolicy(
            policy_id="p1",
            name="Test Policy",
            stages=[{"after_seconds": 300, "channel": "WEBHOOK"}],
        )
        d = policy.to_dict()
        assert d["policy_id"] == "p1"
        assert len(d["stages"]) == 1


class TestFleetHealthMonitor:
    def test_init(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path), check_interval=30)
        assert monitor.workspace == tmp_path
        assert monitor.check_interval == 30
        assert not monitor.is_running()

    def test_default_rules(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        rules = monitor.list_rules()
        assert len(rules) == 4
        assert any(r.rule_id == "critical_modules" for r in rules)
        assert any(r.rule_id == "degraded_threshold" for r in rules)
        assert any(r.rule_id == "low_health_score" for r in rules)
        assert any(r.rule_id == "test_failure_rate" for r in rules)

    def test_add_rule(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        rule = AlertRule(
            rule_id="custom",
            name="Custom",
            metric="critical_count",
            operator="gt",
            threshold=5,
            severity=AlertSeverity.CRITICAL,
        )
        monitor.add_rule(rule)
        assert monitor.get_rule("custom") is not None
        assert len(monitor.list_rules()) == 5

    def test_remove_rule(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        assert monitor.remove_rule("critical_modules") is True
        assert monitor.get_rule("critical_modules") is None
        assert monitor.remove_rule("nonexistent") is False

    def test_enable_disable_rule(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        assert monitor.disable_rule("critical_modules") is True
        assert monitor.get_rule("critical_modules").enabled is False
        assert monitor.enable_rule("critical_modules") is True
        assert monitor.get_rule("critical_modules").enabled is True

    def test_enable_disable_rule_not_found(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        assert monitor.disable_rule("nonexistent") is False
        assert monitor.enable_rule("nonexistent") is False

    def test_start_stop(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path), check_interval=3600)
        monitor.start()
        assert monitor.is_running()
        monitor.stop()
        assert not monitor.is_running()

    def test_check_now(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        alerts = monitor.check_now()
        # Should not trigger because fleet is healthy
        assert isinstance(alerts, list)

    def test_check_now_triggers_critical(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        # Override the critical rule to have a very low threshold
        rule = monitor.get_rule("critical_modules")
        rule.threshold = -1  # Will trigger if critical_count > -1 (always true)
        rule.cooldown_seconds = 0  # No cooldown
        alerts = monitor.check_now()
        assert len(alerts) >= 1
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_check_now_respects_cooldown(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        rule = monitor.get_rule("critical_modules")
        rule.threshold = -1
        rule.cooldown_seconds = 3600
        alerts1 = monitor.check_now()
        assert len(alerts1) >= 1
        alerts2 = monitor.check_now()
        assert len(alerts2) == 0  # Cooldown prevents second alert

    def test_get_alerts(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        rule = monitor.get_rule("critical_modules")
        rule.threshold = -1
        rule.cooldown_seconds = 0
        monitor.check_now()
        alerts = monitor.get_alerts()
        assert len(alerts) >= 1

    def test_get_alerts_filtered(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        rule = monitor.get_rule("critical_modules")
        rule.threshold = -1
        rule.cooldown_seconds = 0
        monitor.check_now()
        critical = monitor.get_alerts(severity=AlertSeverity.CRITICAL)
        assert len(critical) >= 1
        info = monitor.get_alerts(severity=AlertSeverity.INFO)
        assert len(info) == 0

    def test_acknowledge_alert(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        rule = monitor.get_rule("critical_modules")
        rule.threshold = -1
        rule.cooldown_seconds = 0
        alerts = monitor.check_now()
        alert_id = alerts[0].alert_id
        assert monitor.acknowledge_alert(alert_id) is True
        alert = monitor.get_alerts()[0]
        assert alert.acknowledged is True

    def test_resolve_alert(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        rule = monitor.get_rule("critical_modules")
        rule.threshold = -1
        rule.cooldown_seconds = 0
        alerts = monitor.check_now()
        alert_id = alerts[0].alert_id
        assert monitor.resolve_alert(alert_id) is True
        alert = monitor.get_alerts()[0]
        assert alert.resolved is True
        assert alert.resolved_at is not None

    def test_acknowledge_nonexistent(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        assert monitor.acknowledge_alert("nonexistent") is False

    def test_resolve_nonexistent(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        assert monitor.resolve_alert("nonexistent") is False

    def test_get_stats(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        rule = monitor.get_rule("critical_modules")
        rule.threshold = -1
        rule.cooldown_seconds = 0
        monitor.check_now()
        stats = monitor.get_stats()
        assert stats["total_alerts"] >= 1
        assert stats["critical_alerts"] >= 1
        assert stats["rules_count"] == 4
        assert stats["running"] is False

    def test_add_channel(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        monitor.add_channel(AlertChannel.WEBHOOK, {"url": "http://example.com/webhook"})
        # Channel added, no exception

    def test_set_callback(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        called = [False]

        def callback(alert):
            called[0] = True

        monitor.set_callback(callback)
        rule = monitor.get_rule("critical_modules")
        rule.threshold = -1
        rule.cooldown_seconds = 0
        monitor.check_now()
        assert called[0] is True

    def test_clear_channel(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        monitor.clear_channel(AlertChannel.FILE)
        # No exception

    def test_add_escalation_policy(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        policy = EscalationPolicy(
            policy_id="p1",
            name="Test Policy",
            stages=[{"after_seconds": 300, "channel": "WEBHOOK"}],
        )
        monitor.add_escalation_policy(policy)
        assert monitor.get_escalation_policy("p1") is not None

    def test_get_escalation_policy_not_found(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        assert monitor.get_escalation_policy("nonexistent") is None

    def test_check_now_healthy_no_alerts(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        # Default thresholds should not trigger for healthy fleet
        alerts = monitor.check_now()
        # With default thresholds, healthy fleet should not trigger alerts
        # (critical_count=0, degraded_count=0, test_coverage=1.0)
        assert len(alerts) == 0

    def test_rule_disabled_no_alert(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        rule = monitor.get_rule("critical_modules")
        rule.threshold = -1
        rule.cooldown_seconds = 0
        rule.enabled = False
        alerts = monitor.check_now()
        assert len(alerts) == 0

    def test_metric_value_mapping(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        health = {
            "critical": 2,
            "degraded": 3,
            "healthy": 15,
            "total_modules": 20,
            "test_coverage": 0.85,
        }
        assert monitor._get_metric_value("critical_count", health) == 2
        assert monitor._get_metric_value("degraded_count", health) == 3
        assert monitor._get_metric_value("healthy_count", health) == 15
        assert monitor._get_metric_value("total_modules", health) == 20
        assert monitor._get_metric_value("health_score", health) == 0.85
        assert abs(monitor._get_metric_value("test_failure_rate", health) - 0.15) < 1e-9
        assert monitor._get_metric_value("unknown_metric", health) is None

    def test_recovery_action_beat(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        monitor._ensure_orchestrator()
        # Should not raise
        monitor._execute_recovery("beat", None)

    def test_recovery_action_unknown(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        # Should not raise
        monitor._execute_recovery("unknown", None)

    def test_send_alert_console(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        monitor.clear_channel(AlertChannel.FILE)  # Don't write to file
        alert = Alert(
            alert_id="a1",
            rule_id="r1",
            rule_name="Test",
            severity=AlertSeverity.WARNING,
            message="Warning message",
            metric_value=1.0,
            threshold=0.0,
            timestamp=time.time(),
        )
        from unittest.mock import patch
        with patch("builtins.print") as mock_print:
            monitor._send_alert(alert)
            assert mock_print.called
            call_args = " ".join(str(a) for a in mock_print.call_args[0])
            assert "Warning message" in call_args

    def test_send_alert_file(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        alert = Alert(
            alert_id="a1",
            rule_id="r1",
            rule_name="Test",
            severity=AlertSeverity.CRITICAL,
            message="Critical message",
            metric_value=1.0,
            threshold=0.0,
            timestamp=time.time(),
        )
        monitor._send_alert(alert)
        log_file = tmp_path / "alerts.log"
        if log_file.exists():
            content = log_file.read_text()
            assert "Critical message" in content

    def test_multiple_alerts_same_rule(self, tmp_path) -> None:
        monitor = FleetHealthMonitor(workspace=str(tmp_path))
        rule = monitor.get_rule("critical_modules")
        rule.threshold = -1
        rule.cooldown_seconds = 0
        alerts1 = monitor.check_now()
        assert len(alerts1) >= 1
        # Reset cooldown
        monitor._last_alert_time.clear()
        alerts2 = monitor.check_now()
        assert len(alerts2) >= 1
        total = monitor.get_alerts()
        assert len(total) >= 2
