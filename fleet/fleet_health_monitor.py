"""FleetHealthMonitor — continuous health monitoring with alerting and recovery.

Monitors fleet health continuously, triggers alerts when thresholds are breached,
and executes automated recovery actions. Supports multiple alert channels
(console, webhook, file) with escalation policies.

Reference
---------
- Inspired by Datadog monitors and PagerDuty escalation policies
- Uses threading for background monitoring
- Configurable thresholds and cooldown periods
"""

from __future__ import annotations

__all__ = [
    "FleetHealthMonitor",
    "AlertRule",
    "Alert",
    "AlertChannel",
    "EscalationPolicy",
]

import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

from fleet.fleet_orchestrator import FleetOrchestrator
from fleet.harbor import Harbor
from fleet.ternary_types import TernaryValue


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


class AlertChannel(Enum):
    """Alert output channels."""

    CONSOLE = auto()
    FILE = auto()
    WEBHOOK = auto()
    CALLBACK = auto()


@dataclass
class AlertRule:
    """Rule for triggering alerts."""

    rule_id: str
    name: str
    metric: str  # "health_score", "critical_count", "degraded_count", "test_failure_rate"
    operator: str  # "gt", "lt", "gte", "lte", "eq"
    threshold: float
    severity: AlertSeverity
    cooldown_seconds: int = 300
    enabled: bool = True
    auto_recover: bool = False
    recovery_action: str | None = None  # "beat", "restart", "disable"
    metadata: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, value: float) -> bool:
        """Evaluate if the rule is triggered."""
        ops = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "gte": lambda v, t: v >= t,
            "lte": lambda v, t: v <= t,
            "eq": lambda v, t: v == t,
        }
        op = ops.get(self.operator)
        if not op:
            return False
        return op(value, self.threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "metric": self.metric,
            "operator": self.operator,
            "threshold": self.threshold,
            "severity": self.severity.name,
            "cooldown_seconds": self.cooldown_seconds,
            "enabled": self.enabled,
            "auto_recover": self.auto_recover,
            "recovery_action": self.recovery_action,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AlertRule:
        return cls(
            rule_id=d["rule_id"],
            name=d["name"],
            metric=d["metric"],
            operator=d["operator"],
            threshold=d["threshold"],
            severity=AlertSeverity[d["severity"]],
            cooldown_seconds=d.get("cooldown_seconds", 300),
            enabled=d.get("enabled", True),
            auto_recover=d.get("auto_recover", False),
            recovery_action=d.get("recovery_action"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Alert:
    """An alert instance."""

    alert_id: str
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    metric_value: float
    threshold: float
    timestamp: float
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity.name,
            "message": self.message,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at,
        }


@dataclass
class EscalationPolicy:
    """Policy for escalating unacknowledged alerts."""

    policy_id: str
    name: str
    stages: list[dict[str, Any]] = field(default_factory=list)
    # Each stage: {"after_seconds": 300, "channel": "WEBHOOK", "target": "url"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "stages": self.stages,
        }


class FleetHealthMonitor:
    """Continuous health monitor with alerting.

    Parameters
    ----------
    workspace : str
        Path to sunset-ecosystem workspace.
    check_interval : int
        Seconds between health checks.
    """

    def __init__(
        self,
        workspace: str = ".",
        check_interval: int = 60,
    ) -> None:
        self.workspace = Path(workspace)
        self.check_interval = check_interval
        self._orchestrator: FleetOrchestrator | None = None
        self._harbor: Harbor | None = None
        self._rules: dict[str, AlertRule] = {}
        self._alerts: list[Alert] = []
        self._policies: dict[str, EscalationPolicy] = {}
        self._channels: dict[AlertChannel, list[dict[str, Any]]] = {
            AlertChannel.CONSOLE: [{}],  # Always print to console
            AlertChannel.FILE: [{"path": str(self.workspace / "alerts.log")}],
            AlertChannel.WEBHOOK: [],
            AlertChannel.CALLBACK: [],
        }
        self._last_alert_time: dict[str, float] = {}  # rule_id -> timestamp
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._stop_event = threading.Event()

        # Default rules
        self._add_default_rules()

    def _ensure_orchestrator(self) -> None:
        if self._orchestrator is None:
            self._orchestrator = FleetOrchestrator(workspace=str(self.workspace))
            self._orchestrator.initialize_fleet()

    def _ensure_harbor(self) -> None:
        if self._harbor is None:
            self._harbor = Harbor(str(self.workspace))
            self._harbor.bootstrap_fleet()

    def _add_default_rules(self) -> None:
        """Add default alert rules."""
        self.add_rule(AlertRule(
            rule_id="critical_modules",
            name="Critical Modules Detected",
            metric="critical_count",
            operator="gt",
            threshold=0,
            severity=AlertSeverity.CRITICAL,
            cooldown_seconds=60,
            auto_recover=True,
            recovery_action="beat",
        ))
        self.add_rule(AlertRule(
            rule_id="degraded_threshold",
            name="Too Many Degraded Modules",
            metric="degraded_count",
            operator="gt",
            threshold=3,
            severity=AlertSeverity.WARNING,
            cooldown_seconds=300,
        ))
        self.add_rule(AlertRule(
            rule_id="low_health_score",
            name="Low Health Score",
            metric="health_score",
            operator="lt",
            threshold=0.7,
            severity=AlertSeverity.WARNING,
            cooldown_seconds=300,
        ))
        self.add_rule(AlertRule(
            rule_id="test_failure_rate",
            name="High Test Failure Rate",
            metric="test_failure_rate",
            operator="gt",
            threshold=0.1,
            severity=AlertSeverity.CRITICAL,
            cooldown_seconds=600,
        ))

    # ── Rule Management ───────────────────────────────────────

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        with self._lock:
            self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        with self._lock:
            return self._rules.pop(rule_id, None) is not None

    def get_rule(self, rule_id: str) -> AlertRule | None:
        """Get an alert rule."""
        with self._lock:
            return self._rules.get(rule_id)

    def list_rules(self) -> list[AlertRule]:
        """List all alert rules."""
        with self._lock:
            return list(self._rules.values())

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule."""
        with self._lock:
            rule = self._rules.get(rule_id)
            if rule:
                rule.enabled = True
                return True
            return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule."""
        with self._lock:
            rule = self._rules.get(rule_id)
            if rule:
                rule.enabled = False
                return True
            return False

    # ── Channel Management ────────────────────────────────────

    def add_channel(self, channel: AlertChannel, config: dict[str, Any]) -> None:
        """Add an alert channel configuration."""
        with self._lock:
            self._channels[channel].append(config)

    def clear_channel(self, channel: AlertChannel) -> None:
        """Clear all configurations for a channel."""
        with self._lock:
            self._channels[channel] = []

    def set_callback(self, callback: Callable[[Alert], None]) -> None:
        """Set a callback function for CALLBACK channel."""
        with self._lock:
            self._channels[AlertChannel.CALLBACK] = [{"fn": callback}]

    # ── Alert Lifecycle ───────────────────────────────────────

    def _check_rules(self, health: dict[str, Any]) -> list[Alert]:
        """Check all rules against current health data."""
        triggered: list[Alert] = []
        now = time.time()

        with self._lock:
            for rule in self._rules.values():
                if not rule.enabled:
                    continue

                # Check cooldown
                last_alert = self._last_alert_time.get(rule.rule_id, 0)
                if now - last_alert < rule.cooldown_seconds:
                    continue

                # Get metric value
                value = self._get_metric_value(rule.metric, health)
                if value is None:
                    continue

                # Evaluate rule
                if rule.evaluate(value):
                    alert = Alert(
                        alert_id=f"{rule.rule_id}_{int(now)}",
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"{rule.name}: {rule.metric}={value:.2f} (threshold: {rule.threshold})",
                        metric_value=value,
                        threshold=rule.threshold,
                        timestamp=now,
                    )
                    triggered.append(alert)
                    self._last_alert_time[rule.rule_id] = now
                    self._alerts.append(alert)

                    # Auto-recovery
                    if rule.auto_recover and rule.recovery_action:
                        self._execute_recovery(rule.recovery_action, alert)

        return triggered

    def _get_metric_value(self, metric: str, health: dict[str, Any]) -> float | None:
        """Extract metric value from health data."""
        metric_map = {
            "critical_count": health.get("critical", 0),
            "degraded_count": health.get("degraded", 0),
            "healthy_count": health.get("healthy", 0),
            "total_modules": health.get("total_modules", 0),
            "health_score": health.get("test_coverage", 0.0),  # Approximation
            "test_failure_rate": 1.0 - health.get("test_coverage", 1.0),
        }
        return metric_map.get(metric)

    def _execute_recovery(self, action: str, alert: Alert) -> None:
        """Execute a recovery action."""
        try:
            if action == "beat" and self._orchestrator:
                self._orchestrator.beat()
            elif action == "restart":
                # Placeholder for restart logic
                pass
            elif action == "disable":
                # Placeholder for disable logic
                pass
        except Exception:
            pass

    def _send_alert(self, alert: Alert) -> None:
        """Send alert through all configured channels."""
        with self._lock:
            channels = dict(self._channels)

        for channel, configs in channels.items():
            for config in configs:
                try:
                    if channel == AlertChannel.CONSOLE:
                        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}
                        print(
                            f"{emoji.get(alert.severity.name, '🔔')} "
                            f"[{alert.severity.name}] {alert.message}"
                        )
                    elif channel == AlertChannel.FILE:
                        path = config.get("path", "alerts.log")
                        with open(path, "a") as f:
                            f.write(json.dumps(alert.to_dict()) + "\n")
                    elif channel == AlertChannel.WEBHOOK:
                        url = config.get("url")
                        if url:
                            import urllib.request
                            data = json.dumps(alert.to_dict()).encode()
                            req = urllib.request.Request(
                                url,
                                data=data,
                                headers={"Content-Type": "application/json"},
                                method="POST",
                            )
                            urllib.request.urlopen(req, timeout=5)
                    elif channel == AlertChannel.CALLBACK:
                        fn = config.get("fn")
                        if fn and callable(fn):
                            fn(alert)
                except Exception:
                    pass

    # ── Monitoring Loop ───────────────────────────────────────

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._ensure_orchestrator()
                health = self._orchestrator.check_fleet_health()
                alerts = self._check_rules(health)
                for alert in alerts:
                    self._send_alert(alert)
            except Exception:
                pass

            # Wait for next check or stop
            self._stop_event.wait(self.check_interval)

    def start(self) -> None:
        """Start continuous monitoring."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def is_running(self) -> bool:
        """Check if monitoring is active."""
        return self._running

    # ── Query ─────────────────────────────────────────────────

    def get_alerts(
        self,
        severity: AlertSeverity | None = None,
        acknowledged: bool | None = None,
        resolved: bool | None = None,
        limit: int = 100,
    ) -> list[Alert]:
        """Get alerts with optional filtering."""
        with self._lock:
            alerts = self._alerts[:]

        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]

        return alerts[-limit:]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    return True
            return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = time.time()
                    return True
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get monitor statistics."""
        with self._lock:
            total = len(self._alerts)
            critical = sum(1 for a in self._alerts if a.severity == AlertSeverity.CRITICAL)
            warning = sum(1 for a in self._alerts if a.severity == AlertSeverity.WARNING)
            unresolved = sum(1 for a in self._alerts if not a.resolved)
            unacknowledged = sum(1 for a in self._alerts if not a.acknowledged)

        return {
            "total_alerts": total,
            "critical_alerts": critical,
            "warning_alerts": warning,
            "unresolved_alerts": unresolved,
            "unacknowledged_alerts": unacknowledged,
            "rules_count": len(self._rules),
            "running": self._running,
            "check_interval": self.check_interval,
        }

    def check_now(self) -> list[Alert]:
        """Run a health check immediately."""
        self._ensure_orchestrator()
        health = self._orchestrator.check_fleet_health()
        alerts = self._check_rules(health)
        for alert in alerts:
            self._send_alert(alert)
        return alerts

    def add_escalation_policy(self, policy: EscalationPolicy) -> None:
        """Add an escalation policy."""
        with self._lock:
            self._policies[policy.policy_id] = policy

    def get_escalation_policy(self, policy_id: str) -> EscalationPolicy | None:
        """Get an escalation policy."""
        with self._lock:
            return self._policies.get(policy_id)
