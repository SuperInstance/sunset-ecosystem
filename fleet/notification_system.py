from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A fleet alert."""

    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    source: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "acknowledged": self.acknowledged,
        }


class NotificationChannel:
    """Base class for notification channels."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    def send(self, alert: Alert) -> bool:
        """Send an alert. Return True if sent."""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "type": type(self).__name__}


class WebhookChannel(NotificationChannel):
    """Webhook notification channel."""

    def send(self, alert: Alert) -> bool:
        # Mock: log instead of actual HTTP
        return True


class LogChannel(NotificationChannel):
    """Log notification channel."""

    def __init__(self):
        super().__init__("log", {})
        self._entries: List[str] = []

    def send(self, alert: Alert) -> bool:
        entry = f"[{alert.severity.value.upper()}] {alert.title}: {alert.message}"
        self._entries.append(entry)
        return True

    def get_entries(self) -> List[str]:
        return self._entries


class NotificationSystem:
    """
    Fleet notification system.

    Routes alerts to channels based on severity and source rules.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self.channels: Dict[str, NotificationChannel] = {}
        self._alerts: List[Alert] = []
        self._rules: List[Dict[str, Any]] = []
        self._silenced: set = set()

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self.channels[channel.name] = channel

    def add_rule(
        self,
        severity: Optional[str] = None,
        source: Optional[str] = None,
        channel_name: Optional[str] = None,
    ) -> None:
        """Add a routing rule."""
        self._rules.append(
            {
                "severity": severity,
                "source": source,
                "channel": channel_name,
            }
        )

    def send(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        source: str = "fleet",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Send an alert."""
        alert = Alert(
            alert_id=f"alert_{int(time.time() * 1000000)}",
            title=title,
            message=message,
            severity=severity,
            source=source,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self._alerts.append(alert)

        # Route to channels
        for rule in self._rules:
            if rule.get("severity") and rule["severity"] != severity.value:
                continue
            if rule.get("source") and rule["source"] != source:
                continue
            channel_name = rule.get("channel")
            if channel_name and channel_name in self.channels:
                self.channels[channel_name].send(alert)

        # Default: send to all channels if no rules match
        if not self._rules:
            for channel in self.channels.values():
                channel.send(alert)

        return alert

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_alerts(
        self, severity: Optional[str] = None, acknowledged: Optional[bool] = None
    ) -> List[Alert]:
        """Get alerts with optional filters."""
        alerts = self._alerts
        if severity:
            alerts = [a for a in alerts if a.severity.value == severity]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        return alerts

    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        by_severity = {}
        for alert in self._alerts:
            s = alert.severity.value
            by_severity[s] = by_severity.get(s, 0) + 1
        return {
            "total_alerts": len(self._alerts),
            "channels": len(self.channels),
            "rules": len(self._rules),
            "by_severity": by_severity,
        }

    def export_json(self) -> str:
        """Export alerts as JSON."""
        return json.dumps(
            {
                "node": self.fleet_node_id,
                "alerts": [a.to_dict() for a in self._alerts[-100:]],
                "stats": self.get_stats(),
            },
            indent=2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
        }
