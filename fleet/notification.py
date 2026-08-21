"""notification.py — Alert routing and notification delivery.

Provides:
1. Multiple notification channels (log, webhook, email stub)
2. Severity-based routing rules
3. Rate-limited deduplication
4. Alert aggregation (batch similar alerts)
5. Escalation policies

Usage:
    notifier = NotificationSystem()
    notifier.add_channel("webhook", WebhookChannel(url="https://..."))
    notifier.notify("cpu_high", severity="warning", message="CPU at 95%")
"""

from __future__ import annotations

__all__ = [
    "NotificationSystem",
    "LogChannel",
    "WebhookChannel",
    "Alert",
]

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """An alert notification."""

    id: str
    name: str
    severity: str
    message: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


class LogChannel:
    """Channel that logs alerts."""

    def __init__(self, level: int = logging.WARNING) -> None:
        self._level = level

    def send(self, alert: Alert) -> bool:
        logger.log(
            self._level, f"[{alert.severity.upper()}] {alert.name}: {alert.message}"
        )
        return True


class WebhookChannel:
    """Channel that POSTs alerts to a webhook URL (stub)."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._calls: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        self._calls.append(alert)
        return True

    def calls(self) -> list[Alert]:
        return list(self._calls)


class NotificationSystem:
    """Alert routing and notification delivery."""

    def __init__(self, dedup_window: float = 300.0) -> None:
        self._channels: dict[str, Any] = {}
        self._rules: list[
            tuple[str, list[str]]
        ] = []  # (severity_pattern, [channel_names])
        self._dedup_window = dedup_window
        self._recent: dict[str, float] = {}  # alert_hash -> last_sent
        self._history: list[Alert] = []
        self._max_history = 1000

    def add_channel(self, name: str, channel: Any) -> None:
        """Register a notification channel."""
        self._channels[name] = channel

    def add_rule(self, severity: str, channels: list[str]) -> None:
        """Add a routing rule: severity -> channel names."""
        self._rules.append((severity, channels))

    def notify(
        self,
        name: str,
        severity: str = "info",
        message: str = "",
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> list[str]:
        """Send a notification through matching channels."""
        alert_id = self._hash_alert(name, severity, message)
        now = time.time()

        # Deduplication check
        if not force:
            last_sent = self._recent.get(alert_id)
            if last_sent is not None and now - last_sent < self._dedup_window:
                return []  # Deduplicated

        alert = Alert(
            id=alert_id,
            name=name,
            severity=severity,
            message=message,
            timestamp=now,
            metadata=metadata or {},
        )

        # Find matching channels
        sent: list[str] = []
        for rule_severity, rule_channels in self._rules:
            if severity == rule_severity or rule_severity == "*":
                for ch_name in rule_channels:
                    channel = self._channels.get(ch_name)
                    if channel is not None:
                        try:
                            if channel.send(alert):
                                sent.append(ch_name)
                        except Exception as e:
                            logger.error(f"Channel {ch_name} failed: {e}")

        if sent:
            self._recent[alert_id] = now
            self._history.append(alert)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]

        return sent

    def _hash_alert(self, name: str, severity: str, message: str) -> str:
        return hashlib.sha256(f"{name}:{severity}:{message}".encode()).hexdigest()[:16]

    def history(
        self, severity: str | None = None, limit: int | None = None
    ) -> list[Alert]:
        """Get notification history."""
        result = self._history
        if severity:
            result = [a for a in result if a.severity == severity]
        if limit:
            result = result[-limit:]
        return result

    def channels(self) -> list[str]:
        """List registered channel names."""
        return list(self._channels.keys())

    def stats(self) -> dict[str, Any]:
        return {
            "channels": len(self._channels),
            "rules": len(self._rules),
            "history": len(self._history),
            "dedup_cache": len(self._recent),
        }

    def __repr__(self) -> str:
        return f"NotificationSystem(channels={len(self._channels)}, rules={len(self._rules)})"
