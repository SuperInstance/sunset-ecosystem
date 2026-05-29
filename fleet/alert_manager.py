"""Threshold-based alert system with deduplication.

Monitors values against thresholds and emits alerts with deduplication
to prevent alert spam. Used for fleet health monitoring and anomaly
detection.

Usage:
    alerts = AlertManager(cooldown_sec=60)
    alerts.add_rule("cpu_high", threshold=90, op="gt")
    alerts.check("cpu_high", 95)  # triggers alert
    alerts.check("cpu_high", 95)  # suppressed (cooldown)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AlertRule:
    """An alert rule definition."""

    name: str
    threshold: float
    op: str  # gt, lt, gte, lte, eq
    cooldown_sec: float
    last_alert: float = 0.0
    alert_count: int = 0


class AlertManager:
    """
    Alert manager with threshold rules and cooldown deduplication.

    :param default_cooldown: Default cooldown between repeated alerts.
    """

    OPS: Dict[str, Callable[[float, float], bool]] = {
        "gt": lambda v, t: v > t,
        "lt": lambda v, t: v < t,
        "gte": lambda v, t: v >= t,
        "lte": lambda v, t: v <= t,
        "eq": lambda v, t: v == t,
    }

    def __init__(self, default_cooldown: float = 300.0):
        self._default_cooldown = default_cooldown
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(
        self,
        name: str,
        threshold: float,
        op: str = "gt",
        cooldown_sec: Optional[float] = None,
    ) -> None:
        """Add an alert rule."""
        if op not in self.OPS:
            raise ValueError(f"Unknown op: {op}. Use: {list(self.OPS.keys())}")
        self._rules[name] = AlertRule(
            name=name,
            threshold=threshold,
            op=op,
            cooldown_sec=cooldown_sec or self._default_cooldown,
        )

    def remove_rule(self, name: str) -> bool:
        """Remove a rule."""
        if name in self._rules:
            del self._rules[name]
            return True
        return False

    # ------------------------------------------------------------------
    # Checking
    # ------------------------------------------------------------------

    def check(self, name: str, value: float, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check a value against a rule.

        :returns: True if alert was triggered.
        """
        rule = self._rules.get(name)
        if not rule:
            return False
        op_fn = self.OPS[rule.op]
        if not op_fn(value, rule.threshold):
            return False
        now = time.time()
        if now - rule.last_alert < rule.cooldown_sec:
            return False
        rule.last_alert = now
        rule.alert_count += 1
        alert = {
            "rule": name,
            "value": value,
            "threshold": rule.threshold,
            "timestamp": now,
            "context": context or {},
        }
        self._alerts.append(alert)
        return True

    def check_all(self, values: Dict[str, float]) -> List[str]:
        """
        Check multiple values against their rules.

        :returns: List of triggered rule names.
        """
        triggered: List[str] = []
        for name, value in values.items():
            if self.check(name, value):
                triggered.append(name)
        return triggered

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_rules(self) -> List[str]:
        return list(self._rules.keys())

    def get_rule(self, name: str) -> Optional[AlertRule]:
        return self._rules.get(name)

    def alert_history(self) -> List[Dict[str, Any]]:
        return list(self._alerts)

    def clear_history(self) -> None:
        self._alerts.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "rules": len(self._rules),
            "total_alerts": len(self._alerts),
            "alerts_by_rule": {
                name: rule.alert_count for name, rule in self._rules.items()
            },
        }

    def __repr__(self) -> str:
        return f"<AlertManager rules={len(self._rules)} alerts={len(self._alerts)}>"
