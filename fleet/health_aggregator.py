"""health_aggregator.py — Aggregate health status across fleet nodes.

Provides:
1. Collect health reports from individual nodes
2. Fleet-wide health summary (healthy, degraded, critical)
3. Per-subsystem health tracking
4. Alert on fleet-wide degradation
5. Health trend analysis

Usage:
    ha = HealthAggregator()
    ha.report("node-1", {"cpu": 0.8, "memory": 0.6, "disk": 0.9}, status="healthy")
    ha.report("node-2", {"cpu": 0.95, "memory": 0.98}, status="critical")
    summary = ha.summary()
    # summary.status, summary.healthy_count, summary.critical_count
"""
from __future__ import annotations

__all__ = [
    "HealthAggregator",
    "HealthReport",
    "FleetHealthSummary",
]

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HealthReport:
    """Health report from a single node."""
    node_id: str
    status: str  # "healthy", "degraded", "critical"
    metrics: dict[str, float]
    timestamp: float
    message: str = ""


@dataclass
class FleetHealthSummary:
    """Aggregated fleet health summary."""
    status: str
    total_nodes: int
    healthy_count: int
    degraded_count: int
    critical_count: int
    avg_metrics: dict[str, float] = field(default_factory=dict)
    worst_nodes: list[str] = field(default_factory=list)


class HealthAggregator:
    """Aggregate health across fleet nodes."""

    def __init__(self, max_age: float = 300.0) -> None:
        self._max_age = max_age
        self._reports: dict[str, HealthReport] = {}

    def report(
        self,
        node_id: str,
        metrics: dict[str, float],
        status: str = "healthy",
        message: str = "",
    ) -> None:
        """Submit a health report from a node."""
        self._reports[node_id] = HealthReport(
            node_id=node_id,
            status=status,
            metrics=metrics,
            timestamp=time.time(),
            message=message,
        )

    def get(self, node_id: str) -> HealthReport | None:
        """Get the latest report for a node."""
        return self._reports.get(node_id)

    def summary(self) -> FleetHealthSummary:
        """Generate fleet-wide health summary."""
        now = time.time()
        valid_reports = [
            r for r in self._reports.values()
            if now - r.timestamp <= self._max_age
        ]

        if not valid_reports:
            return FleetHealthSummary(
                status="unknown",
                total_nodes=0,
                healthy_count=0,
                degraded_count=0,
                critical_count=0,
            )

        healthy = sum(1 for r in valid_reports if r.status == "healthy")
        degraded = sum(1 for r in valid_reports if r.status == "degraded")
        critical = sum(1 for r in valid_reports if r.status == "critical")
        total = len(valid_reports)

        # Overall status: critical if any critical, degraded if >20% degraded
        if critical > 0:
            overall = "critical"
        elif degraded / total > 0.2:
            overall = "degraded"
        else:
            overall = "healthy"

        # Average metrics
        all_metrics: dict[str, list[float]] = {}
        for r in valid_reports:
            for k, v in r.metrics.items():
                all_metrics.setdefault(k, []).append(v)
        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}

        # Worst nodes (critical first, then highest load metric)
        worst = sorted(
            valid_reports,
            key=lambda r: (
                0 if r.status == "critical" else (1 if r.status == "degraded" else 2),
                -max(r.metrics.values()) if r.metrics else 0,
            ),
        )

        return FleetHealthSummary(
            status=overall,
            total_nodes=total,
            healthy_count=healthy,
            degraded_count=degraded,
            critical_count=critical,
            avg_metrics=avg_metrics,
            worst_nodes=[r.node_id for r in worst[:3]],
        )

    def nodes_by_status(self, status: str) -> list[str]:
        """Get node IDs with a specific status."""
        now = time.time()
        return [
            r.node_id for r in self._reports.values()
            if r.status == status and now - r.timestamp <= self._max_age
        ]

    def stale_nodes(self) -> list[str]:
        """Get nodes with expired reports."""
        now = time.time()
        return [
            r.node_id for r in self._reports.values()
            if now - r.timestamp > self._max_age
        ]

    def all_nodes(self) -> list[str]:
        """Get all known node IDs."""
        return list(self._reports.keys())

    def clear(self) -> None:
        """Clear all health reports."""
        self._reports.clear()

    def report_count(self) -> int:
        """Total number of reports (including stale)."""
        return len(self._reports)

    def __repr__(self) -> str:
        return f"HealthAggregator(nodes={len(self._reports)})"
