"""FleetMetricsCollector — continuous metrics collection and trend analysis.

Aggregates metrics from all fleet modules on each beat, stores them in
time-series format, and provides trend analysis (health improving/degrading).

Usage
-----
    from fleet.fleet_metrics_collector import FleetMetricsCollector

    collector = FleetMetricsCollector()
    collector.record_beat_metrics()

    trends = collector.analyze_trends(window=10)
    print(f"Health trend: {trends['health']}")  # "improving", "stable", "degrading"
"""

from __future__ import annotations

__all__ = [
    "FleetMetricsCollector",
    "MetricsSnapshot",
    "TrendAnalysis",
    "ModuleMetrics",
]

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fleet.fleet_orchestrator import FleetOrchestrator
from fleet.harbor import Harbor
from fleet.ternary_types import TernaryValue


@dataclass
class ModuleMetrics:
    """Metrics for a single module at a point in time."""

    name: str
    timestamp: float
    health_ternary: int
    health_emoji: str
    test_count: int
    test_passed: int
    test_coverage: float
    status: str


@dataclass
class MetricsSnapshot:
    """Complete snapshot of fleet metrics at a point in time."""

    timestamp: float
    cycle_number: int
    total_modules: int
    healthy_modules: int
    warning_modules: int
    critical_modules: int
    total_tests: int
    tests_passed: int
    tests_failed: int
    test_coverage_pct: float
    integration_count: int
    tested_integrations: int
    module_metrics: list[ModuleMetrics] = field(default_factory=list)

    @property
    def health_score(self) -> float:
        if self.total_modules == 0:
            return 0.0
        return self.healthy_modules / self.total_modules


@dataclass
class TrendAnalysis:
    """Trend analysis for a specific metric."""

    metric_name: str
    direction: str  # "improving", "stable", "degrading"
    slope: float
    current_value: float
    previous_value: float
    change_pct: float
    confidence: float  # 0.0-1.0


class FleetMetricsCollector:
    """Fleet metrics collector and trend analyzer.

    Parameters
    ----------
    workspace : str
        Path to sunset-ecosystem workspace.
    max_history : int
        Maximum number of snapshots to retain in memory.
    """

    def __init__(self, workspace: str = ".", max_history: int = 100) -> None:
        self.workspace = Path(workspace)
        self.max_history = max_history
        self._history: list[MetricsSnapshot] = []
        self._orchestrator: FleetOrchestrator | None = None
        self._harbor: Harbor | None = None

    def _ensure_initialized(self) -> None:
        """Initialize subsystems lazily."""
        if self._orchestrator is None:
            self._orchestrator = FleetOrchestrator(workspace=str(self.workspace))
            self._orchestrator.initialize_fleet()
        if self._harbor is None:
            self._harbor = Harbor(str(self.workspace))
            self._harbor.bootstrap_fleet()

    # ── Metrics Collection ────────────────────────────────────

    def record_beat_metrics(self) -> MetricsSnapshot:
        """Record metrics from a single fleet beat.

        Returns
        -------
        MetricsSnapshot
            The recorded snapshot.
        """
        self._ensure_initialized()

        if not self._harbor or not self._orchestrator:
            return MetricsSnapshot(
                timestamp=time.time(),
                cycle_number=0,
                total_modules=0,
                healthy_modules=0,
                warning_modules=0,
                critical_modules=0,
                total_tests=0,
                tests_passed=0,
                tests_failed=0,
                test_coverage_pct=0.0,
                integration_count=0,
                tested_integrations=0,
            )

        # Run a beat to get current state
        beat = self._orchestrator.beat()
        report = self._orchestrator.generate_report()
        health = self._orchestrator.check_fleet_health()
        harbor_stats = self._harbor.get_stats()
        harbor_report = self._harbor.generate_fleet_report()

        # Count module health states
        healthy = 0
        warning = 0
        critical = 0
        module_metrics: list[ModuleMetrics] = []

        for mod in self._harbor.modules.values():
            if mod.health_ternary == TernaryValue.POS:
                healthy += 1
            elif mod.health_ternary == TernaryValue.NEG:
                critical += 1
            else:
                warning += 1

            module_metrics.append(
                ModuleMetrics(
                    name=mod.name,
                    timestamp=time.time(),
                    health_ternary=mod.health_ternary,
                    health_emoji=TernaryValue.to_emoji(mod.health_ternary),
                    test_count=mod.test_count,
                    test_passed=mod.test_passed,
                    test_coverage=mod.test_coverage,
                    status=mod.status,
                )
            )

        snapshot = MetricsSnapshot(
            timestamp=time.time(),
            cycle_number=self._orchestrator.cycle_number,
            total_modules=harbor_stats.get("modules", 0),
            healthy_modules=healthy,
            warning_modules=warning,
            critical_modules=critical,
            total_tests=harbor_stats.get("tests", 0),
            tests_passed=harbor_report.get("tests_passed", 0),
            tests_failed=harbor_report.get("tests_failed", 0),
            test_coverage_pct=harbor_report.get("test_coverage", 0.0),
            integration_count=harbor_stats.get("integration_count", 0),
            tested_integrations=sum(
                1 for p in self._harbor.integrations if p.status == "tested"
            ),
            module_metrics=module_metrics,
        )

        self._history.append(snapshot)

        # Trim history if too large
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

        return snapshot

    def get_latest_snapshot(self) -> MetricsSnapshot | None:
        """Get the most recent metrics snapshot."""
        if not self._history:
            return None
        return self._history[-1]

    def get_history(self) -> list[MetricsSnapshot]:
        """Get all recorded snapshots."""
        return self._history.copy()

    # ── Trend Analysis ───────────────────────────────────────

    def analyze_trends(self, window: int = 10) -> dict[str, TrendAnalysis]:
        """Analyze trends over the recent history window.

        Parameters
        ----------
        window : int
            Number of recent snapshots to analyze.

        Returns
        -------
        dict[str, TrendAnalysis]
            Trend analysis for each key metric.
        """
        if len(self._history) < 2:
            return {}

        recent = self._history[-window:]
        if len(recent) < 2:
            recent = self._history

        trends: dict[str, TrendAnalysis] = {}

        # Health score trend
        health_values = [s.health_score for s in recent]
        trends["health"] = self._compute_trend("health", health_values)

        # Test coverage trend
        coverage_values = [s.test_coverage_pct for s in recent]
        trends["coverage"] = self._compute_trend("coverage", coverage_values)

        # Test count trend
        test_values = [s.total_tests for s in recent]
        trends["tests"] = self._compute_trend("tests", test_values)

        # Integration trend
        integration_values = [
            s.tested_integrations / max(s.integration_count, 1) for s in recent
        ]
        trends["integrations"] = self._compute_trend("integrations", integration_values)

        # Critical module count trend (inverse - fewer is better)
        critical_values = [s.critical_modules for s in recent]
        trends["critical"] = self._compute_trend(
            "critical", critical_values, inverse=True
        )

        return trends

    def _compute_trend(
        self,
        name: str,
        values: list[float],
        inverse: bool = False,
    ) -> TrendAnalysis:
        """Compute trend analysis for a single metric."""
        if len(values) < 2:
            return TrendAnalysis(
                metric_name=name,
                direction="stable",
                slope=0.0,
                current_value=values[-1] if values else 0.0,
                previous_value=values[-1] if values else 0.0,
                change_pct=0.0,
                confidence=0.0,
            )

        current = values[-1]
        previous = values[0]
        change = current - previous
        change_pct = (change / previous) * 100 if previous != 0 else 0.0

        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0.0

        # Determine direction
        if inverse:
            # For inverse metrics (like critical count), negative slope is improving
            if slope < -0.01:
                direction = "improving"
            elif slope > 0.01:
                direction = "degrading"
            else:
                direction = "stable"
        else:
            if slope > 0.01:
                direction = "improving"
            elif slope < -0.01:
                direction = "degrading"
            else:
                direction = "stable"

        # Confidence based on number of data points
        confidence = min(1.0, len(values) / 10.0)

        return TrendAnalysis(
            metric_name=name,
            direction=direction,
            slope=slope,
            current_value=current,
            previous_value=previous,
            change_pct=change_pct,
            confidence=confidence,
        )

    # ── Alerts ──────────────────────────────────────────────

    def check_alerts(self) -> list[dict[str, Any]]:
        """Check for alerts based on current metrics.

        Returns
        -------
        list[dict[str, Any]]
            List of alert dictionaries.
        """
        snapshot = self.get_latest_snapshot()
        if not snapshot:
            return []

        alerts: list[dict[str, Any]] = []

        if snapshot.critical_modules > 0:
            alerts.append(
                {
                    "level": "critical",
                    "metric": "critical_modules",
                    "value": snapshot.critical_modules,
                    "message": f"{snapshot.critical_modules} modules are in critical state",
                }
            )

        if snapshot.test_coverage_pct < 0.5:
            alerts.append(
                {
                    "level": "warning",
                    "metric": "test_coverage",
                    "value": snapshot.test_coverage_pct,
                    "message": f"Test coverage is {snapshot.test_coverage_pct * 100:.0f}% (below 50%)",
                }
            )

        if (
            snapshot.integration_count > 0
            and snapshot.tested_integrations / snapshot.integration_count < 0.5
        ):
            alerts.append(
                {
                    "level": "warning",
                    "metric": "integration_coverage",
                    "value": snapshot.tested_integrations / snapshot.integration_count,
                    "message": f"Only {snapshot.tested_integrations}/{snapshot.integration_count} integrations are tested",
                }
            )

        # Check trends
        if len(self._history) >= 5:
            trends = self.analyze_trends(window=5)
            health_trend = trends.get("health")
            if health_trend and health_trend.direction == "degrading":
                alerts.append(
                    {
                        "level": "warning",
                        "metric": "health_trend",
                        "value": health_trend.slope,
                        "message": "Fleet health is degrading over recent beats",
                    }
                )

        return alerts

    # ── Persistence ───────────────────────────────────────────

    def save_history(self, path: str | Path) -> None:
        """Save metrics history to JSON file.

        Parameters
        ----------
        path : str | Path
            Path to write the history.
        """
        data = [asdict(s) for s in self._history]
        Path(path).write_text(json.dumps(data, indent=2))

    def load_history(self, path: str | Path) -> None:
        """Load metrics history from JSON file.

        Parameters
        ----------
        path : str | Path
            Path to read the history from.
        """
        data = json.loads(Path(path).read_text())
        self._history = []
        for item in data:
            module_metrics = [
                ModuleMetrics(**m) for m in item.pop("module_metrics", [])
            ]
            snapshot = MetricsSnapshot(**item, module_metrics=module_metrics)
            self._history.append(snapshot)

    # ── Report Generation ─────────────────────────────────────

    def generate_trend_report(self, output_path: str | Path) -> str:
        """Generate a Markdown trend report.

        Parameters
        ----------
        output_path : str | Path
            Path to write the report.

        Returns
        -------
        str
            The Markdown content.
        """
        snapshot = self.get_latest_snapshot()
        trends = self.analyze_trends()
        alerts = self.check_alerts()

        lines: list[str] = []
        lines.append("# 📈 Fleet Metrics Trend Report")
        lines.append("")
        if snapshot:
            lines.append(
                f"*Cycle {snapshot.cycle_number} | {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(snapshot.timestamp))}*"
            )
        lines.append("")

        # Current snapshot
        if snapshot:
            lines.append("## Current Snapshot")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Health Score | {snapshot.health_score * 100:.1f}% |")
            lines.append(
                f"| Healthy Modules | {snapshot.healthy_modules}/{snapshot.total_modules} |"
            )
            lines.append(f"| Critical Modules | {snapshot.critical_modules} |")
            lines.append(
                f"| Total Tests | {snapshot.tests_passed}/{snapshot.total_tests} |"
            )
            lines.append(f"| Test Coverage | {snapshot.test_coverage_pct * 100:.1f}% |")
            lines.append(
                f"| Integrations Tested | {snapshot.tested_integrations}/{snapshot.integration_count} |"
            )
            lines.append("")

        # Trends
        if trends:
            lines.append("## Trends")
            lines.append("")
            lines.append(f"| Metric | Direction | Change | Confidence |")
            lines.append(f"|--------|-----------|--------|------------|")
            for name, trend in trends.items():
                emoji = (
                    "📈"
                    if trend.direction == "improving"
                    else "📉"
                    if trend.direction == "degrading"
                    else "➡️"
                )
                lines.append(
                    f"| {name} | {emoji} {trend.direction} | {trend.change_pct:+.1f}% | {trend.confidence * 100:.0f}% |"
                )
            lines.append("")

        # Alerts
        if alerts:
            lines.append("## Alerts")
            lines.append("")
            for alert in alerts:
                emoji = "🔴" if alert["level"] == "critical" else "🟡"
                lines.append(
                    f"- {emoji} **{alert['level'].upper()}**: {alert['message']}"
                )
            lines.append("")
        else:
            lines.append("## Alerts")
            lines.append("")
            lines.append("✅ No alerts at this time.")
            lines.append("")

        # History size
        lines.append(f"## History")
        lines.append("")
        lines.append(f"- **Snapshots recorded**: {len(self._history)}")
        lines.append(f"- **Max history**: {self.max_history}")
        lines.append("")

        content = "\n".join(lines)
        Path(output_path).write_text(content)
        return content

    # ── Console Output ───────────────────────────────────────

    def print_summary(self) -> None:
        """Print a console summary of current metrics."""
        snapshot = self.get_latest_snapshot()
        if not snapshot:
            print("No metrics recorded yet.")
            return

        print("═" * 50)
        print(" 📈 FLEET METRICS COLLECTOR")
        print("═" * 50)
        print(f"  Cycle:          {snapshot.cycle_number}")
        print(f"  Health:         {snapshot.health_score * 100:5.1f}%")
        print(f"  Modules:        {snapshot.healthy_modules}/{snapshot.total_modules}")
        print(f"  Critical:       {snapshot.critical_modules}")
        print(f"  Tests:          {snapshot.tests_passed}/{snapshot.total_tests}")
        print(f"  Coverage:       {snapshot.test_coverage_pct * 100:5.1f}%")
        print(
            f"  Integrations:   {snapshot.tested_integrations}/{snapshot.integration_count}"
        )
        print("═" * 50)

        trends = self.analyze_trends()
        if trends:
            print("  Trends:")
            for name, trend in trends.items():
                arrow = (
                    "↑"
                    if trend.direction == "improving"
                    else "↓"
                    if trend.direction == "degrading"
                    else "→"
                )
                print(
                    f"    {name:12} {arrow} {trend.direction} ({trend.change_pct:+.1f}%)"
                )
            print("═" * 50)

        alerts = self.check_alerts()
        if alerts:
            print("  Alerts:")
            for alert in alerts:
                level = "CRIT" if alert["level"] == "critical" else "WARN"
                print(f"    [{level}] {alert['message']}")
            print("═" * 50)
