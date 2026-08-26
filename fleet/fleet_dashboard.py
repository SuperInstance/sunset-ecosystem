"""FleetDashboard — generates visual dashboard reports from fleet data.

An emergent application that aggregates data from Harbor, FleetOrchestrator,
and all fleet modules to produce comprehensive dashboard reports in Markdown
and structured formats.

Usage
-----
    from fleet.fleet_dashboard import FleetDashboard

    dashboard = FleetDashboard()
    dashboard.generate_markdown_report("docs/FLEET_DASHBOARD.md")

    # Or get structured data
    data = dashboard.get_fleet_metrics()
    print(f"Fleet health: {data['health_score']:.1%}")
"""

from __future__ import annotations

__all__ = [
    "FleetDashboard",
    "FleetMetrics",
    "ModuleCard",
    "IntegrationEdge",
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
class ModuleCard:
    """Visual card for a fleet module."""

    name: str
    status: str
    health_emoji: str
    test_count: int = 0
    test_passed: int = 0
    coverage_pct: float = 0.0
    integrations: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class IntegrationEdge:
    """Visual edge for an integration path."""

    source: str
    target: str
    status: str
    status_emoji: str
    description: str = ""


@dataclass
class FleetMetrics:
    """High-level fleet metrics."""

    total_modules: int = 0
    healthy_modules: int = 0
    total_tests: int = 0
    tests_passed: int = 0
    test_coverage_pct: float = 0.0
    integration_count: int = 0
    tested_integrations: int = 0
    health_score: float = 0.0
    mean_tests_per_module: float = 0.0
    modules_with_tests: int = 0
    orphan_modules: int = 0
    hub_modules: int = 0


class FleetDashboard:
    """Fleet dashboard generator.

    Parameters
    ----------
    workspace : str
        Path to sunset-ecosystem workspace.
    """

    def __init__(self, workspace: str = ".") -> None:
        self.workspace = Path(workspace)
        self._orchestrator: FleetOrchestrator | None = None
        self._harbor: Harbor | None = None

    def _ensure_initialized(self) -> None:
        """Initialize subsystems if needed."""
        if self._orchestrator is None:
            self._orchestrator = FleetOrchestrator(workspace=str(self.workspace))
            self._orchestrator.initialize_fleet()
        if self._harbor is None:
            self._harbor = Harbor(str(self.workspace))
            self._harbor.bootstrap_fleet()

    # ── Metrics ───────────────────────────────────────────────

    def get_fleet_metrics(self) -> FleetMetrics:
        """Get high-level fleet metrics."""
        self._ensure_initialized()

        if not self._harbor:
            return FleetMetrics()

        stats = self._harbor.get_stats()
        report = self._harbor.generate_fleet_report()

        total = stats.get("modules", 0)
        tests = stats.get("tests", 0)
        mean_tests = stats.get("mean_tests_per_module", 0.0)
        tested_mods = sum(1 for m in self._harbor.modules.values() if m.test_count > 0)

        healthy = report.get("healthy", 0)
        health_score = healthy / total if total > 0 else 0.0

        integrations = len(self._harbor.integrations)
        tested = sum(1 for p in self._harbor.integrations if p.status == "tested")

        return FleetMetrics(
            total_modules=total,
            healthy_modules=healthy,
            total_tests=tests,
            tests_passed=report.get("tests_passed", 0),
            test_coverage_pct=report.get("test_coverage", 0.0),
            integration_count=integrations,
            tested_integrations=tested,
            health_score=health_score,
            mean_tests_per_module=mean_tests,
            modules_with_tests=tested_mods,
            orphan_modules=stats.get("orphan_modules", 0),
            hub_modules=stats.get("hub_modules", 0),
        )

    def get_module_cards(self) -> list[ModuleCard]:
        """Get visual cards for all modules."""
        self._ensure_initialized()

        if not self._harbor:
            return []

        cards: list[ModuleCard] = []
        for mod in self._harbor.modules.values():
            card = ModuleCard(
                name=mod.name,
                status=mod.status,
                health_emoji=TernaryValue.to_emoji(mod.health_ternary),
                test_count=mod.test_count,
                test_passed=mod.test_passed,
                coverage_pct=mod.test_coverage * 100,
                integrations=mod.integrations,
                description=mod.description,
            )
            cards.append(card)

        return sorted(cards, key=lambda c: c.name)

    def get_integration_edges(self) -> list[IntegrationEdge]:
        """Get visual edges for all integrations."""
        self._ensure_initialized()

        if not self._harbor:
            return []

        status_emojis = {
            "tested": "✅",
            "mapped": "📝",
            "unmapped": "❓",
            "broken": "❌",
        }

        edges: list[IntegrationEdge] = []
        for path in self._harbor.integrations:
            edge = IntegrationEdge(
                source=path.source,
                target=path.target,
                status=path.status,
                status_emoji=status_emojis.get(path.status, "❓"),
                description=path.description,
            )
            edges.append(edge)

        return edges

    # ── Markdown Report ───────────────────────────────────────

    def generate_markdown_report(self, output_path: str | Path) -> str:
        """Generate a comprehensive Markdown dashboard report.

        Parameters
        ----------
        output_path : str | Path
            Path to write the report.

        Returns
        -------
        str
            The Markdown content.
        """
        self._ensure_initialized()

        metrics = self.get_fleet_metrics()
        cards = self.get_module_cards()
        edges = self.get_integration_edges()

        lines: list[str] = []
        lines.append("# 🌅 Sunset Ecosystem Fleet Dashboard")
        lines.append("")
        lines.append(
            f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}*"
        )
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| **Modules** | {metrics.total_modules} |")
        lines.append(
            f"| **Healthy** | {metrics.healthy_modules} ({metrics.health_score * 100:.0f}%) |"
        )
        lines.append(f"| **Total Tests** | {metrics.total_tests} |")
        lines.append(f"| **Test Coverage** | {metrics.test_coverage_pct * 100:.0f}% |")
        lines.append(
            f"| **Integrations** | {metrics.tested_integrations}/{metrics.integration_count} tested |"
        )
        lines.append(f"| **Mean Tests/Module** | {metrics.mean_tests_per_module:.1f} |")
        lines.append(f"| **Orphan Modules** | {metrics.orphan_modules} |")
        lines.append(f"| **Hub Modules** | {metrics.hub_modules} |")
        lines.append("")

        # Health Score
        health_bar = self._render_progress_bar(metrics.health_score)
        lines.append(
            f"### Fleet Health: {health_bar} {metrics.health_score * 100:.0f}%"
        )
        lines.append("")

        # Module Cards
        lines.append("## Module Registry")
        lines.append("")
        for card in cards:
            test_bar = self._render_progress_bar(
                card.coverage_pct / 100.0 if card.test_count > 0 else 0
            )
            lines.append(f"### {card.health_emoji} {card.name}")
            lines.append(f"- **Status:** {card.status}")
            lines.append(
                f"- **Tests:** {card.test_passed}/{card.test_count} {test_bar}"
            )
            if card.integrations:
                lines.append(f"- **Integrations:** {', '.join(card.integrations)}")
            lines.append("")

        # Integration Map
        lines.append("## Integration Map")
        lines.append("")
        lines.append(f"| Source | Target | Status |")
        lines.append(f"|--------|--------|--------|")
        for edge in edges:
            lines.append(
                f"| {edge.source} | {edge.target} | {edge.status_emoji} {edge.status} |"
            )
        lines.append("")

        # Gaps
        gaps = self._harbor.find_integration_gaps() if self._harbor else []
        if gaps:
            lines.append("## Integration Gaps")
            lines.append("")
            for gap in gaps:
                lines.append(f"- **{gap['source']} → {gap['target']}**: {gap['issue']}")
            lines.append("")

        # Dependency Order
        if self._harbor:
            order = self._harbor.get_dependency_order()
            lines.append("## Dependency Order")
            lines.append("")
            for i, name in enumerate(order, 1):
                lines.append(f"{i}. {name}")
            lines.append("")

        # Recommendations
        if self._orchestrator:
            report = self._orchestrator.generate_report()
            recs = report.get("health", {}).get("recommendations", [])
            if recs:
                lines.append("## Recommendations")
                lines.append("")
                for rec in recs:
                    lines.append(f"- ⚠️ {rec}")
                lines.append("")

        content = "\n".join(lines)
        Path(output_path).write_text(content)
        return content

    def _render_progress_bar(self, fraction: float, width: int = 20) -> str:
        """Render a simple ASCII progress bar."""
        filled = int(fraction * width)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"

    # ── JSON Report ─────────────────────────────────────────

    def generate_json_report(self, output_path: str | Path) -> str:
        """Generate a structured JSON report.

        Parameters
        ----------
        output_path : str | Path
            Path to write the report.

        Returns
        -------
        str
            The JSON content.
        """
        self._ensure_initialized()

        data = {
            "timestamp": time.time(),
            "metrics": asdict(self.get_fleet_metrics()),
            "modules": [asdict(c) for c in self.get_module_cards()],
            "integrations": [asdict(e) for e in self.get_integration_edges()],
        }

        if self._orchestrator:
            data["orchestrator"] = self._orchestrator.generate_report()

        content = json.dumps(data, indent=2)
        Path(output_path).write_text(content)
        return content

    # ── Console Summary ─────────────────────────────────────

    def print_console_summary(self) -> None:
        """Print a console-friendly summary of fleet status."""
        self._ensure_initialized()

        metrics = self.get_fleet_metrics()
        print("═" * 50)
        print(" 🌅 SUNSET ECOSYSTEM FLEET DASHBOARD")
        print("═" * 50)
        print(f"  Modules:      {metrics.total_modules:3d}")
        print(
            f"  Healthy:      {metrics.healthy_modules:3d} ({metrics.health_score * 100:5.1f}%)"
        )
        print(
            f"  Tests:        {metrics.tests_passed:3d}/{metrics.total_tests:3d} ({metrics.test_coverage_pct * 100:5.1f}%)"
        )
        print(
            f"  Integrations: {metrics.tested_integrations:3d}/{metrics.integration_count:3d}"
        )
        print(f"  Mean Tests:   {metrics.mean_tests_per_module:5.1f}/module")
        print(f"  Orphans:      {metrics.orphan_modules:3d}")
        print(f"  Hubs:         {metrics.hub_modules:3d}")
        print("═" * 50)

        if self._orchestrator:
            stats = self._orchestrator.get_stats()
            print(f"  Beats:        {stats.get('beats', 0):3d}")
            print(f"  Success Rate: {stats.get('success_rate', 0.0) * 100:5.1f}%")
            print(f"  Health:       {stats.get('current_health', 'UNKNOWN')}")
            print("═" * 50)
