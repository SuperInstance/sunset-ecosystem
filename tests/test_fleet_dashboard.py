"""Tests for FleetDashboard — fleet dashboard report generator.

Reference: fleet/fleet_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fleet.fleet_dashboard import (
    FleetDashboard,
    FleetMetrics,
    IntegrationEdge,
    ModuleCard,
)


class TestFleetMetrics:
    def test_defaults(self) -> None:
        m = FleetMetrics()
        assert m.total_modules == 0
        assert m.health_score == 0.0

    def test_fields(self) -> None:
        m = FleetMetrics(
            total_modules=20,
            healthy_modules=18,
            total_tests=350,
            health_score=0.9,
        )
        assert m.total_modules == 20
        assert m.healthy_modules == 18
        assert m.health_score == 0.9


class TestModuleCard:
    def test_defaults(self) -> None:
        c = ModuleCard(name="Test", status="healthy", health_emoji="🟢")
        assert c.test_count == 0
        assert c.integrations == []

    def test_coverage(self) -> None:
        c = ModuleCard(
            name="Test",
            status="healthy",
            health_emoji="🟢",
            test_count=10,
            test_passed=8,
            coverage_pct=80.0,
        )
        assert c.coverage_pct == 80.0


class TestIntegrationEdge:
    def test_status_emoji(self) -> None:
        e = IntegrationEdge(source="A", target="B", status="tested", status_emoji="✅")
        assert e.status_emoji == "✅"


class TestFleetDashboard:
    def test_init(self) -> None:
        dash = FleetDashboard()
        assert dash._orchestrator is None
        assert dash._harbor is None

    def test_ensure_initialized(self) -> None:
        dash = FleetDashboard()
        dash._ensure_initialized()
        assert dash._orchestrator is not None
        assert dash._harbor is not None

    def test_get_fleet_metrics(self) -> None:
        dash = FleetDashboard()
        metrics = dash.get_fleet_metrics()
        assert isinstance(metrics, FleetMetrics)
        assert metrics.total_modules == 20
        assert metrics.healthy_modules == 20
        assert metrics.total_tests > 300
        assert metrics.health_score == 1.0

    def test_get_module_cards(self) -> None:
        dash = FleetDashboard()
        cards = dash.get_module_cards()
        assert len(cards) == 20
        assert all(isinstance(c, ModuleCard) for c in cards)
        assert all(c.health_emoji in ["🟢", "🟡", "🔴"] for c in cards)

    def test_get_integration_edges(self) -> None:
        dash = FleetDashboard()
        edges = dash.get_integration_edges()
        assert len(edges) > 0
        assert all(isinstance(e, IntegrationEdge) for e in edges)
        assert all(e.status_emoji in ["✅", "📝", "❓", "❌"] for e in edges)

    def test_generate_markdown_report(self, tmp_path: Path) -> None:
        dash = FleetDashboard()
        output = tmp_path / "dashboard.md"
        content = dash.generate_markdown_report(output)
        assert output.exists()
        assert "# 🌅 Sunset Ecosystem Fleet Dashboard" in content
        assert "Executive Summary" in content
        assert "Module Registry" in content
        assert "Integration Map" in content

    def test_markdown_has_module_cards(self, tmp_path: Path) -> None:
        dash = FleetDashboard()
        output = tmp_path / "dashboard.md"
        content = dash.generate_markdown_report(output)
        # Should have at least one module card
        assert "🟢" in content
        # Should have progress bars
        assert "█" in content or "░" in content

    def test_generate_json_report(self, tmp_path: Path) -> None:
        dash = FleetDashboard()
        output = tmp_path / "dashboard.json"
        content = dash.generate_json_report(output)
        assert output.exists()
        data = json.loads(content)
        assert "timestamp" in data
        assert "metrics" in data
        assert "modules" in data
        assert "integrations" in data

    def test_json_metrics(self, tmp_path: Path) -> None:
        dash = FleetDashboard()
        output = tmp_path / "dashboard.json"
        dash.generate_json_report(output)
        data = json.loads(output.read_text())
        assert data["metrics"]["total_modules"] == 20
        assert data["metrics"]["health_score"] == 1.0

    def test_json_modules(self, tmp_path: Path) -> None:
        dash = FleetDashboard()
        output = tmp_path / "dashboard.json"
        dash.generate_json_report(output)
        data = json.loads(output.read_text())
        assert len(data["modules"]) == 20

    def test_print_console_summary(self, capsys) -> None:
        dash = FleetDashboard()
        dash.print_console_summary()
        captured = capsys.readouterr()
        assert "SUNSET ECOSYSTEM FLEET DASHBOARD" in captured.out
        assert "Modules:" in captured.out
        assert "Healthy:" in captured.out

    def test_progress_bar(self) -> None:
        dash = FleetDashboard()
        bar = dash._render_progress_bar(0.5, width=10)
        assert len(bar) == 12  # [ + 10 chars + ]
        assert "█" in bar
        assert "░" in bar

    def test_progress_bar_full(self) -> None:
        dash = FleetDashboard()
        bar = dash._render_progress_bar(1.0, width=10)
        assert "█" * 10 in bar
        assert "░" not in bar

    def test_progress_bar_empty(self) -> None:
        dash = FleetDashboard()
        bar = dash._render_progress_bar(0.0, width=10)
        assert "░" * 10 in bar
        assert "█" not in bar

    def test_markdown_gaps(self, tmp_path: Path) -> None:
        dash = FleetDashboard()
        output = tmp_path / "dashboard.md"
        content = dash.generate_markdown_report(output)
        # May or may not have gaps depending on bootstrap
        assert "## Integration Gaps" in content or "## Dependency Order" in content

    def test_markdown_dependency_order(self, tmp_path: Path) -> None:
        dash = FleetDashboard()
        output = tmp_path / "dashboard.md"
        content = dash.generate_markdown_report(output)
        assert "## Dependency Order" in content

    def test_markdown_recommendations(self, tmp_path: Path) -> None:
        dash = FleetDashboard()
        output = tmp_path / "dashboard.md"
        content = dash.generate_markdown_report(output)
        # All modules are healthy, so no critical recommendations
        assert "## Recommendations" in content or "## Dependency Order" in content

    def test_orchestrator_in_json(self, tmp_path: Path) -> None:
        dash = FleetDashboard()
        output = tmp_path / "dashboard.json"
        dash.generate_json_report(output)
        data = json.loads(output.read_text())
        assert "orchestrator" in data
        assert data["orchestrator"]["initialized"] is True

