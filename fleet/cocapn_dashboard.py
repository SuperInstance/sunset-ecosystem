"""
cocapn-dashboard — Terminal UI for Fleet Status

A rich terminal dashboard for monitoring the Cocapn Fleet in real-time.
Shows breeding campaigns, spatial state, swarm health, and agent status.

Usage:
    from fleet.cocapn_dashboard import FleetDashboard
    dash = FleetDashboard()
    dash.render()  # Single render
    # Or for live monitoring:
    dash.run_live(update_interval=1.0)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DashboardPanel:
    """A panel in the dashboard."""
    title: str
    content: str
    width: int = 40
    height: int = 10
    color: str = "white"


class FleetDashboard:
    """
    Terminal dashboard for the Cocapn Fleet.

    Displays:
    - Breeding campaigns (active, completed, best fitness)
    - Spatial state (agent positions, room distribution)
    - Fleet health (tests, commits, module status)
    - Agent activity (swarm particles, breeding generations)
    """

    def __init__(self):
        self.panels: List[DashboardPanel] = []
        self.width = 120
        self.height = 40

    def _format_bar(self, value: float, max_val: float, width: int = 20) -> str:
        """Create a simple ASCII bar."""
        if max_val <= 0:
            return " " * width
        filled = int((value / max_val) * width)
        return "█" * filled + "░" * (width - filled)

    def _format_number(self, n: float, decimals: int = 1) -> str:
        return f"{n:.{decimals}f}"

    def build_breeding_panel(self, campaigns: List[Dict]) -> DashboardPanel:
        """Build panel showing active breeding campaigns."""
        lines = ["Campaign                Status    Gen    Best Fit    Module"]
        lines.append("-" * 56)

        for c in campaigns:
            name = c.get("name", "unknown")[:18]
            status = c.get("status", "?")[:8]
            gen = str(c.get("generation", 0))[:4]
            best = self._format_number(c.get("best_fitness", 0))[:8]
            module = c.get("module", "")[:12]
            lines.append(f"{name:20} {status:8} {gen:4} {best:10} {module}")

        if not campaigns:
            lines.append("No active campaigns")

        return DashboardPanel(
            title="🧬 Breeding Campaigns",
            content="\n".join(lines),
            width=60,
            height=max(5, len(lines) + 2)
        )

    def build_spatial_panel(self, spatial_state: Dict) -> DashboardPanel:
        """Build panel showing spatial state."""
        lines = ["Room           Agents    Avg Position    Entropy"]
        lines.append("-" * 50)

        rooms = spatial_state.get("rooms", {})
        for room_id, data in rooms.items():
            n_agents = data.get("n_agents", 0)
            avg_pos = data.get("avg_position", (0, 0))
            entropy = data.get("entropy", 0)
            pos_str = f"({avg_pos[0]:.1f}, {avg_pos[1]:.1f})"
            lines.append(f"{room_id:15} {n_agents:5} {pos_str:15} {entropy:.2f}")

        if not rooms:
            lines.append("No spatial data")

        total_agents = spatial_state.get("total_agents", 0)
        lines.append(f"\nTotal agents: {total_agents}")

        return DashboardPanel(
            title="🗺️ Spatial State",
            content="\n".join(lines),
            width=60,
            height=max(5, len(lines) + 2)
        )

    def build_health_panel(self, health: Dict) -> DashboardPanel:
        """Build panel showing fleet health."""
        lines = []

        tests = health.get("tests", {})
        lines.append(f"Tests: {tests.get('passed', 0)}/{tests.get('total', 0)} passed")
        if tests.get("total", 0) > 0:
            ratio = tests["passed"] / tests["total"]
            lines.append(f"  [{self._format_bar(ratio, 1.0)}] {ratio*100:.0f}%")

        lines.append("")
        modules = health.get("modules", [])
        lines.append(f"Modules: {len(modules)} loaded")
        for mod in modules[:5]:
            status = "✅" if mod.get("status") == "ok" else "❌"
            lines.append(f"  {status} {mod.get('name', '?')}")

        lines.append("")
        commits = health.get("commits", 0)
        lines.append(f"Commits: {commits}")

        return DashboardPanel(
            title="💓 Fleet Health",
            content="\n".join(lines),
            width=40,
            height=max(5, len(lines) + 2)
        )

    def build_swarm_panel(self, swarm: Dict) -> DashboardPanel:
        """Build panel showing swarm intelligence state."""
        lines = []

        lines.append(f"Generation: {swarm.get('generation', 0)}")
        lines.append(f"Particles: {swarm.get('n_particles', 0)}")
        lines.append(f"Best Fitness: {self._format_number(swarm.get('best_fitness', 0))}")
        lines.append(f"Avg Fitness: {self._format_number(swarm.get('avg_fitness', 0))}")
        lines.append(f"Diversity: {self._format_number(swarm.get('diversity', 0))}")
        lines.append(f"Pheromones: {swarm.get('pheromone_trails', 0)}")
        lines.append("")
        lines.append(f"Clustering: {self._format_number(swarm.get('avg_clustering', 0))}")

        return DashboardPanel(
            title="🐝 Swarm State",
            content="\n".join(lines),
            width=35,
            height=10
        )

    def build_trinity_panel(self, scores: Dict) -> DashboardPanel:
        """Build panel showing trinity scores."""
        lines = ["Agent              Ethos  Pathos  Logos  Trinity"]
        lines.append("-" * 50)

        for agent_id, scores in scores.items():
            e = scores.get("ethos", 0)
            p = scores.get("pathos", 0)
            l = scores.get("logos", 0)
            t = e * p * l
            lines.append(f"{agent_id:18} {e:.2f}  {p:.2f}  {l:.2f}  {t:.3f}")

        if not scores:
            lines.append("No agents reporting")

        return DashboardPanel(
            title="🔥 Trinity Scores",
            content="\n".join(lines),
            width=55,
            height=max(5, len(lines) + 2)
        )

    def render(self, data: Optional[Dict] = None) -> str:
        """
        Render the dashboard as a string.

        Args:
            data: Optional pre-built data dictionary with keys:
                - campaigns, spatial, health, swarm, trinity
        """
        data = data or {}

        panels = [
            self.build_breeding_panel(data.get("campaigns", [])),
            self.build_spatial_panel(data.get("spatial", {})),
            self.build_health_panel(data.get("health", {})),
            self.build_swarm_panel(data.get("swarm", {})),
            self.build_trinity_panel(data.get("trinity", {})),
        ]

        # Simple layout: stack panels vertically
        lines = []
        lines.append("╔" + "═" * 118 + "╗")
        lines.append("║" + " " * 40 + "COCAPN FLEET DASHBOARD" + " " * 54 + "║")
        lines.append("╠" + "═" * 118 + "╣")

        for panel in panels:
            lines.append(f"║ {panel.title:116} ║")
            lines.append("╟" + "─" * 118 + "╢")
            for line in panel.content.split("\n"):
                truncated = line[:118]
                lines.append(f"║ {truncated:118} ║")
            lines.append("╠" + "═" * 118 + "╣")

        lines.append("╚" + "═" * 118 + "╝")

        return "\n".join(lines)

    def render_compact(self, data: Optional[Dict] = None) -> str:
        """Render a compact one-line status."""
        data = data or {}

        health = data.get("health", {})
        tests = health.get("tests", {})
        passed = tests.get("passed", 0)
        total = tests.get("total", 0)

        swarm = data.get("swarm", {})
        gen = swarm.get("generation", 0)
        best = swarm.get("best_fitness", 0)

        spatial = data.get("spatial", {})
        agents = spatial.get("total_agents", 0)

        return (
            f"[Fleet] tests={passed}/{total} "
            f"gen={gen} best={best:.2f} "
            f"agents={agents}"
        )

    def run_live(self, data_source: Optional[Callable] = None,
                 update_interval: float = 1.0, max_iterations: int = 100):
        """
        Run live dashboard (would use curses in real implementation).
        For now, just prints periodically.
        """
        for i in range(max_iterations):
            data = data_source() if data_source else {}
            compact = self.render_compact(data)
            print(f"\r{compact}", end="", flush=True)
            time.sleep(update_interval)
        print()

    def export_json(self, data: Optional[Dict] = None) -> str:
        """Export dashboard data as JSON."""
        data = data or {}
        return json.dumps(data, indent=2)
