"""
Tests for cocapn-dashboard.

Covers: DashboardPanel, FleetDashboard, rendering, JSON export.
"""

import json

import pytest

from fleet.cocapn_dashboard import DashboardPanel, FleetDashboard


class TestDashboardPanel:
    def test_init(self):
        p = DashboardPanel(title="Test", content="Hello", width=40, height=5)
        assert p.title == "Test"
        assert p.content == "Hello"
        assert p.width == 40
        assert p.height == 5


class TestFleetDashboard:
    def test_init(self):
        dash = FleetDashboard()
        assert dash.width == 120
        assert dash.height == 40

    def test_format_bar(self):
        dash = FleetDashboard()
        bar = dash._format_bar(0.5, 1.0, width=10)
        assert len(bar) == 10
        assert "█" in bar
        assert "░" in bar

    def test_format_bar_zero(self):
        dash = FleetDashboard()
        bar = dash._format_bar(0, 1.0, width=10)
        assert bar == "░░░░░░░░░░"

    def test_format_number(self):
        dash = FleetDashboard()
        assert dash._format_number(3.14159) == "3.1"
        assert dash._format_number(100, 0) == "100"

    def test_build_breeding_panel(self):
        dash = FleetDashboard()
        campaigns = [
            {
                "name": "test-1",
                "status": "running",
                "generation": 5,
                "best_fitness": 100.0,
                "module": "pso",
            },
        ]
        panel = dash.build_breeding_panel(campaigns)
        assert "test-1" in panel.content
        assert "running" in panel.content
        assert panel.title == "🧬 Breeding Campaigns"

    def test_build_breeding_panel_empty(self):
        dash = FleetDashboard()
        panel = dash.build_breeding_panel([])
        assert "No active campaigns" in panel.content

    def test_build_spatial_panel(self):
        dash = FleetDashboard()
        spatial = {
            "rooms": {
                "ethos": {"n_agents": 5, "avg_position": (0, 0), "entropy": 0.8},
            },
            "total_agents": 5,
        }
        panel = dash.build_spatial_panel(spatial)
        assert "ethos" in panel.content
        assert "Total agents: 5" in panel.content

    def test_build_spatial_panel_empty(self):
        dash = FleetDashboard()
        panel = dash.build_spatial_panel({})
        assert "No spatial data" in panel.content

    def test_build_health_panel(self):
        dash = FleetDashboard()
        health = {
            "tests": {"passed": 100, "total": 120},
            "modules": [{"name": "test_mod", "status": "ok"}],
            "commits": 50,
        }
        panel = dash.build_health_panel(health)
        assert "100/120" in panel.content
        assert "test_mod" in panel.content
        assert "50" in panel.content

    def test_build_swarm_panel(self):
        dash = FleetDashboard()
        swarm = {
            "generation": 10,
            "n_particles": 50,
            "best_fitness": 100.0,
            "avg_fitness": 80.0,
            "diversity": 0.5,
            "pheromone_trails": 20,
            "avg_clustering": 0.3,
        }
        panel = dash.build_swarm_panel(swarm)
        assert "Generation: 10" in panel.content
        assert "Particles: 50" in panel.content

    def test_build_trinity_panel(self):
        dash = FleetDashboard()
        scores = {
            "agent-1": {"ethos": 0.8, "pathos": 0.9, "logos": 0.7},
            "agent-2": {"ethos": 0.6, "pathos": 0.6, "logos": 0.6},
        }
        panel = dash.build_trinity_panel(scores)
        assert "agent-1" in panel.content
        assert "0.8" in panel.content
        assert "0.504" in panel.content  # 0.8 * 0.9 * 0.7

    def test_build_trinity_panel_empty(self):
        dash = FleetDashboard()
        panel = dash.build_trinity_panel({})
        assert "No agents reporting" in panel.content

    def test_render(self):
        dash = FleetDashboard()
        data = {
            "campaigns": [
                {
                    "name": "c1",
                    "status": "running",
                    "generation": 5,
                    "best_fitness": 100,
                    "module": "pso",
                }
            ],
            "spatial": {"rooms": {}, "total_agents": 0},
            "health": {"tests": {"passed": 0, "total": 0}, "modules": [], "commits": 0},
            "swarm": {
                "generation": 0,
                "n_particles": 0,
                "best_fitness": 0,
                "avg_fitness": 0,
                "diversity": 0,
                "pheromone_trails": 0,
                "avg_clustering": 0,
            },
            "trinity": {},
        }
        output = dash.render(data)
        assert "COCAPN FLEET DASHBOARD" in output
        assert "🧬 Breeding Campaigns" in output
        assert "🗺️ Spatial State" in output
        assert "💓 Fleet Health" in output
        assert "🐝 Swarm State" in output
        assert "🔥 Trinity Scores" in output

    def test_render_compact(self):
        dash = FleetDashboard()
        data = {
            "health": {"tests": {"passed": 100, "total": 120}},
            "swarm": {"generation": 5, "best_fitness": 99.5},
            "spatial": {"total_agents": 10},
        }
        compact = dash.render_compact(data)
        assert "Fleet" in compact
        assert "100/120" in compact
        assert "gen=5" in compact
        assert "best=99.5" in compact
        assert "agents=10" in compact

    def test_export_json(self):
        dash = FleetDashboard()
        data = {"campaigns": [], "health": {"tests": {"passed": 50, "total": 60}}}
        json_str = dash.export_json(data)
        parsed = json.loads(json_str)
        assert parsed["health"]["tests"]["passed"] == 50

    def test_full_data_render(self):
        dash = FleetDashboard()
        data = {
            "campaigns": [
                {
                    "name": "pso-test",
                    "status": "running",
                    "generation": 12,
                    "best_fitness": 150.2,
                    "module": "swarm",
                },
                {
                    "name": "ga-test",
                    "status": "complete",
                    "generation": 50,
                    "best_fitness": 200.0,
                    "module": "standard",
                },
            ],
            "spatial": {
                "rooms": {
                    "ethos": {"n_agents": 3, "avg_position": (0, 0), "entropy": 0.5},
                    "pathos": {"n_agents": 2, "avg_position": (50, 0), "entropy": 0.3},
                },
                "total_agents": 5,
            },
            "health": {
                "tests": {"passed": 200, "total": 210},
                "modules": [
                    {"name": "swarm", "status": "ok"},
                    {"name": "spatial", "status": "ok"},
                ],
                "commits": 25,
            },
            "swarm": {
                "generation": 30,
                "n_particles": 40,
                "best_fitness": 180.0,
                "avg_fitness": 120.0,
                "diversity": 0.4,
                "pheromone_trails": 15,
                "avg_clustering": 0.2,
            },
            "trinity": {
                "alpha": {"ethos": 0.9, "pathos": 0.8, "logos": 0.7},
                "beta": {"ethos": 0.6, "pathos": 0.6, "logos": 0.6},
            },
        }
        output = dash.render(data)
        assert "pso-test" in output
        assert "complete" in output
        assert "ethos" in output
        assert "pathos" in output
        assert "swarm" in output
        assert "spatial" in output
        assert "alpha" in output
        assert "beta" in output
