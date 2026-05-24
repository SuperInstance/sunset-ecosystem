"""Tests for Tide Pool Ambient Visualization.

Covers:
- Snapshot key completeness
- HTML output contains expected metrics
- ASCII output is non-empty
- Snapshot updates meaningfully after a tick
"""

from logos.tide_pool_viz import AgentSnapshot, FleetSnapshot, TidePoolVisualizer


class TestTidePoolSnapshot:
    """1. Snapshot keys and structure."""

    def test_snapshot_has_all_keys(self):
        viz = TidePoolVisualizer()
        agents = [
            AgentSnapshot(id="a-001", domain="compiler", fitness=0.82, age_ticks=10, thermal_load=0.4, status="active"),
            AgentSnapshot(id="a-002", domain="research", fitness=0.91, age_ticks=5, thermal_load=0.6, status="breeding"),
            AgentSnapshot(id="a-003", domain="compiler", fitness=0.75, age_ticks=20, thermal_load=0.3, status="idle"),
        ]
        snap = viz.generate_snapshot(
            agents=agents,
            n_rooms=12,
            recent_events=[{"type": "breed", "message": "test"}],
            thermal_state={"cuda:0": 0.5, "cpu": 0.2},
        )

        assert isinstance(snap, FleetSnapshot)
        assert snap.n_agents == 3
        assert snap.n_rooms == 12
        # (0.82 + 0.91 + 0.75) / 3 = 0.826666... → round(..., 3) = 0.827
        assert snap.mean_fitness == 0.827
        assert snap.diversity > 0.0
        assert isinstance(snap.chaos_level, float)
        assert 0.0 <= snap.chaos_level <= 1.0
        assert "cuda:0" in snap.thermal_state
        assert len(snap.recent_events) == 1
        assert len(snap.top_agents) <= 5
        assert snap.top_agents[0]["id"] == "a-002"  # highest fitness
        assert "domains" in snap.__dataclass_fields__

    def test_empty_agent_list(self):
        viz = TidePoolVisualizer()
        snap = viz.generate_snapshot(agents=[], n_rooms=8)
        assert snap.n_agents == 0
        assert snap.mean_fitness == 0.0
        assert snap.diversity == 0.0
        assert snap.top_agents == []

    def test_diversity_normalized(self):
        viz = TidePoolVisualizer()
        # All same domain → diversity = 0
        agents_same = [
            AgentSnapshot(id=f"a-{i}", domain="compiler", fitness=0.5, age_ticks=1, thermal_load=0.3, status="active")
            for i in range(10)
        ]
        snap_same = viz.generate_snapshot(agents=agents_same, n_rooms=5)
        assert snap_same.diversity == 0.0

        # Evenly split 4 domains → diversity near 1.0
        agents_mixed = [
            AgentSnapshot(id=f"a-{i}", domain=["a", "b", "c", "d"][i % 4], fitness=0.5, age_ticks=1, thermal_load=0.3, status="active")
            for i in range(40)
        ]
        snap_mixed = viz.generate_snapshot(agents=agents_mixed, n_rooms=5)
        assert snap_mixed.diversity > 0.9


class TestTidePoolHTML:
    """2. HTML render contains metrics."""

    def test_html_contains_metrics(self):
        viz = TidePoolVisualizer()
        agents = [
            AgentSnapshot(id="a-001", domain="compiler", fitness=0.82, age_ticks=10, thermal_load=0.4, status="active"),
            AgentSnapshot(id="a-002", domain="research", fitness=0.91, age_ticks=5, thermal_load=0.6, status="breeding"),
        ]
        snap = viz.generate_snapshot(agents=agents, n_rooms=10)
        html = viz.render_html(snap)

        assert "Tide Pool" in html
        assert "Agents" in html
        assert str(snap.n_agents) in html
        assert str(snap.n_rooms) in html
        assert "Mean Fitness" in html
        assert "Diversity" in html
        assert "Chaos" in html
        assert "thermal" in html.lower() or "cuda" in html.lower() or "device" in html.lower()

    def test_html_contains_hex_grid(self):
        viz = TidePoolVisualizer()
        agents = [AgentSnapshot(id=f"a-{i}", domain="x", fitness=0.5, age_ticks=1, thermal_load=0.3, status="active") for i in range(20)]
        snap = viz.generate_snapshot(agents=agents, n_rooms=16)
        html = viz.render_html(snap)
        assert "hex-grid" in html or "hex" in html

    def test_html_contains_top_agents(self):
        viz = TidePoolVisualizer()
        agents = [
            AgentSnapshot(id="a-001", domain="compiler", fitness=0.82, age_ticks=10, thermal_load=0.4, status="active"),
            AgentSnapshot(id="a-002", domain="research", fitness=0.91, age_ticks=5, thermal_load=0.6, status="breeding"),
        ]
        snap = viz.generate_snapshot(agents=agents, n_rooms=10)
        html = viz.render_html(snap)
        assert "a-002" in html
        assert "research" in html

    def test_html_uses_external_template(self):
        viz = TidePoolVisualizer()
        tpl_path = "sunset-ecosystem/logos/templates/tide_pool.html"
        import os
        if os.path.exists(tpl_path):
            html = viz.render_html(template_path=tpl_path)
            assert "Tide Pool" in html


class TestTidePoolASCII:
    """3. ASCII output is non-empty and well-formed."""

    def test_ascii_non_empty(self):
        viz = TidePoolVisualizer()
        agents = [AgentSnapshot(id=f"a-{i}", domain="x", fitness=0.5, age_ticks=1, thermal_load=0.3, status="active") for i in range(5)]
        snap = viz.generate_snapshot(agents=agents, n_rooms=4)
        ascii_art = viz.render_ascii(snap)
        assert ascii_art.strip()
        assert len(ascii_art) > 100

    def test_ascii_contains_agents_and_rooms(self):
        viz = TidePoolVisualizer()
        agents = [AgentSnapshot(id=f"a-{i}", domain="x", fitness=0.5, age_ticks=1, thermal_load=0.3, status="active") for i in range(5)]
        snap = viz.generate_snapshot(agents=agents, n_rooms=4)
        ascii_art = viz.render_ascii(snap)
        assert "Agents" in ascii_art
        assert "Rooms" in ascii_art or "ROOMS" in ascii_art

    def test_ascii_contains_thermal(self):
        viz = TidePoolVisualizer()
        agents = [AgentSnapshot(id="a-001", domain="x", fitness=0.5, age_ticks=1, thermal_load=0.3, status="active")]
        snap = viz.generate_snapshot(
            agents=agents, n_rooms=2, thermal_state={"cuda:0": 0.6}
        )
        ascii_art = viz.render_ascii(snap)
        assert "THERMAL" in ascii_art or "thermal" in ascii_art


class TestTidePoolTick:
    """4. Snapshot updates after tick / auto_refresh."""

    def test_snapshot_updates_after_tick(self):
        viz = TidePoolVisualizer()
        agents_v1 = [
            AgentSnapshot(id="a-001", domain="compiler", fitness=0.80, age_ticks=1, thermal_load=0.3, status="active"),
        ]
        snap1 = viz.generate_snapshot(agents=agents_v1, n_rooms=4)
        assert snap1.n_agents == 1
        assert snap1.mean_fitness == 0.8

        agents_v2 = [
            AgentSnapshot(id="a-001", domain="compiler", fitness=0.80, age_ticks=1, thermal_load=0.3, status="active"),
            AgentSnapshot(id="a-002", domain="research", fitness=0.95, age_ticks=1, thermal_load=0.5, status="active"),
        ]
        snap2 = viz.generate_snapshot(agents=agents_v2, n_rooms=4)
        assert snap2.n_agents == 2
        assert snap2.mean_fitness > snap1.mean_fitness
        assert len(viz._history) == 2

    def test_auto_refresh_runs(self):
        viz = TidePoolVisualizer()
        ticks = []

        def callback(snap):
            ticks.append(snap)

        def source():
            return {
                "agents": [
                    AgentSnapshot(id=f"a-{len(ticks)}", domain="x", fitness=0.5, age_ticks=1, thermal_load=0.3, status="active")
                ],
                "n_rooms": 4,
            }

        # Run exactly 3 iterations
        viz.auto_refresh(callback, source, interval_seconds=0.01, max_iterations=3)
        assert len(ticks) == 3
        assert ticks[0].n_agents == 1
        assert ticks[1].n_agents == 1
        assert ticks[2].n_agents == 1

    def test_chaos_rises_with_errors(self):
        viz = TidePoolVisualizer()
        agents = [AgentSnapshot(id="a-001", domain="x", fitness=0.5, age_ticks=1, thermal_load=0.3, status="active")]
        snap_normal = viz.generate_snapshot(agents=agents, n_rooms=2, recent_events=[{"type": "info", "message": "ok"}])
        snap_chaos = viz.generate_snapshot(
            agents=agents, n_rooms=2, recent_events=[{"type": "error", "message": "boom"}]
        )
        assert snap_chaos.chaos_level >= snap_normal.chaos_level
