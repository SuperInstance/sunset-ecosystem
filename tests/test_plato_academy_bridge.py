"""Tests for Plato Academy Bridge — agent training and progression.

Covers enrollment, module completion, level progression, friction fixes,
and cohort findings.
"""

import pytest

from fleet.plato_academy_bridge import PlatoAcademyBridge, AgentProgression


class TestPlatoAcademyBridge:
    def test_init(self):
        bridge = PlatoAcademyBridge(node_id="alpha")
        assert bridge.node_id == "alpha"
        assert len(bridge._cohort_results) == 10

    def test_enroll_agent(self):
        bridge = PlatoAcademyBridge(node_id="alpha")
        agent = bridge.enroll_agent("agent_001", level="greenhorn")
        assert agent.agent_id == "agent_001"
        assert agent.level == "greenhorn"
        assert agent.score == 0.0

    def test_enroll_default_level(self):
        bridge = PlatoAcademyBridge(node_id="alpha")
        agent = bridge.enroll_agent("agent_002")
        assert agent.level == "greenhorn"

    def test_run_module(self):
        bridge = PlatoAcademyBridge(node_id="alpha")
        bridge.enroll_agent("agent_001")
        result = bridge.run_module("agent_001", module="boot_camp")
        assert result["success"] is True
        assert result["module"] == "boot_camp"
        assert result["score"] > 0

    def test_run_module_not_enrolled(self):
        bridge = PlatoAcademyBridge(node_id="alpha")
        result = bridge.run_module("missing", module="boot_camp")
        assert result["success"] is False

    def test_progression(self):
        bridge = PlatoAcademyBridge(node_id="alpha")
        bridge.enroll_agent("agent_001")

        # Complete modules to reach explorer
        bridge.run_module("agent_001", "boot_camp")  # 10
        bridge.run_module("agent_001", "room_exploration")  # 15
        bridge.run_module("agent_001", "tile_creation")  # 20

        progress = bridge.get_progression("agent_001")
        assert progress["level"] == "explorer"
        assert progress["score"] == 45.0

    def test_captain_progression(self):
        bridge = PlatoAcademyBridge(node_id="alpha")
        bridge.enroll_agent("agent_001")

        # Complete all modules to reach captain
        for module in [
            "boot_camp",
            "room_exploration",
            "tile_creation",
            "spell_casting",
            "api_integration",
            "orchestration",
            "captain_chair",
        ]:
            bridge.run_module("agent_001", module)

        progress = bridge.get_progression("agent_001")
        assert progress["level"] == "captain"
        assert progress["score"] >= 150

    def test_promote_to_fleet(self):
        bridge = PlatoAcademyBridge(node_id="alpha")
        bridge.enroll_agent("agent_001")

        for module in [
            "boot_camp",
            "room_exploration",
            "tile_creation",
            "spell_casting",
            "api_integration",
            "orchestration",
            "captain_chair",
        ]:
            bridge.run_module("agent_001", module)

        assert bridge.promote_to_fleet("agent_001") is True

    def test_promote_not_ready(self):
        bridge = PlatoAcademyBridge(node_id="alpha")
        bridge.enroll_agent("agent_001")
        bridge.run_module("agent_001", "boot_camp")
        assert bridge.promote_to_fleet("agent_001") is False

    def test_promote_not_enrolled(self):
        bridge = PlatoAcademyBridge(node_id="alpha")
        assert bridge.promote_to_fleet("missing") is False

    def test_fix_friction_point(self):
        bridge = PlatoAcademyBridge(node_id="alpha")
        fix = bridge.fix_friction_point("zero_authentication", "add_auth")
        assert fix["finding"] == "zero_authentication"
        assert fix["fix_type"] == "add_auth"
        assert fix["status"] == "applied"

    def test_get_friction_points(self):
        bridge = PlatoAcademyBridge(node_id="alpha")
        points = bridge.get_friction_points()
        assert len(points) == 10
        assert any(p["finding"] == "zero_authentication" for p in points)
        assert any(p["finding"] == "no_global_fleet_map" for p in points)

    def test_get_fixes(self):
        bridge = PlatoAcademyBridge(node_id="alpha")
        bridge.fix_friction_point("auth", "add_auth")
        bridge.fix_friction_point("ui", "add_web_ui")
        fixes = bridge.get_fixes()
        assert len(fixes) == 2

    def test_stats(self):
        bridge = PlatoAcademyBridge(node_id="alpha")
        bridge.enroll_agent("a1")
        bridge.enroll_agent("a2", level="captain")
        stats = bridge.get_stats()
        assert stats["total_agents"] == 2
        assert stats["level_distribution"]["greenhorn"] == 1
        assert stats["level_distribution"]["captain"] == 1
        assert stats["cohort_findings"] == 10

    def test_multiple_modules(self):
        bridge = PlatoAcademyBridge(node_id="alpha")
        bridge.enroll_agent("agent_001")
        for i in range(5):
            bridge.run_module("agent_001", "boot_camp")
        progress = bridge.get_progression("agent_001")
        assert len(progress["modules_completed"]) == 5
        assert progress["score"] == 50.0

    def test_cohort_severity(self):
        bridge = PlatoAcademyBridge(node_id="alpha")
        points = bridge.get_friction_points()
        critical = [p for p in points if p["severity"] == "critical"]
        assert len(critical) == 1
        assert critical[0]["finding"] == "zero_authentication"
