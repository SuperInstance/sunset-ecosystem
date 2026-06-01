"""tests/test_swarm_coordinator_bridge.py — Test suite for swarm coordinator bridge.

Covers:
- Agent registration and roles
- Trust matrix
- All 4 conflict resolution strategies
- All 5 decomposition strategies
- Task assignment
- Knowledge isolation levels
- ASCII visualization
- Serialization
"""

import pytest
from fleet.swarm_coordinator_bridge import (
    SwarmCoordinator,
    AgentRole,
    KnowledgeIsolation,
    DecompositionStrategy,
    ConflictResolutionStrategy,
    AgentProfile,
    TaskNode,
)


class TestAgentRegistration:
    def test_register_basic(self):
        c = SwarmCoordinator()
        profile = c.register_agent("scout-1", AgentRole.SCOUT)
        assert profile.id == "scout-1"
        assert profile.role == AgentRole.SCOUT
        assert "scout-1" in c.agents

    def test_register_with_capabilities(self):
        c = SwarmCoordinator()
        c.register_agent("builder-1", AgentRole.BUILDER, capabilities=["code", "test"])
        assert c.agents["builder-1"].capabilities == ["code", "test"]

    def test_max_agents_limit(self):
        c = SwarmCoordinator(max_agents=2)
        c.register_agent("a1", AgentRole.SCOUT)
        c.register_agent("a2", AgentRole.SCOUT)
        with pytest.raises(ValueError):
            c.register_agent("a3", AgentRole.SCOUT)

    def test_all_roles(self):
        c = SwarmCoordinator(max_agents=10)
        for role in AgentRole:
            c.register_agent(f"agent-{role.name}", role)
        assert len(c.agents) == len(AgentRole)


class TestTrustMatrix:
    def test_set_get_trust(self):
        c = SwarmCoordinator()
        c.register_agent("a", AgentRole.COORDINATOR)
        c.register_agent("b", AgentRole.EXECUTOR)
        c.set_trust("a", "b", 0.8)
        assert c.get_trust("a", "b") == 0.8

    def test_default_trust(self):
        c = SwarmCoordinator()
        c.register_agent("a", AgentRole.COORDINATOR)
        c.register_agent("b", AgentRole.EXECUTOR)
        assert c.get_trust("a", "b") == 0.5

    def test_trust_clamping(self):
        c = SwarmCoordinator()
        c.register_agent("a", AgentRole.COORDINATOR)
        c.register_agent("b", AgentRole.EXECUTOR)
        c.set_trust("a", "b", 2.0)
        assert c.get_trust("a", "b") == 1.0
        c.set_trust("a", "b", -1.0)
        assert c.get_trust("a", "b") == 0.0


class TestConflictResolution:
    def test_voting(self):
        c = SwarmCoordinator()
        c.register_agent("a1", AgentRole.SCOUT)
        c.register_agent("a2", AgentRole.SCOUT)
        c.register_agent("a3", AgentRole.SCOUT)
        report = c.resolve_conflict(
            ["option-a", "option-b"],
            strategy="voting",
            agent_votes={"a1": "option-a", "a2": "option-b", "a3": "option-a"},
        )
        assert report.winner == "option-a"
        assert report.strategy == "voting"
        assert report.confidence == pytest.approx(2 / 3)

    def test_weighted(self):
        c = SwarmCoordinator()
        c.register_agent("a1", AgentRole.SCOUT, trust_score=0.9, performance_score=1.0)
        c.register_agent("a2", AgentRole.SCOUT, trust_score=0.5, performance_score=0.5)
        report = c.resolve_conflict(
            ["option-a", "option-b"],
            strategy="weighted",
            agent_votes={"a1": "option-a", "a2": "option-b"},
        )
        assert report.winner == "option-a"  # 0.9 * 1.0 > 0.5 * 0.5
        assert report.strategy == "weighted"

    def test_hierarchical(self):
        c = SwarmCoordinator()
        c.register_agent("boss", AgentRole.COORDINATOR, hierarchy_level=5)
        c.register_agent("worker", AgentRole.EXECUTOR, hierarchy_level=1)
        report = c.resolve_conflict(
            ["option-a", "option-b"],
            strategy="hierarchical",
            agent_votes={"boss": "option-b", "worker": "option-a"},
        )
        assert report.winner == "option-b"  # boss has higher hierarchy
        assert report.confidence == 1.0

    def test_consensus_reached(self):
        c = SwarmCoordinator()
        c.register_agent("a1", AgentRole.SCOUT)
        c.register_agent("a2", AgentRole.SCOUT)
        c.register_agent("a3", AgentRole.SCOUT)
        report = c.resolve_conflict(
            ["option-a", "option-b"],
            strategy="consensus",
            agent_votes={"a1": "option-a", "a2": "option-a", "a3": "option-b"},
        )
        assert report.winner == "option-a"
        assert report.confidence == pytest.approx(2 / 3)
        assert "Consensus achieved" in report.reasoning

    def test_consensus_not_reached(self):
        c = SwarmCoordinator()
        c.register_agent("a1", AgentRole.SCOUT)
        c.register_agent("a2", AgentRole.SCOUT)
        c.register_agent("a3", AgentRole.SCOUT)
        report = c.resolve_conflict(
            ["option-a", "option-b", "option-c"],
            strategy="consensus",
            agent_votes={"a1": "option-a", "a2": "option-b", "a3": "option-c"},
        )
        assert report.confidence == 0.0
        assert "No consensus" in report.reasoning


class TestTaskDecomposition:
    def test_parallel(self):
        c = SwarmCoordinator()
        nodes = c.decompose_task(
            "Build a bridge",
            strategy="parallel",
            subtasks=["Design", "Code", "Test"],
        )
        assert len(nodes) == 3
        assert all(len(n.dependencies) == 0 for n in nodes)

    def test_sequential(self):
        c = SwarmCoordinator()
        nodes = c.decompose_task(
            "Build a bridge",
            strategy="sequential",
            subtasks=["Design", "Code", "Test"],
        )
        assert len(nodes) == 3
        assert nodes[0].dependencies == []
        assert nodes[1].dependencies == ["Design"]
        assert nodes[2].dependencies == ["Code"]

    def test_pipeline(self):
        c = SwarmCoordinator()
        nodes = c.decompose_task(
            "Build a bridge",
            strategy="pipeline",
            subtasks=["Stage 1", "Stage 2", "Stage 3"],
        )
        assert len(nodes) == 3
        assert nodes[0].dependencies == []
        assert nodes[1].dependencies == ["stage-0"]
        assert nodes[2].dependencies == ["stage-1"]

    def test_map_reduce(self):
        c = SwarmCoordinator()
        nodes = c.decompose_task("Process data", strategy="map_reduce")
        assert len(nodes) == 3
        assert nodes[0].id == "map-1"
        assert nodes[1].id == "map-2"
        assert nodes[2].id == "reduce"
        assert "map-1" in nodes[2].dependencies
        assert "map-2" in nodes[2].dependencies

    def test_divide_conquer(self):
        c = SwarmCoordinator()
        nodes = c.decompose_task("Sort array", strategy="divide_conquer")
        assert len(nodes) == 4
        assert nodes[0].id == "divide"
        assert nodes[1].id == "conquer-1"
        assert nodes[2].id == "conquer-2"
        assert nodes[3].id == "merge"
        assert nodes[1].dependencies == ["divide"]
        assert nodes[3].dependencies == ["conquer-1", "conquer-2"]


class TestTaskAssignment:
    def test_assign_matching(self):
        c = SwarmCoordinator()
        c.register_agent("builder", AgentRole.BUILDER, capabilities=["code", "test"])
        c.register_agent("scout", AgentRole.SCOUT, capabilities=["search"])
        task = TaskNode(id="t1", description="Write tests")
        assigned = c.assign_task(task, ["code", "test"])
        assert assigned == "builder"

    def test_assign_no_match(self):
        c = SwarmCoordinator()
        c.register_agent("scout", AgentRole.SCOUT, capabilities=["search"])
        task = TaskNode(id="t1", description="Write tests")
        assert c.assign_task(task, ["code"]) is None

    def test_assign_best_match(self):
        c = SwarmCoordinator()
        c.register_agent("a1", AgentRole.BUILDER, capabilities=["code"], trust_score=0.5)
        c.register_agent("a2", AgentRole.BUILDER, capabilities=["code"], trust_score=0.9)
        task = TaskNode(id="t1", description="Code")
        assigned = c.assign_task(task, ["code"])
        assert assigned == "a2"


class TestKnowledgeIsolation:
    def test_strict(self):
        c = SwarmCoordinator(knowledge_isolation="strict")
        c.register_agent("a", AgentRole.SCOUT, knowledge_level=2)
        assert c.can_access_knowledge("a", 2) is True
        assert c.can_access_knowledge("a", 3) is False

    def test_moderate(self):
        c = SwarmCoordinator(knowledge_isolation="moderate")
        c.register_agent("a", AgentRole.SCOUT, hierarchy_level=3)
        assert c.can_access_knowledge("a", 4) is True  # level + 1
        assert c.can_access_knowledge("a", 5) is False

    def test_relaxed(self):
        c = SwarmCoordinator(knowledge_isolation="relaxed")
        c.register_agent("a", AgentRole.SCOUT)
        assert c.can_access_knowledge("a", 100) is True

    def test_unknown_agent(self):
        c = SwarmCoordinator()
        assert c.can_access_knowledge("unknown", 0) is False


class TestSerialization:
    def test_to_dict(self):
        c = SwarmCoordinator(max_agents=5, knowledge_isolation="strict")
        c.register_agent("a1", AgentRole.COORDINATOR, capabilities=["manage"])
        c.set_trust("a1", "a1", 1.0)
        d = c.to_dict()
        assert d["max_agents"] == 5
        assert d["knowledge_isolation"] == "STRICT"
        assert "a1" in d["agents"]
        assert d["trust_matrix"]["a1→a1"] == 1.0


class TestASCIIVisualization:
    def test_render_ascii(self):
        c = SwarmCoordinator()
        c.register_agent("boss", AgentRole.COORDINATOR)
        c.register_agent("scout", AgentRole.SCOUT)
        ascii_art = c.render_ascii()
        assert "SWARM COORDINATOR" in ascii_art
        assert "boss" in ascii_art
        assert "scout" in ascii_art
        assert "COORDINATOR" in ascii_art
        assert "SCOUT" in ascii_art
