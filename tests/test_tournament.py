"""Tests for tournament and thermal modules."""

import pytest

from swarm.tournament import (
    AgentScore,
    TournamentMatch,
    TournamentRound,
    dominated_by,
    breed,
    sunset_candidates,
)
from swarm.thermal import DeviceBudget, DeviceType, ThermalBudget, DEFAULT_BUDGETS


class TestAgentScore:
    def test_create(self):
        s = AgentScore(agent_id="a", ethos=0.8, pathos=0.7, logos=0.9)
        assert s.ethos == 0.8

    def test_product(self):
        s = AgentScore(agent_id="a", ethos=0.5, pathos=0.5, logos=0.5)
        assert s.product == pytest.approx(0.125)

    def test_validation(self):
        with pytest.raises(ValueError):
            AgentScore(agent_id="a", ethos=1.5, pathos=0.5, logos=0.5)

    def test_repr(self):
        s = AgentScore(agent_id="a", ethos=0.5, pathos=0.5, logos=0.5)
        assert "E=0.50" in repr(s)


class TestTournamentMatch:
    def test_resolve(self):
        m = TournamentMatch(agent_a="a", agent_b="b", scores={"a": 0.8, "b": 0.5})
        winner = m.resolve()
        assert winner == "a"

    def test_no_scores(self):
        m = TournamentMatch(agent_a="a", agent_b="b")
        with pytest.raises(ValueError):
            m.resolve()


class TestTournamentRound:
    def test_basic_round(self):
        pop = [
            AgentScore("a", 0.9, 0.8, 0.7),
            AgentScore("b", 0.5, 0.5, 0.5),
            AgentScore("c", 0.3, 0.3, 0.3),
        ]
        tr = TournamentRound(pop)
        results = tr.run()
        assert len(results) == 3
        assert results[0].rank == 1  # "a" should win

    def test_pareto_frontier(self):
        pop = [
            AgentScore("a", 0.9, 0.3, 0.3),  # high ethos only
            AgentScore("b", 0.3, 0.9, 0.3),  # high pathos only
            AgentScore("c", 0.3, 0.3, 0.9),  # high logos only
            AgentScore("d", 0.2, 0.2, 0.2),  # dominated
        ]
        tr = TournamentRound(pop)
        tr.run()
        frontier_ids = {s.agent_id for s in tr.pareto_frontier}
        assert "a" in frontier_ids
        assert "b" in frontier_ids
        assert "c" in frontier_ids
        assert "d" not in frontier_ids  # dominated by all others

    def test_matches_run(self):
        pop = [
            AgentScore("a", 0.8, 0.7, 0.6),
            AgentScore("b", 0.5, 0.5, 0.5),
        ]
        tr = TournamentRound(pop)
        tr.run()
        assert len(tr.matches) == 1  # C(3,2) = 1 match for 2 agents


class TestDominatedBy:
    def test_dominated(self):
        weak = AgentScore("w", 0.2, 0.2, 0.2)
        strong = AgentScore("s", 0.9, 0.9, 0.9)
        assert dominated_by(weak, [weak, strong]) is True

    def test_not_dominated(self):
        a = AgentScore("a", 0.9, 0.3, 0.3)
        b = AgentScore("b", 0.3, 0.9, 0.3)
        assert dominated_by(a, [a, b]) is False

    def test_self_not_dominated(self):
        a = AgentScore("a", 0.5, 0.5, 0.5)
        assert dominated_by(a, [a]) is False


class TestBreed:
    def test_two_parents(self):
        winners = [
            AgentScore("a", 0.8, 0.7, 0.6),
            AgentScore("b", 0.6, 0.8, 0.7),
        ]
        children = breed(winners, 3)
        assert len(children) == 3
        for child in children:
            assert "id" in child
            assert "parent_a" in child
            assert 0.0 <= child["ethos"] <= 1.0

    def test_single_parent(self):
        winners = [AgentScore("a", 0.8, 0.7, 0.6)]
        children = breed(winners, 2)
        assert len(children) == 2

    def test_no_parents(self):
        children = breed([], 2)
        assert len(children) == 2


class TestSunsetCandidates:
    def test_finds_dominated(self):
        pop = [
            AgentScore("a", 0.9, 0.9, 0.9),
            AgentScore("b", 0.2, 0.2, 0.2),
        ]
        candidates = sunset_candidates(pop)
        assert len(candidates) == 1
        assert candidates[0].agent_id == "b"

    def test_no_candidates(self):
        pop = [
            AgentScore("a", 0.9, 0.3, 0.3),
            AgentScore("b", 0.3, 0.9, 0.3),
            AgentScore("c", 0.3, 0.3, 0.9),
        ]
        candidates = sunset_candidates(pop)
        assert len(candidates) == 0


# ── Thermal Tests ────────────────────────────────────────────


class TestDeviceBudget:
    def test_available(self):
        db = DeviceBudget(DeviceType.GPU, max_agents=9, current_agents=5)
        assert db.available == 4

    def test_utilization(self):
        db = DeviceBudget(DeviceType.GPU, max_agents=10, current_agents=7)
        assert db.utilization == pytest.approx(0.7)

    def test_repr(self):
        db = DeviceBudget(DeviceType.CPU, max_agents=36, current_agents=10)
        assert "cpu" in repr(db)


class TestThermalBudget:
    def test_defaults(self):
        tb = ThermalBudget()
        assert tb.total_max == 65  # 9+36+14+6

    def test_allocate_and_release(self):
        tb = ThermalBudget()
        assert tb.allocate("agent-1", DeviceType.GPU) is True
        assert tb.total_current == 1
        assert tb.release("agent-1") is True
        assert tb.total_current == 0

    def test_cannot_double_allocate(self):
        tb = ThermalBudget()
        tb.allocate("agent-1", DeviceType.GPU)
        with pytest.raises(ValueError):
            tb.allocate("agent-1", DeviceType.CPU)

    def test_device_full(self):
        tb = ThermalBudget(budgets={DeviceType.NPU: 2})
        assert tb.allocate("a", DeviceType.NPU) is True
        assert tb.allocate("b", DeviceType.NPU) is True
        assert tb.allocate("c", DeviceType.NPU) is False

    def test_can_spawn(self):
        tb = ThermalBudget(budgets={DeviceType.GPU: 1})
        assert tb.can_spawn(DeviceType.GPU) is True
        tb.allocate("a", DeviceType.GPU)
        assert tb.can_spawn(DeviceType.GPU) is False

    def test_thermal_headroom(self):
        tb = ThermalBudget(budgets={DeviceType.CPU: 10})
        tb.allocate("a", DeviceType.CPU)
        assert tb.thermal_headroom() == pytest.approx(0.1)

    def test_parent_sacrifice(self):
        tb = ThermalBudget(budgets={DeviceType.GPU: 1})
        tb.allocate("parent", DeviceType.GPU)
        # GPU is full, but sacrificing parent frees the slot
        assert tb.parent_sacrifice_before_spawn("parent", DeviceType.GPU) is True

    def test_parent_sacrifice_no_parent(self):
        tb = ThermalBudget(budgets={DeviceType.GPU: 1})
        tb.allocate("a", DeviceType.GPU)
        assert tb.parent_sacrifice_before_spawn("nonexistent", DeviceType.GPU) is False

    def test_get_device(self):
        tb = ThermalBudget()
        tb.allocate("agent-1", DeviceType.CPU)
        assert tb.get_device("agent-1") == DeviceType.CPU
        assert tb.get_device("nonexistent") is None

    def test_release_nonexistent(self):
        tb = ThermalBudget()
        assert tb.release("nonexistent") is False

    def test_reset(self):
        tb = ThermalBudget()
        tb.allocate("a", DeviceType.GPU)
        tb.allocate("b", DeviceType.CPU)
        tb.reset()
        assert tb.total_current == 0

    def test_repr(self):
        tb = ThermalBudget()
        assert "agents" in repr(tb)
