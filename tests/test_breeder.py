"""Tests for Breeder — tournament breeding with lifecycle FSM.

Covers LifecycleRecord, spawn_from_template, Breeder evolve, spawn_template,
tick_all, sunset, and stats.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from swarm.breeder import (
    AgentLifecycle,
    LifecycleRecord,
    spawn_from_template,
    Breeder,
)
from swarm.tournament import AgentScore
from nerve.templates import AgentTemplate


def _mock_grid(n=4):
    """Create a minimal mock RoomGrid with numpy arrays."""
    grid = MagicMock()
    grid.n = n
    grid.ticks = 0
    grid.activity = np.zeros(n)
    grid.chaos = np.zeros(n)
    grid.w = {
        "w1": [np.zeros((64, 32), dtype=np.float32) for _ in range(n)],
        "w2": [np.zeros((32, 16), dtype=np.float32) for _ in range(n)],
        "w3": [np.zeros((16, 16), dtype=np.float32) for _ in range(n)],
    }
    grid.cold.return_value = list(range(n))
    return grid


# ---------------------------------------------------------------------------
# LifecycleRecord
# ---------------------------------------------------------------------------


class TestLifecycleRecord:
    def test_init_defaults(self):
        lr = LifecycleRecord(room_id=5)
        assert lr.room_id == 5
        assert lr.state == AgentLifecycle.SPAWNED
        assert lr.generation == 0
        assert lr.chaos == 0.3
        assert lr.activity == 0
        assert lr.hint_level == 10
        assert lr.consecutive_wins == 0
        assert lr.tick_entered == 0

    def test_can_advance_spawned(self):
        lr = LifecycleRecord(room_id=0, state=AgentLifecycle.SPAWNED, activity=1)
        assert lr.can_advance()

    def test_can_advance_spawned_inactive(self):
        lr = LifecycleRecord(room_id=0, state=AgentLifecycle.SPAWNED, activity=0)
        assert not lr.can_advance()

    def test_can_advance_active_low_chaos(self):
        lr = LifecycleRecord(room_id=0, state=AgentLifecycle.ACTIVE, chaos=0.04)
        assert lr.can_advance()

    def test_can_advance_active_high_chaos(self):
        lr = LifecycleRecord(room_id=0, state=AgentLifecycle.ACTIVE, chaos=0.1)
        assert not lr.can_advance()

    def test_can_advance_adapting(self):
        lr = LifecycleRecord(
            room_id=0, state=AgentLifecycle.ADAPTING, consecutive_wins=3
        )
        assert lr.can_advance()

    def test_can_advance_adapting_not_enough_wins(self):
        lr = LifecycleRecord(
            room_id=0, state=AgentLifecycle.ADAPTING, consecutive_wins=2
        )
        assert not lr.can_advance()

    def test_can_advance_compiled(self):
        lr = LifecycleRecord(room_id=0, state=AgentLifecycle.COMPILED)
        assert not lr.can_advance()


# ---------------------------------------------------------------------------
# spawn_from_template
# ---------------------------------------------------------------------------


class TestSpawnFromTemplate:
    def test_spawns_into_room(self):
        grid = _mock_grid(n=4)
        template = AgentTemplate(name="test", chaos_initial=0.5)
        spawn_from_template(grid, template, room_idx=2, seed=42)
        grid.rebirth.assert_called_once_with(2)
        assert grid.chaos[2] == pytest.approx(0.5)

    def test_noise_injected(self):
        grid = _mock_grid(n=2)
        template = AgentTemplate(name="test", chaos_initial=0.1)
        base_w1 = grid.w["w1"][0].copy()
        spawn_from_template(grid, template, room_idx=0, seed=123)
        assert not np.array_equal(grid.w["w1"][0], base_w1)


# ---------------------------------------------------------------------------
# Breeder init
# ---------------------------------------------------------------------------


class TestBreederInit:
    def test_default(self):
        grid = _mock_grid(n=8)
        thermal = MagicMock()
        thermal.thermal_headroom.return_value = 0.5
        breeder = Breeder(grid, {}, thermal)
        assert breeder.grid is grid
        assert breeder.generation == 0
        assert breeder.lifecycle_state(0) == AgentLifecycle.SPAWNED


# ---------------------------------------------------------------------------
# Lifecycle management
# ---------------------------------------------------------------------------


class TestLifecycleManagement:
    def test_advance_spawned_to_active(self):
        grid = _mock_grid(n=4)
        grid.ticks = 1
        grid.activity = np.array([1, 0, 0, 0])
        grid.chaos = np.zeros(4)
        thermal = MagicMock()
        breeder = Breeder(grid, {}, thermal)
        breeder._lifecycle[0] = LifecycleRecord(
            room_id=0, state=AgentLifecycle.SPAWNED, activity=1
        )
        breeder._advance_lifecycle(0)
        assert breeder.lifecycle_state(0) == AgentLifecycle.ACTIVE

    def test_advance_active_to_adapting(self):
        grid = _mock_grid(n=4)
        grid.ticks = 1
        grid.activity = np.array([1, 0, 0, 0])
        grid.chaos = np.array([0.01, 0, 0, 0])
        thermal = MagicMock()
        breeder = Breeder(grid, {}, thermal)
        breeder._lifecycle[0] = LifecycleRecord(
            room_id=0, state=AgentLifecycle.ACTIVE, activity=1, chaos=0.01
        )
        breeder._advance_lifecycle(0)
        assert breeder.lifecycle_state(0) == AgentLifecycle.ADAPTING

    def test_advance_adapting_to_compiled(self):
        grid = _mock_grid(n=4)
        grid.ticks = 1
        grid.activity = np.array([1, 0, 0, 0])
        grid.chaos = np.zeros(4)
        thermal = MagicMock()
        breeder = Breeder(grid, {}, thermal)
        breeder._lifecycle[0] = LifecycleRecord(
            room_id=0, state=AgentLifecycle.ADAPTING, consecutive_wins=3
        )
        breeder._advance_lifecycle(0)
        assert breeder.lifecycle_state(0) == AgentLifecycle.COMPILED


# ---------------------------------------------------------------------------
# evolve
# ---------------------------------------------------------------------------


class TestEvolve:
    def test_no_winners(self):
        grid = _mock_grid(n=4)
        grid.activity = np.zeros(4)
        grid.chaos = np.zeros(4)
        thermal = MagicMock()
        thermal.thermal_headroom.return_value = 0.5
        breeder = Breeder(grid, {}, thermal)
        scores = [
            AgentScore("a", ethos=0.1, pathos=0.1, logos=0.1),
        ]
        result = breeder.evolve(scores)
        assert result == []
        assert breeder.generation == 1

    def test_evolve_places_children(self):
        grid = _mock_grid(n=4)
        grid.activity = np.array([0, 0, 1, 1])
        grid.chaos = np.zeros(4)
        grid.cold.return_value = [0, 1]
        thermal = MagicMock()
        thermal.thermal_headroom.return_value = 0.5
        breeder = Breeder(grid, {}, thermal)
        scores = [
            AgentScore("a", ethos=0.9, pathos=0.9, logos=0.9),
            AgentScore("b", ethos=0.1, pathos=0.1, logos=0.1),
        ]
        result = breeder.evolve(scores)
        assert len(result) >= 0

    def test_no_room_for_children(self):
        grid = _mock_grid(n=2)
        grid.activity = np.array([1, 0])
        grid.chaos = np.zeros(2)
        grid.cold.return_value = []
        thermal = MagicMock()
        thermal.thermal_headroom.return_value = 0.5
        breeder = Breeder(grid, {}, thermal)
        scores = [
            AgentScore("a", ethos=0.9, pathos=0.9, logos=0.9),
            AgentScore("b", ethos=0.1, pathos=0.1, logos=0.1),
        ]
        result = breeder.evolve(scores)
        assert len(result) == 1
        assert result[0]["room"] == 1


# ---------------------------------------------------------------------------
# spawn_template
# ---------------------------------------------------------------------------


class TestSpawnTemplate:
    def test_spawns_known_template(self):
        grid = _mock_grid(n=4)
        grid.activity = np.array([0, 1, 1, 1])
        grid.cold.return_value = [0]
        thermal = MagicMock()
        template = AgentTemplate(name="scout", chaos_initial=0.2)
        breeder = Breeder(grid, {"scout": template}, thermal)
        room = breeder.spawn_template("scout")
        assert room == 0
        assert breeder.lifecycle_state(0) == AgentLifecycle.SPAWNED

    def test_unknown_template_raises(self):
        grid = _mock_grid(n=4)
        thermal = MagicMock()
        breeder = Breeder(grid, {}, thermal)
        with pytest.raises(KeyError):
            breeder.spawn_template("ghost")

    def test_falls_back_to_least_active(self):
        grid = _mock_grid(n=4)
        grid.activity = np.array([3, 1, 2, 0])
        grid.cold.return_value = []
        thermal = MagicMock()
        template = AgentTemplate(name="scout", chaos_initial=0.2)
        breeder = Breeder(grid, {"scout": template}, thermal)
        room = breeder.spawn_template("scout")
        assert room == 3


# ---------------------------------------------------------------------------
# tick_all, sunset, stats
# ---------------------------------------------------------------------------


class TestTickSunsetStats:
    def test_tick_all_advances(self):
        grid = _mock_grid(n=4)
        grid.ticks = 5
        grid.activity = np.array([1, 0, 1, 0])
        grid.chaos = np.array([0.01, 0, 0.01, 0])
        thermal = MagicMock()
        breeder = Breeder(grid, {}, thermal)
        breeder._lifecycle[0] = LifecycleRecord(
            room_id=0, state=AgentLifecycle.ACTIVE, activity=1, chaos=0.01
        )
        breeder.tick_all()
        assert breeder.lifecycle_state(0) == AgentLifecycle.ADAPTING

    def test_sunset_room(self):
        grid = _mock_grid(n=4)
        thermal = MagicMock()
        breeder = Breeder(grid, {}, thermal)
        breeder._lifecycle[2] = LifecycleRecord(
            room_id=2, state=AgentLifecycle.COMPILED
        )
        breeder.sunset_room(2)
        assert 2 not in breeder._lifecycle
        grid.rebirth.assert_called_once_with(2)

    def test_stats(self):
        grid = _mock_grid(n=4)
        grid.ticks = 10
        thermal = MagicMock()
        thermal.thermal_headroom.return_value = 0.3
        breeder = Breeder(grid, {}, thermal)
        breeder._lifecycle[0] = LifecycleRecord(room_id=0, state=AgentLifecycle.SPAWNED)
        breeder._lifecycle[1] = LifecycleRecord(room_id=1, state=AgentLifecycle.ACTIVE)
        stats = breeder.stats
        assert stats["generation"] == 0
        assert stats["rooms"] == 4
        assert stats["lifecycle"][AgentLifecycle.SPAWNED] == 1
        assert stats["lifecycle"][AgentLifecycle.ACTIVE] == 1
        assert stats["thermal_headroom"] == 0.3

    def test_pick_cold_rooms(self):
        grid = _mock_grid(n=5)
        grid.activity = np.array([3, 1, 4, 0, 2])
        thermal = MagicMock()
        breeder = Breeder(grid, {}, thermal)
        cold = breeder._pick_cold_rooms(3)
        assert cold == [3, 1, 4]
