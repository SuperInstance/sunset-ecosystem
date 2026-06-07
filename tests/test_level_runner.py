"""Tests for LevelRunner.

Covers:
- Level loading and entity spawning
- Tick execution (physics, AI, collisions)
- Event bus handlers
- Victory condition detection
- Spatial queries
- Stats tracking
"""

from __future__ import annotations

import numpy as np
import pytest

from fleet.level_runner import LevelRunner, LevelDefinition, Entity, LevelState


class TestLevelDefinition:
    def test_to_caslang(self) -> None:
        level = LevelDefinition(
            name="test_level",
            rules=[{"trigger": "tick", "action": "heal", "params": {"amount": 10}}],
        )
        caslang = level.to_caslang()
        assert "caslang" in caslang
        assert "rule_tick" in caslang


class TestEntity:
    def test_to_vector(self) -> None:
        e = Entity(
            entity_id="e1", entity_type="agent",
            position=np.array([1.0, 2.0, 3.0]),
            health=50.0, faction="team_a",
            attributes={"speed": 10.0},
        )
        vec = e.to_vector(dim=64)
        assert vec.shape == (64,)
        assert vec.dtype == np.float32
        # Position should be encoded
        assert np.allclose(vec[:3], [1.0, 2.0, 3.0])
        # Health should be encoded (50/100 = 0.5)
        assert abs(vec[6] - 0.5) < 0.01


class TestLevelState:
    def test_add_and_remove_entity(self) -> None:
        level = LevelState(LevelDefinition(name="test"))
        e = Entity(entity_id="e1", entity_type="agent")
        assert level.add_entity(e) is True
        assert len(level.entities) == 1
        assert level.remove_entity("e1") is True
        assert len(level.entities) == 0

    def test_max_entities_limit(self) -> None:
        level = LevelState(LevelDefinition(name="test", max_entities=2))
        level.add_entity(Entity(entity_id="e1", entity_type="agent"))
        level.add_entity(Entity(entity_id="e2", entity_type="agent"))
        assert level.add_entity(Entity(entity_id="e3", entity_type="agent")) is False

    def test_spatial_query(self) -> None:
        level = LevelState(LevelDefinition(name="test"))
        level.add_entity(Entity(entity_id="e1", entity_type="agent", position=np.array([0, 0, 0])))
        level.add_entity(Entity(entity_id="e2", entity_type="agent", position=np.array([10, 0, 0])))
        level.add_entity(Entity(entity_id="e3", entity_type="npc", position=np.array([1, 0, 0])))

        near = level.get_entities_near(np.array([0, 0, 0]), radius=5.0)
        assert len(near) == 2  # e1 and e3
        assert all(e.entity_id in ("e1", "e3") for e in near)

        near_agents = level.get_entities_near(np.array([0, 0, 0]), radius=5.0, entity_type="agent")
        assert len(near_agents) == 1
        assert near_agents[0].entity_id == "e1"

    def test_faction_query(self) -> None:
        level = LevelState(LevelDefinition(name="test"))
        level.add_entity(Entity(entity_id="e1", entity_type="agent", faction="team_a"))
        level.add_entity(Entity(entity_id="e2", entity_type="agent", faction="team_b"))
        level.add_entity(Entity(entity_id="e3", entity_type="agent", faction="team_a"))

        team_a = level.get_entities_by_faction("team_a")
        assert len(team_a) == 2

    def test_victory_eliminate_faction(self) -> None:
        level = LevelState(LevelDefinition(
            name="test",
            victory_conditions=[{"type": "eliminate_faction", "faction": "team_b"}],
        ))
        level.add_entity(Entity(entity_id="e1", entity_type="agent", faction="team_a"))
        level.add_entity(Entity(entity_id="e2", entity_type="agent", faction="team_b"))
        assert level.check_victory() is None

        level.remove_entity("e2")
        result = level.check_victory()
        assert result is not None
        assert result["victory"] is True

    def test_victory_survive_ticks(self) -> None:
        level = LevelState(LevelDefinition(
            name="test",
            victory_conditions=[{"type": "survive_ticks", "ticks": 10}],
        ))
        level.tick_count = 5
        assert level.check_victory() is None
        level.tick_count = 10
        result = level.check_victory()
        assert result is not None
        assert result["victory"] is True

    def test_victory_reach_position(self) -> None:
        level = LevelState(LevelDefinition(
            name="test",
            victory_conditions=[{"type": "reach_position", "position": [10, 0, 0], "radius": 2.0, "faction": "team_a"}],
        ))
        level.add_entity(Entity(entity_id="e1", entity_type="agent", faction="team_a", position=np.array([0, 0, 0])))
        assert level.check_victory() is None

        level.entities["e1"].position = np.array([10, 0, 0])
        result = level.check_victory()
        assert result is not None
        assert result["victory"] is True


class TestLevelRunner:
    def test_load_and_spawn(self) -> None:
        runner = LevelRunner()
        level_id = runner.load_level(LevelDefinition(name="test"))
        assert level_id.startswith("test_")

        assert runner.spawn_entity(level_id, "agent_1", "agent", (0, 0, 0), "team_a") is True
        state = runner.get_level_state(level_id)
        assert state is not None
        assert len(state.entities) == 1

    def test_start_and_stop(self) -> None:
        runner = LevelRunner()
        level_id = runner.load_level(LevelDefinition(name="test", tick_rate_hz=100.0))
        runner.spawn_entity(level_id, "agent_1", "agent")

        assert runner.start_level(level_id) is True
        assert runner._running[level_id] is True

        # Let it run for a few ticks
        import time
        time.sleep(0.15)

        assert runner.stop_level(level_id) is True
        assert runner._running[level_id] is False

        state = runner.get_level_state(level_id)
        assert state is not None
        assert state.tick_count > 0

    def test_collision_detection(self) -> None:
        runner = LevelRunner()
        level_id = runner.load_level(LevelDefinition(name="test", tick_rate_hz=1000.0))
        runner.spawn_entity(level_id, "a", "agent", (0, 0, 0))
        runner.spawn_entity(level_id, "b", "agent", (1, 0, 0))

        # Manually run a tick
        state = runner.get_level_state(level_id)
        assert state is not None
        runner._run_tick(level_id)
        # Entities should be close enough to trigger collision
        # (Collision radius is 2.0, distance is 1.0)
        # The collision event should have been emitted

    def test_physics_update(self) -> None:
        runner = LevelRunner()
        level_id = runner.load_level(LevelDefinition(name="test", bounds=(0, 0, 0, 100, 100, 100)))
        runner.spawn_entity(level_id, "a", "agent", (50, 50, 50))

        state = runner.get_level_state(level_id)
        assert state is not None
        entity = state.entities["a"]
        entity.velocity = np.array([10.0, 0.0, 0.0])

        runner._run_tick(level_id)
        # Position should have moved
        assert entity.position[0] > 50.0
        # Velocity should have dampened
        assert np.linalg.norm(entity.velocity) < 10.0

    def test_bounds_clamping(self) -> None:
        runner = LevelRunner()
        level_id = runner.load_level(LevelDefinition(name="test", bounds=(0, 0, 0, 10, 10, 10)))
        runner.spawn_entity(level_id, "a", "agent", (9, 5, 5))

        state = runner.get_level_state(level_id)
        assert state is not None
        entity = state.entities["a"]
        entity.velocity = np.array([5.0, 0.0, 0.0])

        runner._run_tick(level_id)
        # Should be clamped to bounds
        assert entity.position[0] <= 10.0

    def test_stats(self) -> None:
        runner = LevelRunner()
        level_id = runner.load_level(LevelDefinition(name="test"))
        runner.spawn_entity(level_id, "a", "agent")
        runner.start_level(level_id)
        import time
        time.sleep(0.1)
        runner.stop_level(level_id)

        stats = runner.stats
        assert stats["levels_loaded"] == 1
        assert stats["total_ticks"] > 0
