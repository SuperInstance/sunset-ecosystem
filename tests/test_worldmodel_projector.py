"""
Tests for Stable-WorldModel A2A Projector.

Covers: WorldModelObservation, WorldModelProjector.
"""

import pytest

from fleet.worldmodel_projector import WorldModelObservation, WorldModelProjector
from fleet.spatial_projector import SpatialProjector


class TestWorldModelObservation:
    def test_to_a2a_spatial_card(self):
        obs = WorldModelObservation(
            agent_id="agent-1",
            position=(10.0, 20.0, 0.0),
            room_id="room_0",
            nearby_agents=[("agent-2", 15.0)],
            semantic_features={"temperature": 65.0},
            timestamp=1000.0,
        )
        card = obs.to_a2a_spatial_card()
        assert card["type"] == "spatial_observation"
        assert card["agent_id"] == "agent-1"
        assert card["position"] == [10.0, 20.0, 0.0]
        assert card["room_id"] == "room_0"
        assert len(card["nearby_agents"]) == 1
        assert card["nearby_agents"][0]["agent_id"] == "agent-2"


class TestWorldModelProjector:
    def test_init(self, monkeypatch):
        # Patch _try_import_worldmodel to avoid hanging on stable_worldmodel import
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        proj = WorldModelProjector()
        assert proj.mock_mode is True
        assert proj.worldmodel is None

    def test_init_with_spatial(self, monkeypatch):
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        spatial = SpatialProjector("test-node")
        proj = WorldModelProjector(spatial_projector=spatial)
        assert proj.spatial is spatial

    def test_initialize_fleet_space(self, monkeypatch):
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        proj = WorldModelProjector()
        proj.initialize_fleet_space(n_rooms=3, n_agents=5)
        assert len(proj.rooms) == 3
        assert len(proj.agent_positions) == 5

    def test_agent_in_room(self, monkeypatch):
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        proj = WorldModelProjector()
        proj.initialize_fleet_space(n_rooms=1, n_agents=1)
        room_id = proj._get_room_for_agent("agent_0")
        assert room_id is not None
        assert room_id.startswith("room_")

    def test_move_agent(self, monkeypatch):
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        proj = WorldModelProjector()
        proj.initialize_fleet_space(n_rooms=1, n_agents=1, room_size=100.0)
        pos_before = proj.agent_positions["agent_0"]
        result = proj.move_agent("agent_0", (5.0, 5.0, 0.0))
        pos_after = proj.agent_positions["agent_0"]
        assert result is True
        assert pos_after != pos_before

    def test_move_agent_out_of_bounds(self, monkeypatch):
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        proj = WorldModelProjector()
        proj.initialize_fleet_space(n_rooms=1, n_agents=1, room_size=10.0)
        result = proj.move_agent("agent_0", (100.0, 0.0, 0.0))
        assert result is False

    def test_move_nonexistent_agent(self, monkeypatch):
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        proj = WorldModelProjector()
        result = proj.move_agent("nonexistent", (1.0, 0.0, 0.0))
        assert result is False

    def test_get_observation(self, monkeypatch):
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        proj = WorldModelProjector()
        proj.initialize_fleet_space(n_rooms=2, n_agents=3)
        obs = proj.get_observation("agent_0")
        assert obs.agent_id == "agent_0"
        assert obs.room_id is not None
        assert len(obs.position) == 3

    def test_get_observation_unknown_agent(self, monkeypatch):
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        proj = WorldModelProjector()
        obs = proj.get_observation("nonexistent")
        assert obs.agent_id == "nonexistent"
        assert obs.room_id == "unknown"

    def test_get_all_observations(self, monkeypatch):
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        proj = WorldModelProjector()
        proj.initialize_fleet_space(n_rooms=2, n_agents=5)
        all_obs = proj.get_all_observations()
        assert len(all_obs) == 5
        assert all(isinstance(o, WorldModelObservation) for o in all_obs)

    def test_predict_collision(self, monkeypatch):
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        proj = WorldModelProjector()
        proj.initialize_fleet_space(n_rooms=1, n_agents=2, room_size=100.0)
        # Place agents close together
        proj.agent_positions["agent_0"] = (0.0, 0.0, 0.0)
        proj.agent_positions["agent_1"] = (1.0, 0.0, 0.0)
        trajectory = [(0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
        collision = proj.predict_collision("agent_0", trajectory)
        assert collision == "agent_1"

    def test_predict_no_collision(self, monkeypatch):
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        proj = WorldModelProjector()
        proj.initialize_fleet_space(n_rooms=1, n_agents=2, room_size=100.0)
        proj.agent_positions["agent_0"] = (0.0, 0.0, 0.0)
        proj.agent_positions["agent_1"] = (50.0, 50.0, 0.0)
        trajectory = [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        collision = proj.predict_collision("agent_0", trajectory)
        assert collision is None

    def test_to_fleet_state(self, monkeypatch):
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        proj = WorldModelProjector()
        proj.initialize_fleet_space(n_rooms=2, n_agents=3)
        state = proj.to_fleet_state()
        assert state["n_rooms"] == 2
        assert state["n_agents"] == 3
        assert state["mock_mode"] is True
        assert "rooms" in state
        assert "agents" in state

    def test_get_a2a_spatial_broadcast(self, monkeypatch):
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        proj = WorldModelProjector()
        proj.initialize_fleet_space(n_rooms=1, n_agents=2)
        cards = proj.get_a2a_spatial_broadcast()
        assert len(cards) == 2
        assert all(c["type"] == "spatial_observation" for c in cards)

    def test_step(self, monkeypatch):
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        proj = WorldModelProjector()
        proj.initialize_fleet_space(n_rooms=1, n_agents=2, room_size=100.0)
        actions = {
            "agent_0": (5.0, 0.0, 0.0),
            "agent_1": (0.0, 5.0, 0.0),
        }
        observations = proj.step(actions)
        assert len(observations) == 2
        assert "agent_0" in observations
        assert "agent_1" in observations

    def test_random_position_in_room(self, monkeypatch):
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        proj = WorldModelProjector()
        proj.initialize_fleet_space(n_rooms=1, n_agents=0, room_size=50.0)
        pos = proj._random_position_in_room("room_0")
        assert len(pos) == 3
        assert 0 <= pos[0] <= 50.0
        assert 0 <= pos[1] <= 50.0

    def test_in_room_bounds(self, monkeypatch):
        monkeypatch.setattr(
            WorldModelProjector, "_try_import_worldmodel", lambda self: None
        )
        proj = WorldModelProjector()
        proj.initialize_fleet_space(n_rooms=1, n_agents=0, room_size=10.0)
        assert proj._in_room_bounds((5.0, 5.0, 5.0), "room_0") is True
        assert proj._in_room_bounds((15.0, 5.0, 5.0), "room_0") is False
        assert proj._in_room_bounds((-1.0, 5.0, 5.0), "room_0") is False
