"""
Tests for Plato Room Auto-Sync.

Covers: RoomContext, RoomMapping, trinity_room_mapper, PlatoRoomSync
"""

import pytest

from fleet.spatial_projector import SpatialProjector, WorldState
from fleet.plato_sync import (
    RoomContext,
    RoomMapping,
    trinity_room_mapper,
    PlatoRoomSync,
)


class TestTrinityRoomMapper:
    def test_ethos_mapping(self):
        mapping = trinity_room_mapper("ethos-thermal")
        assert mapping.room_type == "ethos"
        assert mapping.default_position == (0.0, 0.0, 0.0)
        assert mapping.default_semantics["domain"] == "values"

    def test_pathos_mapping(self):
        mapping = trinity_room_mapper("pathos-emotional")
        assert mapping.room_type == "pathos"
        assert mapping.default_position == (100.0, 0.0, 0.0)
        assert mapping.default_semantics["domain"] == "emotions"

    def test_logos_mapping(self):
        mapping = trinity_room_mapper("logos-reasoning")
        assert mapping.room_type == "logos"
        assert mapping.default_position == (50.0, 100.0, 0.0)
        assert mapping.default_semantics["domain"] == "reason"

    def test_unknown_mapping(self):
        mapping = trinity_room_mapper("random-room")
        assert mapping.room_type == "general"
        assert mapping.default_position == (50.0, 50.0, 0.0)


class TestPlatoRoomSync:
    @pytest.fixture
    def sync(self):
        proj = SpatialProjector("node-test", dimension=3)
        return PlatoRoomSync(projector=proj)

    def test_on_enter(self, sync):
        sync.on_enter("agent-1", "ethos-thermal", context={"temperature": 65.4})
        assert sync.is_in_room("agent-1", "ethos-thermal")
        assert sync.get_room_population("ethos-thermal") == 1

    def test_on_enter_projects_state(self, sync):
        sync.on_enter("agent-1", "ethos-thermal", context={"temperature": 65.4})
        state = sync.projector.get_agent_state("agent-1")
        assert state is not None
        assert state.semantics.get("temperature") == 65.4
        assert state.semantics.get("event") == "enter"

    def test_on_exit(self, sync):
        sync.on_enter("agent-1", "ethos-thermal")
        sync.on_exit("agent-1", "ethos-thermal")
        assert not sync.is_in_room("agent-1", "ethos-thermal")
        assert sync.get_room_population("ethos-thermal") == 0

    def test_multiple_rooms(self, sync):
        sync.on_enter("agent-1", "ethos-thermal")
        sync.on_enter("agent-1", "pathos-emotional")
        rooms = sync.get_agent_rooms("agent-1")
        assert "ethos-thermal" in rooms
        assert "pathos-emotional" in rooms
        assert len(rooms) == 2

    def test_multiple_agents_same_room(self, sync):
        sync.on_enter("agent-1", "ethos-thermal")
        sync.on_enter("agent-2", "ethos-thermal")
        agents = sync.get_room_agents("ethos-thermal")
        assert "agent-1" in agents
        assert "agent-2" in agents
        assert len(agents) == 2

    def test_get_all_rooms(self, sync):
        sync.on_enter("agent-1", "ethos-thermal")
        sync.on_enter("agent-2", "pathos-emotional")
        rooms = sync.get_all_rooms()
        assert "ethos-thermal" in rooms
        assert "pathos-emotional" in rooms

    def test_trinity_distribution(self, sync):
        sync.on_enter("agent-1", "ethos-thermal")
        sync.on_enter("agent-2", "ethos-thermal")
        sync.on_enter("agent-3", "pathos-emotional")
        sync.on_enter("agent-4", "logos-reasoning")
        dist = sync.get_trinity_distribution()
        assert dist["ethos"] == 2
        assert dist["pathos"] == 1
        assert dist["logos"] == 1
        assert dist["other"] == 0

    def test_broadcast_to_room(self, sync):
        sync.on_enter("agent-1", "ethos-thermal")
        sync.on_enter("agent-2", "ethos-thermal")
        received = sync.broadcast_to_room("ethos-thermal", {"alert": "test"})
        assert "agent-1" in received
        assert "agent-2" in received

    def test_to_dict(self, sync):
        sync.on_enter("agent-1", "ethos-thermal")
        sync.on_enter("agent-2", "pathos-emotional")
        d = sync.to_dict()
        assert d["active_agents"] == 2
        assert "ethos-thermal" in d["active_rooms"]
        assert d["trinity_distribution"]["ethos"] == 1

    def test_callbacks(self, sync):
        events = []
        sync.on_enter_callback = lambda ctx: events.append(("enter", ctx.agent_id, ctx.room_id))
        sync.on_exit_callback = lambda ctx: events.append(("exit", ctx.agent_id, ctx.room_id))

        sync.on_enter("agent-1", "ethos-thermal")
        sync.on_exit("agent-1", "ethos-thermal")

        assert events == [
            ("enter", "agent-1", "ethos-thermal"),
            ("exit", "agent-1", "ethos-thermal"),
        ]

    def test_exit_with_auto_remove(self, sync):
        sync.on_enter("agent-1", "ethos-thermal")
        sync.on_exit("agent-1", "ethos-thermal")
        state = sync.projector.get_agent_state("agent-1")
        assert state.semantics.get("event") == "exit"

    def test_exit_without_auto_remove(self):
        proj = SpatialProjector("node-test", dimension=3)
        sync = PlatoRoomSync(projector=proj, auto_remove_on_exit=False)
        sync.on_enter("agent-1", "ethos-thermal")
        sync.on_exit("agent-1", "ethos-thermal")
        # State should still be there (not removed)
        state = sync.projector.get_agent_state("agent-1")
        assert state is not None
