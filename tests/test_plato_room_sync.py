"""Tests for PlatoRoomSync — auto-sync agent WorldState on room entry/exit.

Covers enter, exit, transition, callbacks, transition history, and stats.
"""

import pytest
from unittest.mock import MagicMock

from fleet.plato_room_sync import (
    RoomTransition,
    PlatoRoomSync,
)
from fleet.spatial_projector import WorldState

def _ws():
    return WorldState(position=(0.0,))


# ---------------------------------------------------------------------------
# RoomTransition
# ---------------------------------------------------------------------------

class TestRoomTransition:
    def test_defaults(self):
        state = _ws()
        rt = RoomTransition(agent_id="a1", from_room="r1", to_room="r2", state=state)
        assert rt.agent_id == "a1"
        assert rt.from_room == "r1"
        assert rt.to_room == "r2"
        assert rt.state is state
        assert rt.timestamp > 0

    def test_to_dict_serde(self):
        state = _ws()
        rt = RoomTransition(agent_id="a1", from_room=None, to_room="r2", state=state)
        assert rt.to_room == "r2"
        assert rt.from_room is None


# ---------------------------------------------------------------------------
# PlatoRoomSync init
# ---------------------------------------------------------------------------

class TestPlatoRoomSyncInit:
    def test_default(self):
        projector = MagicMock()
        sync = PlatoRoomSync(projector)
        assert sync.projector is projector
        assert sync.get_stats()["agents_tracked"] == 0
        assert sync.get_stats()["transitions"] == 0


# ---------------------------------------------------------------------------
# Enter / Exit
# ---------------------------------------------------------------------------

class TestEnterExit:
    def test_on_enter_tracks_room(self):
        projector = MagicMock()
        sync = PlatoRoomSync(projector)
        state = _ws()
        sync.on_enter("a1", "roomA", state)
        assert sync.get_room("a1") == "roomA"
        projector.update.assert_called_once()

    def test_on_enter_sets_ids(self):
        projector = MagicMock()
        sync = PlatoRoomSync(projector)
        state = _ws()
        sync.on_enter("a1", "roomA", state)
        assert state.agent_id == "a1"
        assert state.room_id == "roomA"

    def test_on_exit_clears_room(self):
        projector = MagicMock()
        sync = PlatoRoomSync(projector)
        state = _ws()
        sync.on_enter("a1", "roomA", state)
        sync.on_exit("a1", "roomA", state)
        assert sync.get_room("a1") is None
        projector.remove.assert_called_once()

    def test_on_exit_wrong_room_ignored(self):
        projector = MagicMock()
        sync = PlatoRoomSync(projector)
        state = _ws()
        sync.on_enter("a1", "roomA", state)
        projector.reset_mock()
        sync.on_exit("a1", "roomB", state)
        projector.remove.assert_not_called()
        assert sync.get_room("a1") == "roomA"

    def test_double_enter_moves_room(self):
        projector = MagicMock()
        sync = PlatoRoomSync(projector)
        state1 = _ws()
        state2 = _ws()
        sync.on_enter("a1", "roomA", state1)
        sync.on_enter("a1", "roomB", state2)
        assert sync.get_room("a1") == "roomB"
        assert projector.update.call_count == 2
        projector.remove.assert_called_once()

    def test_exit_not_entered(self):
        projector = MagicMock()
        sync = PlatoRoomSync(projector)
        state = _ws()
        sync.on_exit("ghost", "roomA", state)
        assert sync.get_room("ghost") is None


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------

class TestTransition:
    def test_atomic_transition(self):
        projector = MagicMock()
        sync = PlatoRoomSync(projector)
        state = _ws()
        sync.on_enter("a1", "roomA", state)
        sync.transition("a1", "roomA", "roomB", state)
        assert sync.get_room("a1") == "roomB"

    def test_transition_records_history(self):
        projector = MagicMock()
        sync = PlatoRoomSync(projector)
        state = _ws()
        sync.on_enter("a1", "roomA", state)
        sync.transition("a1", "roomA", "roomB", state)
        history = sync.get_transitions("a1")
        assert len(history) == 3


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TestCallbacks:
    def test_register_and_notify(self):
        projector = MagicMock()
        sync = PlatoRoomSync(projector)
        called = []

        def cb(t):
            called.append(t)

        sync.register_callback(cb)
        state = _ws()
        sync.on_enter("a1", "roomA", state)
        assert len(called) == 1
        assert called[0].agent_id == "a1"

    def test_unregister(self):
        projector = MagicMock()
        sync = PlatoRoomSync(projector)
        called = []

        def cb(t):
            called.append(t)

        sync.register_callback(cb)
        sync.unregister_callback(cb)
        state = _ws()
        sync.on_enter("a1", "roomA", state)
        assert len(called) == 0

    def test_callback_exception_isolated(self):
        projector = MagicMock()
        sync = PlatoRoomSync(projector)
        good_called = []

        def bad_cb(t):
            raise RuntimeError("boom")

        def good_cb(t):
            good_called.append(t)

        sync.register_callback(bad_cb)
        sync.register_callback(good_cb)
        state = _ws()
        sync.on_enter("a1", "roomA", state)
        assert len(good_called) == 1


# ---------------------------------------------------------------------------
# History & Stats
# ---------------------------------------------------------------------------

class TestHistoryAndStats:
    def test_get_transitions_filtered(self):
        projector = MagicMock()
        sync = PlatoRoomSync(projector)
        sync.on_enter("a1", "roomA", _ws())
        sync.on_enter("a2", "roomB", _ws())
        assert len(sync.get_transitions("a1")) == 1
        assert len(sync.get_transitions("a2")) == 1
        assert len(sync.get_transitions()) == 2

    def test_get_room_missing(self):
        projector = MagicMock()
        sync = PlatoRoomSync(projector)
        assert sync.get_room("nobody") is None

    def test_stats(self):
        projector = MagicMock()
        sync = PlatoRoomSync(projector)
        sync.on_enter("a1", "roomA", _ws())
        sync.on_enter("a2", "roomB", _ws())
        sync.transition("a1", "roomA", "roomC", _ws())
        stats = sync.get_stats()
        assert stats["agents_tracked"] == 2
        assert stats["transitions"] == 4
        assert stats["callbacks"] == 0

    def test_to_dict(self):
        projector = MagicMock()
        sync = PlatoRoomSync(projector)
        d = sync.to_dict()
        assert "projector_type" in d
        assert "stats" in d
