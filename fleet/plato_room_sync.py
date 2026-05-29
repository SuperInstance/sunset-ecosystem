"""Plato Room Auto-Sync — Auto-project WorldState on room entry/exit."""

from typing import Callable, Dict, List, Optional
from dataclasses import dataclass, field
import time

from fleet.spatial_projector import WorldState, SpatialProjector


@dataclass
class RoomTransition:
    """Record of a room transition."""
    agent_id: str
    from_room: Optional[str]
    to_room: Optional[str]
    state: WorldState
    timestamp: float = field(default_factory=time.time)


class PlatoRoomSync:
    """Auto-syncs agent WorldState with the spatial projector on room entry/exit."""

    def __init__(self, projector: SpatialProjector):
        self.projector = projector
        self._agent_rooms: Dict[str, str] = {}
        self._transitions: List[RoomTransition] = []
        self._callbacks: List[Callable] = []

    def on_enter(self, agent_id: str, room_id: str, state: WorldState) -> None:
        """Called when an agent enters a room."""
        old_room = self._agent_rooms.get(agent_id)

        if old_room is not None and old_room != room_id:
            # Transition: exit old first
            self.on_exit(agent_id, old_room, state)

        # Project state into new room
        state.agent_id = agent_id
        state.room_id = room_id
        self.projector.update(state)
        self._agent_rooms[agent_id] = room_id

        transition = RoomTransition(
            agent_id=agent_id,
            from_room=old_room,
            to_room=room_id,
            state=state,
        )
        self._transitions.append(transition)
        self._notify(transition)

    def on_exit(self, agent_id: str, room_id: str, state: WorldState) -> None:
        """Called when an agent exits a room."""
        current_room = self._agent_rooms.get(agent_id)
        if current_room != room_id:
            # Agent is not in this room, ignore or handle mismatch
            return

        # Mark as departed (remove or flag)
        state.agent_id = agent_id
        state.room_id = room_id
        # Remove from spatial index
        try:
            self.projector.remove(state)
        except Exception:
            pass  # May not exist in projector

        self._agent_rooms[agent_id] = None
        transition = RoomTransition(
            agent_id=agent_id,
            from_room=room_id,
            to_room=None,
            state=state,
        )
        self._transitions.append(transition)
        self._notify(transition)

    def transition(self, agent_id: str, from_room: str, to_room: str, state: WorldState) -> None:
        """Atomic room transition: exit old, enter new."""
        self.on_exit(agent_id, from_room, state)
        self.on_enter(agent_id, to_room, state)

    def get_room(self, agent_id: str) -> Optional[str]:
        """Current room for an agent."""
        return self._agent_rooms.get(agent_id)

    def get_transitions(self, agent_id: Optional[str] = None) -> List[RoomTransition]:
        """Get transition history, optionally filtered by agent."""
        if agent_id is None:
            return list(self._transitions)
        return [t for t in self._transitions if t.agent_id == agent_id]

    def register_callback(self, fn: Callable[[RoomTransition], None]) -> None:
        self._callbacks.append(fn)

    def unregister_callback(self, fn: Callable[[RoomTransition], None]) -> None:
        if fn in self._callbacks:
            self._callbacks.remove(fn)

    def _notify(self, transition: RoomTransition) -> None:
        for fn in self._callbacks:
            try:
                fn(transition)
            except Exception:
                pass

    def get_stats(self) -> Dict:
        return {
            "agents_tracked": len(self._agent_rooms),
            "transitions": len(self._transitions),
            "callbacks": len(self._callbacks),
        }

    def to_dict(self) -> Dict:
        return {
            "projector_type": type(self.projector).__name__,
            "stats": self.get_stats(),
        }
