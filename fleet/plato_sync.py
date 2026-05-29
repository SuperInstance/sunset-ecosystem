"""
Plato Room Auto-Sync

Automatically synchronizes Plato room entry/exit with the A2A Spatial
Projector. When a fleet agent enters a Plato room, its WorldState is
automatically projected into the spatial index. When it exits, the state
is aged out or removed.

This creates a seamless bridge between the Plato inter-agent intelligence
protocol and the fleet's spatial awareness layer.

Usage:
    sync = PlatoRoomSync(projector, room_mapper=trinity_room_mapper)
    sync.on_enter("agent-1", "ethos-thermal", context={"temperature": 65.4})
    sync.on_exit("agent-1", "ethos-thermal")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from fleet.spatial_projector import SpatialProjector, WorldState


@dataclass
class RoomContext:
    """Context for a room entry event."""
    room_id: str
    agent_id: str
    timestamp: float
    semantics: Dict[str, Any] = field(default_factory=dict)
    position: Optional[tuple] = None


@dataclass
class RoomMapping:
    """Maps Plato room IDs to spatial coordinates and semantics."""
    room_id: str
    default_position: tuple
    default_semantics: Dict[str, Any] = field(default_factory=dict)
    room_type: str = "general"


def trinity_room_mapper(room_id: str) -> RoomMapping:
    """
    Default room mapper for Trinity rooms (ethos, pathos, logos).
    Assigns spatial coordinates based on room type.
    """
    room_type = room_id.split("-")[0] if "-" in room_id else room_id

    mappings = {
        "ethos": RoomMapping(
            room_id=room_id,
            default_position=(0.0, 0.0, 0.0),
            default_semantics={"room_type": "ethos", "domain": "values"},
            room_type="ethos"
        ),
        "pathos": RoomMapping(
            room_id=room_id,
            default_position=(100.0, 0.0, 0.0),
            default_semantics={"room_type": "pathos", "domain": "emotions"},
            room_type="pathos"
        ),
        "logos": RoomMapping(
            room_id=room_id,
            default_position=(50.0, 100.0, 0.0),
            default_semantics={"room_type": "logos", "domain": "reason"},
            room_type="logos"
        ),
    }

    return mappings.get(room_type, RoomMapping(
        room_id=room_id,
        default_position=(50.0, 50.0, 0.0),
        default_semantics={"room_type": "general"},
        room_type="general"
    ))


class PlatoRoomSync:
    """
    Synchronizes Plato room events with the spatial projector.

    On room entry: Projects WorldState into spatial index
    On room exit: Removes or ages out the spatial projection
    """

    def __init__(self,
                 projector: SpatialProjector,
                 room_mapper: Optional[Callable[[str], RoomMapping]] = None,
                 auto_remove_on_exit: bool = True,
                 age_out_seconds: float = 60.0):
        self.projector = projector
        self.room_mapper = room_mapper or trinity_room_mapper
        self.auto_remove_on_exit = auto_remove_on_exit
        self.age_out_seconds = age_out_seconds

        # Track which agents are in which rooms
        self.agent_rooms: Dict[str, Set[str]] = {}
        self.room_contexts: Dict[str, RoomContext] = {}

        # Callbacks
        self.on_enter_callback: Optional[Callable[[RoomContext], None]] = None
        self.on_exit_callback: Optional[Callable[[RoomContext], None]] = None

    def on_enter(self, agent_id: str, room_id: str,
                 context: Optional[Dict[str, Any]] = None):
        """
        Called when an agent enters a Plato room.
        Projects the agent's WorldState into the spatial index.
        """
        context = context or {}

        # Record room membership
        if agent_id not in self.agent_rooms:
            self.agent_rooms[agent_id] = set()
        self.agent_rooms[agent_id].add(room_id)

        # Create room context
        room_ctx = RoomContext(
            room_id=room_id,
            agent_id=agent_id,
            timestamp=time.time(),
            semantics=context
        )
        self.room_contexts[f"{agent_id}:{room_id}"] = room_ctx

        # Map room to spatial coordinates
        mapping = self.room_mapper(room_id)

        # Build WorldState from room mapping + context
        position = context.get("position", mapping.default_position)
        semantics = mapping.default_semantics.copy()
        semantics.update(context)
        semantics["room_id"] = room_id
        semantics["event"] = "enter"

        # Project into spatial index
        self.projector.project_state(
            agent_id=agent_id,
            room_id=room_id,
            state=WorldState(
                position=position,
                semantics=semantics,
                confidence=1.0,
                agent_id=agent_id
            )
        )

        if self.on_enter_callback:
            self.on_enter_callback(room_ctx)

    def on_exit(self, agent_id: str, room_id: str):
        """
        Called when an agent exits a Plato room.
        Removes or ages out the spatial projection.
        """
        # Remove from room tracking
        if agent_id in self.agent_rooms:
            self.agent_rooms[agent_id].discard(room_id)
            if not self.agent_rooms[agent_id]:
                del self.agent_rooms[agent_id]

        key = f"{agent_id}:{room_id}"
        if key in self.room_contexts:
            room_ctx = self.room_contexts[key]
            del self.room_contexts[key]
        else:
            room_ctx = RoomContext(room_id=room_id, agent_id=agent_id,
                                   timestamp=time.time())

        if self.auto_remove_on_exit:
            # Re-project with exit semantics (or remove)
            mapping = self.room_mapper(room_id)
            self.projector.project_state(
                agent_id=agent_id,
                room_id=room_id,
                state=WorldState(
                    position=mapping.default_position,
                    semantics={
                        **mapping.default_semantics,
                        "room_id": room_id,
                        "event": "exit",
                        "timestamp": time.time()
                    },
                    confidence=0.5,  # Lower confidence on exit
                    agent_id=agent_id
                )
            )
        else:
            # Just age out naturally
            pass

        if self.on_exit_callback:
            self.on_exit_callback(room_ctx)

    def get_agent_rooms(self, agent_id: str) -> List[str]:
        """Get all rooms an agent is currently in."""
        return list(self.agent_rooms.get(agent_id, set()))

    def get_room_agents(self, room_id: str) -> List[str]:
        """Get all agents currently in a room."""
        return [
            agent_id for agent_id, rooms in self.agent_rooms.items()
            if room_id in rooms
        ]

    def is_in_room(self, agent_id: str, room_id: str) -> bool:
        """Check if an agent is in a specific room."""
        return room_id in self.agent_rooms.get(agent_id, set())

    def get_room_population(self, room_id: str) -> int:
        """Get number of agents in a room."""
        return len(self.get_room_agents(room_id))

    def get_all_rooms(self) -> Set[str]:
        """Get all rooms with at least one agent."""
        rooms = set()
        for agent_rooms in self.agent_rooms.values():
            rooms.update(agent_rooms)
        return rooms

    def get_trinity_distribution(self) -> Dict[str, int]:
        """Get agent distribution across Trinity rooms."""
        dist = {"ethos": 0, "pathos": 0, "logos": 0, "other": 0}
        for agent_id, rooms in self.agent_rooms.items():
            for room_id in rooms:
                room_type = room_id.split("-")[0] if "-" in room_id else room_id
                if room_type in dist:
                    dist[room_type] += 1
                else:
                    dist["other"] += 1
        return dist

    def broadcast_to_room(self, room_id: str, message: Dict[str, Any]) -> List[str]:
        """
        Broadcast a message to all agents in a room.
        Returns list of agent IDs that received the message.
        """
        agents = self.get_room_agents(room_id)
        # In real implementation, send network messages
        # For now, just record in semantics
        for agent_id in agents:
            state = self.projector.get_agent_state(agent_id)
            if state:
                state.semantics["last_message"] = message
                state.semantics["message_timestamp"] = time.time()
        return agents

    def to_dict(self) -> Dict:
        return {
            "active_agents": len(self.agent_rooms),
            "active_rooms": list(self.get_all_rooms()),
            "trinity_distribution": self.get_trinity_distribution(),
            "agent_rooms": {
                aid: list(rooms) for aid, rooms in self.agent_rooms.items()
            },
        }
