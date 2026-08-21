"""
Stable-WorldModel A2A Projector

Projects the fleet's spatial state into a stable-worldmodel environment,
allowing agents to navigate a shared virtual space using A2A spatial cards.

Key features:
- Auto-detects stable-worldmodel or falls back to mock
- Converts SpatialProjector state into worldmodel observations
- A2A-compatible spatial cards for inter-agent navigation
- Trajectory projection with collision prediction

Usage:
    from fleet.worldmodel_projector import WorldModelProjector
    projector = WorldModelProjector()
    projector.initialize_fleet_space(n_rooms=5, n_agents=10)
    obs = projector.get_observation("agent-1")
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fleet.spatial_projector import SpatialProjector, WorldState


@dataclass
class WorldModelObservation:
    """Observation from the worldmodel for an agent."""

    agent_id: str
    position: Tuple[float, ...]
    room_id: str
    nearby_agents: List[Tuple[str, float]] = field(default_factory=list)
    # Distance to each nearby agent
    semantic_features: Dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0

    def to_a2a_spatial_card(self) -> Dict[str, Any]:
        """Convert to A2A spatial card format."""
        return {
            "type": "spatial_observation",
            "agent_id": self.agent_id,
            "position": list(self.position),
            "room_id": self.room_id,
            "nearby_agents": [
                {"agent_id": aid, "distance": dist} for aid, dist in self.nearby_agents
            ],
            "semantic_features": self.semantic_features,
            "timestamp": self.timestamp,
        }


class WorldModelProjector:
    """
    Projects fleet spatial state into a worldmodel environment.

    Wraps stable-worldmodel (if available) or uses mock spatial simulation.
    """

    def __init__(self, spatial_projector: Optional[SpatialProjector] = None):
        self.spatial = spatial_projector or SpatialProjector("worldmodel-node")
        self.worldmodel = None
        self.mock_mode = True
        self._try_import_worldmodel()
        self.rooms: Dict[str, Dict[str, Any]] = {}
        self.agent_positions: Dict[str, Tuple[float, ...]] = {}

    def _try_import_worldmodel(self):
        """Try to import stable-worldmodel, fall back to mock."""
        try:
            import stable_worldmodel

            self.worldmodel = stable_worldmodel
            self.mock_mode = False
        except ImportError:
            self.mock_mode = True

    def initialize_fleet_space(
        self, n_rooms: int = 5, room_size: float = 100.0, n_agents: int = 10
    ):
        """Initialize a fleet navigation space with rooms."""
        self.rooms = {}
        for i in range(n_rooms):
            room_id = f"room_{i}"
            self.rooms[room_id] = {
                "id": room_id,
                "size": room_size,
                "position": (i * room_size, 0.0, 0.0),
                "agents": set(),
            }
            # Register with spatial projector
            self.spatial.update_world(room_id, [0.0, 0.0, 0.0], "room")

        # Place agents randomly
        for j in range(n_agents):
            agent_id = f"agent_{j}"
            room = random.choice(list(self.rooms.keys()))
            pos = self._random_position_in_room(room)
            self.agent_positions[agent_id] = pos
            self.rooms[room]["agents"].add(agent_id)
            self.spatial.register_agent(agent_id, pos)

    def _random_position_in_room(self, room_id: str) -> Tuple[float, ...]:
        """Generate random position within a room."""
        room = self.rooms[room_id]
        base = room["position"]
        size = room["size"]
        return tuple(base[d] + random.uniform(0, size) for d in range(3))

    def move_agent(self, agent_id: str, delta: Tuple[float, ...]) -> bool:
        """Move an agent by a delta. Returns True if move succeeded."""
        if agent_id not in self.agent_positions:
            return False

        old_pos = self.agent_positions[agent_id]
        new_pos = tuple(old_pos[d] + delta[d] for d in range(len(old_pos)))

        # Check room boundaries
        room_id = self._get_room_for_agent(agent_id)
        if room_id and not self._in_room_bounds(new_pos, room_id):
            return False

        self.agent_positions[agent_id] = new_pos
        self.spatial.update_agent(agent_id, new_pos)
        return True

    def _get_room_for_agent(self, agent_id: str) -> Optional[str]:
        """Find which room an agent is in."""
        for room_id, room in self.rooms.items():
            if agent_id in room["agents"]:
                return room_id
        return None

    def _in_room_bounds(self, pos: Tuple[float, ...], room_id: str) -> bool:
        """Check if position is within room bounds."""
        room = self.rooms[room_id]
        base = room["position"]
        size = room["size"]
        for d in range(3):
            if not (base[d] <= pos[d] <= base[d] + size):
                return False
        return True

    def get_observation(self, agent_id: str) -> WorldModelObservation:
        """Get worldmodel observation for an agent."""
        if agent_id not in self.agent_positions:
            return WorldModelObservation(
                agent_id=agent_id,
                position=(0.0, 0.0, 0.0),
                room_id="unknown",
            )

        pos = self.agent_positions[agent_id]
        room_id = self._get_room_for_agent(agent_id) or "unknown"

        # Find nearby agents
        nearby = []
        for other_id, other_pos in self.agent_positions.items():
            if other_id == agent_id:
                continue
            dist = np.sqrt(sum((pos[d] - other_pos[d]) ** 2 for d in range(len(pos))))
            if dist < 50.0:  # Perception radius
                nearby.append((other_id, float(dist)))

        nearby.sort(key=lambda x: x[1])

        # Get semantic features from spatial projector
        state = self.spatial.get_agent_state(agent_id)
        semantics = state.semantics if state else {}

        return WorldModelObservation(
            agent_id=agent_id,
            position=pos,
            room_id=room_id,
            nearby_agents=nearby[:5],  # Top 5 nearest
            semantic_features={
                k: float(v) for k, v in semantics.items() if isinstance(v, (int, float))
            },
            timestamp=random.random() * 1000,  # Mock timestamp
        )

    def get_all_observations(self) -> List[WorldModelObservation]:
        """Get observations for all agents."""
        return [self.get_observation(aid) for aid in self.agent_positions]

    def predict_collision(
        self, agent_id: str, trajectory: List[Tuple[float, ...]]
    ) -> Optional[str]:
        """
        Predict if trajectory collides with another agent.
        Returns agent_id of predicted collision, or None.
        """
        if agent_id not in self.agent_positions:
            return None

        for point in trajectory:
            for other_id, other_pos in self.agent_positions.items():
                if other_id == agent_id:
                    continue
                dist = np.sqrt(
                    sum((point[d] - other_pos[d]) ** 2 for d in range(len(point)))
                )
                if dist < 5.0:  # Collision threshold
                    return other_id
        return None

    def to_fleet_state(self) -> Dict[str, Any]:
        """Export full fleet state as dictionary."""
        return {
            "rooms": {
                rid: {
                    "id": rid,
                    "size": r["size"],
                    "position": list(r["position"]),
                    "agents": list(r["agents"]),
                }
                for rid, r in self.rooms.items()
            },
            "agents": {
                aid: {
                    "position": list(pos),
                    "room": self._get_room_for_agent(aid),
                }
                for aid, pos in self.agent_positions.items()
            },
            "n_agents": len(self.agent_positions),
            "n_rooms": len(self.rooms),
            "mock_mode": self.mock_mode,
        }

    def get_a2a_spatial_broadcast(self) -> List[Dict[str, Any]]:
        """Get all observations as A2A spatial cards."""
        return [obs.to_a2a_spatial_card() for obs in self.get_all_observations()]

    def step(
        self, actions: Dict[str, Tuple[float, ...]]
    ) -> Dict[str, WorldModelObservation]:
        """
        Execute one step: apply actions, update positions, return observations.
        Simulates a worldmodel step.
        """
        for agent_id, delta in actions.items():
            self.move_agent(agent_id, delta)

        return {aid: self.get_observation(aid) for aid in self.agent_positions}
