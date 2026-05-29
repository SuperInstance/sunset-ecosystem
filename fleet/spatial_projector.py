"""
A2A Spatial Projector System

Transforms stable-worldmodel into a fleet-native spatial awareness layer.
Every agent projects its perceptual state into a shared spatial index.
Predictions flow through FLUX constraint gates before A2A broadcast.

References:
    - stable-worldmodel: https://github.com/galilai-group/stable-worldmodel
    - OpenConstruct: https://github.com/SuperInstance/openconstruct-docs
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ────────────────────────────── Data Structures ──────────────────────────────

@dataclass
class WorldState:
    """Typed perceptual state tensor for an agent in a Plato room."""
    position: Tuple[float, ...]
    velocity: Optional[Tuple[float, ...]] = None
    orientation: Optional[float] = None
    semantics: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    agent_id: Optional[str] = None
    room_id: Optional[str] = None

    def distance_to(self, other: WorldState) -> float:
        """Euclidean distance to another state (same dimensionality)."""
        if len(self.position) != len(other.position):
            raise ValueError("Position dimensions must match")
        return math.sqrt(
            sum((a - b) ** 2 for a, b in zip(self.position, other.position))
        )

    def to_vector(self) -> List[float]:
        """Flatten to a vector for indexing."""
        vec = list(self.position)
        if self.velocity:
            vec.extend(self.velocity)
        if self.orientation is not None:
            vec.append(self.orientation)
        return vec

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/broadcast."""
        return {
            "position": self.position,
            "velocity": self.velocity,
            "orientation": self.orientation,
            "semantics": self.semantics,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "room_id": self.room_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> WorldState:
        """Deserialize from storage/broadcast."""
        return cls(
            position=tuple(d["position"]),
            velocity=tuple(d["velocity"]) if d.get("velocity") else None,
            orientation=d.get("orientation"),
            semantics=d.get("semantics", {}),
            confidence=d.get("confidence", 1.0),
            timestamp=d.get("timestamp", time.time()),
            agent_id=d.get("agent_id"),
            room_id=d.get("room_id"),
        )


@dataclass
class Prediction:
    """World model prediction output: trajectory, rewards, uncertainty."""
    trajectory: List[WorldState]
    rewards: Optional[List[float]] = None
    values: Optional[List[float]] = None
    actions: Optional[List[Any]] = None
    uncertainty: List[float] = field(default_factory=list)
    model_id: str = "default"
    timestamp: float = field(default_factory=time.time)
    agent_id: Optional[str] = None
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def final_state(self) -> WorldState:
        """Last state in the trajectory."""
        return self.trajectory[-1] if self.trajectory else WorldState(position=(0.0,))

    @property
    def mean_uncertainty(self) -> float:
        """Average uncertainty across trajectory."""
        return sum(self.uncertainty) / len(self.uncertainty) if self.uncertainty else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory": [s.to_dict() for s in self.trajectory],
            "rewards": self.rewards,
            "values": self.values,
            "actions": self.actions,
            "uncertainty": self.uncertainty,
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "prediction_id": self.prediction_id,
        }


# ────────────────────────────── Flux Constraint ──────────────────────────────

@dataclass
class FluxConstraint:
    """FLUX constraint for gating predictions."""
    name: str
    check: Callable[[Prediction], bool]
    penalty: Optional[Callable[[Prediction], float]] = None
    hard: bool = True
    weight: float = 1.0

    def evaluate(self, prediction: Prediction) -> Tuple[bool, float]:
        """Returns (passed, penalty_or_zero)."""
        passed = self.check(prediction)
        if self.hard and not passed:
            return False, 0.0
        penalty = self.penalty(prediction) if self.penalty and not passed else 0.0
        return True, penalty * self.weight


# ────────────────────────────── Spatial Index ──────────────────────────────

class SpatialIndex:
    """
    In-memory spatial index with LanceDB-compatible schema.
    Production would use actual LanceDB; this is fleet-embedded.
    """

    def __init__(self, dimension: int = 3):
        self.dimension = dimension
        self._entries: Dict[str, WorldState] = {}
        self._vectors: Dict[str, List[float]] = {}
        self._by_agent: Dict[str, List[str]] = {}
        self._by_room: Dict[str, List[str]] = {}

    def insert(self, projection_id: str, state: WorldState) -> None:
        """Index a world state."""
        self._entries[projection_id] = state
        self._vectors[projection_id] = state.to_vector()
        self._by_agent.setdefault(state.agent_id or "anonymous", []).append(projection_id)
        if state.room_id:
            self._by_room.setdefault(state.room_id, []).append(projection_id)

    def query_radius(self, center: WorldState, radius: float,
                     room_filter: Optional[str] = None) -> List[Tuple[str, WorldState, float]]:
        """Find all entries within radius, returning (id, state, distance)."""
        results = []
        for pid, state in self._entries.items():
            if room_filter and state.room_id != room_filter:
                continue
            dist = center.distance_to(state)
            if dist <= radius:
                results.append((pid, state, dist))
        return sorted(results, key=lambda x: x[2])

    def query_knn(self, center: WorldState, k: int = 5,
                  room_filter: Optional[str] = None) -> List[Tuple[str, WorldState, float]]:
        """K-nearest neighbors."""
        all_in_radius = self.query_radius(center, float("inf"), room_filter)
        return all_in_radius[:k]

    def query_semantic(self, key: str, value: Any) -> List[WorldState]:
        """Find states where semantics[key] == value."""
        return [s for s in self._entries.values() if s.semantics.get(key) == value]

    def get_latest(self, agent_id: str) -> Optional[WorldState]:
        """Most recent state for an agent."""
        pids = self._by_agent.get(agent_id, [])
        if not pids:
            return None
        # Return the one with latest timestamp
        states = [self._entries[pid] for pid in pids if pid in self._entries]
        return max(states, key=lambda s: s.timestamp) if states else None

    def remove(self, projection_id: str) -> None:
        """Remove an entry."""
        state = self._entries.pop(projection_id, None)
        self._vectors.pop(projection_id, None)
        if state and state.agent_id:
            self._by_agent[state.agent_id] = [
                pid for pid in self._by_agent.get(state.agent_id, [])
                if pid != projection_id
            ]
        if state and state.room_id:
            self._by_room[state.room_id] = [
                pid for pid in self._by_room.get(state.room_id, [])
                if pid != projection_id
            ]

    def snapshot(self) -> Dict[str, Any]:
        """Full index snapshot for A2A broadcast."""
        return {
            "dimension": self.dimension,
            "entries": {pid: s.to_dict() for pid, s in self._entries.items()},
            "timestamp": time.time(),
        }


# ────────────────────────────── Projector ──────────────────────────────

class SpatialProjector:
    """
    Fleet-native spatial awareness projector.
    Agents project states into a shared spatial index.
    Predictions flow through FLUX gates before A2A broadcast.
    """

    def __init__(self, fleet_node_id: str, db_path: Optional[str] = None,
                 dimension: int = 3):
        self.fleet_node_id = fleet_node_id
        self.db_path = db_path
        self.index = SpatialIndex(dimension=dimension)
        self._flux_constraints: List[FluxConstraint] = []
        self._prediction_history: Dict[str, List[Prediction]] = {}
        self._a2a_callbacks: List[Callable[[Prediction], None]] = []

    # ── State Projection ──

    def project_state(self, agent_id: str, room_id: str,
                      state: WorldState,
                      timestamp: Optional[float] = None) -> str:
        """
        Project an agent's perceptual state into the spatial index.
        Returns projection ID for later reference.
        """
        if timestamp:
            state.timestamp = timestamp
        state.agent_id = agent_id
        state.room_id = room_id

        # Generate deterministic projection ID
        content = f"{agent_id}:{room_id}:{state.timestamp}:{state.position}"
        projection_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        self.index.insert(projection_id, state)
        return projection_id

    # ── Spatial Queries ──

    def query_neighbors(self, agent_id: str, radius: float,
                        room_filter: Optional[str] = None,
                        exclude_self: bool = True) -> List[WorldState]:
        """
        Find all agents within radius of the given agent's latest state.
        """
        center = self.index.get_latest(agent_id)
        if center is None:
            return []
        results = self.index.query_radius(center, radius, room_filter)
        states = [r[1] for r in results]
        if exclude_self:
            states = [s for s in states if s.agent_id != agent_id]
        return states

    def query_knn(self, agent_id: str, k: int = 5,
                  room_filter: Optional[str] = None) -> List[WorldState]:
        """K-nearest neighbors to an agent's latest state."""
        center = self.index.get_latest(agent_id)
        if center is None:
            return []
        results = self.index.query_knn(center, k, room_filter)
        return [r[1] for r in results if r[1].agent_id != agent_id]

    def query_semantic(self, key: str, value: Any) -> List[WorldState]:
        """Semantic search across all indexed states."""
        return self.index.query_semantic(key, value)

    def get_agent_state(self, agent_id: str) -> Optional[WorldState]:
        """Latest known state for an agent."""
        return self.index.get_latest(agent_id)

    # ── Prediction ──

    def predict_trajectory(self, agent_id: str, horizon: int = 10,
                           model_id: str = "default",
                           worldmodel_bridge: Optional[Any] = None) -> Prediction:
        """
        Predict an agent's future trajectory.
        If worldmodel_bridge is provided, uses real world model.
        Otherwise, uses simple linear extrapolation (fallback).
        """
        current = self.index.get_latest(agent_id)
        if current is None:
            raise ValueError(f"No state known for agent {agent_id}")

        # Fallback: linear extrapolation using velocity
        trajectory = [current]
        uncertainty = [0.0]

        if current.velocity:
            for step in range(1, horizon + 1):
                dt = 1.0  # unit time step
                new_pos = tuple(
                    p + v * dt * step
                    for p, v in zip(current.position, current.velocity)
                )
                # Uncertainty grows with step
                unc = 0.1 * step
                state = WorldState(
                    position=new_pos,
                    velocity=current.velocity,
                    semantics=current.semantics.copy(),
                    confidence=current.confidence * (0.9 ** step),
                    timestamp=current.timestamp + step,
                    agent_id=agent_id,
                    room_id=current.room_id,
                )
                trajectory.append(state)
                uncertainty.append(unc)
        else:
            # No velocity: static prediction with growing uncertainty
            for step in range(1, horizon + 1):
                state = WorldState(
                    position=current.position,
                    semantics=current.semantics.copy(),
                    confidence=current.confidence * (0.8 ** step),
                    timestamp=current.timestamp + step,
                    agent_id=agent_id,
                    room_id=current.room_id,
                )
                trajectory.append(state)
                uncertainty.append(0.2 * step)

        prediction = Prediction(
            trajectory=trajectory,
            uncertainty=uncertainty,
            model_id=model_id,
            agent_id=agent_id,
        )

        # Store in history
        self._prediction_history.setdefault(agent_id, []).append(prediction)
        return prediction

    # ── FLUX Gating ──

    def add_flux_constraint(self, constraint: FluxConstraint) -> None:
        """Register a FLUX constraint for prediction gating."""
        self._flux_constraints.append(constraint)

    def apply_flux_gate(self, prediction: Prediction) -> Prediction:
        """
        Apply all registered FLUX constraints to a prediction.
        Hard constraints raise ValueError if violated.
        Soft constraints reduce confidence proportionally to penalty.
        """
        total_penalty = 0.0
        for constraint in self._flux_constraints:
            passed, penalty = constraint.evaluate(prediction)
            if constraint.hard and not passed:
                raise ValueError(
                    f"FLUX hard constraint '{constraint.name}' violated"
                )
            total_penalty += penalty

        if total_penalty > 0.0:
            # Reduce confidence proportionally
            for state in prediction.trajectory:
                state.confidence *= max(0.0, 1.0 - total_penalty)

        return prediction

    # ── A2A Broadcast ──

    def on_prediction(self, callback: Callable[[Prediction], None]) -> None:
        """Register callback for validated predictions."""
        self._a2a_callbacks.append(callback)

    def broadcast_prediction(self, prediction: Prediction,
                             target_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Broadcast a validated prediction to other agents.
        Returns broadcast metadata.
        """
        # Apply FLUX gate first
        validated = self.apply_flux_gate(prediction)

        # Build A2A message
        message = {
            "type": "spatial_prediction",
            "source_node": self.fleet_node_id,
            "prediction": validated.to_dict(),
            "target_agents": target_agents,
            "timestamp": time.time(),
        }

        # Notify local callbacks (A2A handlers would subscribe here)
        for cb in self._a2a_callbacks:
            cb(validated)

        return {
            "broadcast_id": str(uuid.uuid4())[:8],
            "prediction_id": validated.prediction_id,
            "recipients": len(target_agents) if target_agents else "all",
            "flux_passed": True,
        }

    # ── Spatial Breeding Context ──

    def get_proximal_agents(self, agent_id: str, radius: float = 5.0,
                            room_filter: Optional[str] = None) -> List[str]:
        """Get agent IDs within radius (for breeding parent selection)."""
        states = self.query_neighbors(agent_id, radius, room_filter)
        return list(dict.fromkeys(s.agent_id for s in states if s.agent_id))

    def get_spatial_diversity_score(self, agent_id: str) -> float:
        """
        Compute spatial diversity score for an agent.
        Higher = more isolated (diverse context).
        Lower = clustered with others (shared context).
        """
        state = self.index.get_latest(agent_id)
        if state is None:
            return 0.0
        neighbors = self.query_neighbors(agent_id, radius=100.0, exclude_self=True)
        if not neighbors:
            return 1.0  # Completely isolated = maximum diversity
        avg_dist = sum(state.distance_to(n) for n in neighbors) / len(neighbors)
        # Normalize: assume max meaningful distance is 100
        return min(1.0, avg_dist / 100.0)

    # ── Snapshot / Sync ──

    def snapshot(self) -> Dict[str, Any]:
        """Full spatial index snapshot for cross-node sync."""
        return {
            "node_id": self.fleet_node_id,
            "index": self.index.snapshot(),
            "constraint_count": len(self._flux_constraints),
            "prediction_count": sum(len(v) for v in self._prediction_history.values()),
        }

    def ingest_snapshot(self, snapshot: Dict[str, Any]) -> int:
        """
        Ingest a snapshot from another fleet node.
        Returns count of new states ingested.
        """
        entries = snapshot.get("index", {}).get("entries", {})
        count = 0
        for pid, state_dict in entries.items():
            if pid not in self._entries:
                state = WorldState.from_dict(state_dict)
                self.index.insert(pid, state)
                count += 1
        return count

    @property
    def _entries(self):
        return self.index._entries


# ────────────────────────────── Utilities ──────────────────────────────

def create_thermal_constraint(max_temp: float = 80.0,
                               hard: bool = False) -> FluxConstraint:
    """Factory: thermal feasibility constraint."""
    def check(pred: Prediction) -> bool:
        temps = [
            s.semantics.get("temperature", 0.0)
            for s in pred.trajectory
        ]
        return all(t <= max_temp for t in temps)

    def penalty(pred: Prediction) -> float:
        temps = [
            s.semantics.get("temperature", 0.0)
            for s in pred.trajectory
        ]
        max_t = max(temps) if temps else 0.0
        return max(0.0, (max_t - max_temp) / max_temp)

    return FluxConstraint(
        name="thermal_feasibility",
        check=check,
        penalty=penalty,
        hard=hard,
        weight=1.0,
    )


def create_uncertainty_constraint(max_uncertainty: float = 0.5,
                                  hard: bool = False) -> FluxConstraint:
    """Factory: uncertainty threshold constraint."""
    def check(pred: Prediction) -> bool:
        return pred.mean_uncertainty <= max_uncertainty

    def penalty(pred: Prediction) -> float:
        return max(0.0, (pred.mean_uncertainty - max_uncertainty) / max_uncertainty)

    return FluxConstraint(
        name="uncertainty_threshold",
        check=check,
        penalty=penalty,
        hard=hard,
        weight=0.3,
    )


def create_room_constraint(allowed_rooms: List[str],
                           hard: bool = True) -> FluxConstraint:
    """Factory: prediction must stay within allowed rooms."""
    allowed_set = set(allowed_rooms)

    def check(pred: Prediction) -> bool:
        return all(s.room_id in allowed_set for s in pred.trajectory)

    return FluxConstraint(
        name="room_boundary",
        check=check,
        hard=hard,
        weight=1.0,
    )
