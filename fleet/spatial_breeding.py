"""
Spatial Breeding Integration

Connects the A2A Spatial Projector to the Breeding Daemon,
enabling location-aware parent selection and spatial diversity
maintenance in the evolutionary loop.

Usage:
    from fleet.spatial_breeding import SpatialBreedingContext
    from fleet.spatial_projector import SpatialProjector

    projector = SpatialProjector("node-alpha")
    context = SpatialBreedingContext(projector)

    # Select parents that are spatially proximal (shared context)
    parents = context.select_proximal_parents("agent-1", radius=5.0, k=3)

    # Or select parents that are distant (diversity injection)
    parents = context.select_diverse_parents("agent-1", min_distance=20.0, k=2)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from fleet.spatial_projector import SpatialProjector, WorldState


@dataclass
class SpatialParentCandidate:
    """A candidate parent with spatial metadata."""

    agent_id: str
    genome: Any
    fitness: float
    position: Tuple[float, ...]
    distance: float
    room_id: Optional[str] = None
    spatial_diversity_score: float = 0.0


@dataclass
class SpatialBreedingConfig:
    """Configuration for spatial breeding strategies."""

    # Proximity selection
    proximal_radius: float = 10.0
    proximal_k: int = 3

    # Diversity selection
    min_diversity_distance: float = 20.0
    diversity_k: int = 2

    # Hybrid mixing
    proximity_ratio: float = 0.6  # 60% proximal, 40% diverse

    # Room affinity
    same_room_bonus: float = 1.2
    cross_room_penalty: float = 0.8

    # Adaptive radius
    adaptive_radius: bool = True
    min_radius: float = 5.0
    max_radius: float = 50.0

    # Trajectory-based selection
    trajectory_horizon: int = 5
    collision_penalty: float = -1.0


class SpatialBreedingContext:
    """
    Provides spatial context for breeding decisions.
    Wraps a SpatialProjector to enable location-aware parent selection.
    """

    def __init__(
        self,
        projector: SpatialProjector,
        config: Optional[SpatialBreedingConfig] = None,
    ):
        self.projector = projector
        self.config = config or SpatialBreedingConfig()

    # ── Core Selection Strategies ──

    def select_proximal_parents(
        self,
        agent_id: str,
        radius: Optional[float] = None,
        k: Optional[int] = None,
        genome_fn: Optional[Callable[[str], Any]] = None,
        fitness_fn: Optional[Callable[[str], float]] = None,
    ) -> List[SpatialParentCandidate]:
        """
        Select parents spatially proximal to the given agent.
        These parents share context (same room, nearby positions).
        """
        radius = radius or self.config.proximal_radius
        k = k or self.config.proximal_k

        if self.config.adaptive_radius:
            radius = self._adapt_radius(agent_id, radius)

        neighbor_states = self.projector.query_neighbors(agent_id, radius=radius)

        candidates = []
        for state in neighbor_states:
            if state.agent_id is None:
                continue
            dist = self._distance(agent_id, state.agent_id)
            genome = genome_fn(state.agent_id) if genome_fn else None
            fitness = fitness_fn(state.agent_id) if fitness_fn else 0.0

            candidates.append(
                SpatialParentCandidate(
                    agent_id=state.agent_id,
                    genome=genome,
                    fitness=fitness,
                    position=state.position,
                    distance=dist,
                    room_id=state.room_id,
                )
            )

        # Sort by distance, take top k
        candidates.sort(key=lambda c: c.distance)
        return candidates[:k]

    def select_diverse_parents(
        self,
        agent_id: str,
        min_distance: Optional[float] = None,
        k: Optional[int] = None,
        genome_fn: Optional[Callable[[str], Any]] = None,
        fitness_fn: Optional[Callable[[str], float]] = None,
    ) -> List[SpatialParentCandidate]:
        """
        Select parents that are spatially distant from the given agent.
        These parents inject diversity into the population.
        """
        min_distance = min_distance or self.config.min_diversity_distance
        k = k or self.config.diversity_k

        all_agents = self._get_all_agent_ids(exclude=agent_id)
        candidates = []

        for other_id in all_agents:
            dist = self._distance(agent_id, other_id)
            if dist >= min_distance:
                other_state = self.projector.get_agent_state(other_id)
                genome = genome_fn(other_id) if genome_fn else None
                fitness = fitness_fn(other_id) if fitness_fn else 0.0

                candidates.append(
                    SpatialParentCandidate(
                        agent_id=other_id,
                        genome=genome,
                        fitness=fitness,
                        position=other_state.position if other_state else (0.0,),
                        distance=dist,
                        room_id=other_state.room_id if other_state else None,
                        spatial_diversity_score=self.projector.get_spatial_diversity_score(
                            other_id
                        ),
                    )
                )

        # Sort by distance (farthest first), take top k
        candidates.sort(key=lambda c: c.distance, reverse=True)
        return candidates[:k]

    def select_hybrid_parents(
        self,
        agent_id: str,
        total_k: int = 5,
        genome_fn: Optional[Callable[[str], Any]] = None,
        fitness_fn: Optional[Callable[[str], float]] = None,
    ) -> List[SpatialParentCandidate]:
        """
        Hybrid selection: mix of proximal and diverse parents.
        Ratio controlled by config.proximity_ratio.
        """
        n_proximal = max(1, int(total_k * self.config.proximity_ratio))
        n_diverse = max(1, total_k - n_proximal)

        proximal = self.select_proximal_parents(
            agent_id, k=n_proximal, genome_fn=genome_fn, fitness_fn=fitness_fn
        )
        diverse = self.select_diverse_parents(
            agent_id, k=n_diverse, genome_fn=genome_fn, fitness_fn=fitness_fn
        )

        # Combine and deduplicate
        seen = set(p.agent_id for p in proximal)
        result = list(proximal)
        for c in diverse:
            if c.agent_id not in seen:
                result.append(c)
                seen.add(c.agent_id)

        return result[:total_k]

    def select_room_affinity_parents(
        self,
        agent_id: str,
        room_id: str,
        k: int = 3,
        genome_fn: Optional[Callable[[str], Any]] = None,
        fitness_fn: Optional[Callable[[str], float]] = None,
    ) -> List[SpatialParentCandidate]:
        """
        Select parents with affinity for a specific room.
        Agents in the same room get a bonus.
        """
        all_agents = self._get_all_agent_ids(exclude=agent_id)
        candidates = []

        for other_id in all_agents:
            other_state = self.projector.get_agent_state(other_id)
            if other_state is None:
                continue

            dist = self._distance(agent_id, other_id)
            genome = genome_fn(other_id) if genome_fn else None
            fitness = fitness_fn(other_id) if fitness_fn else 0.0

            # Room affinity scoring
            affinity = 1.0
            if other_state.room_id == room_id:
                affinity = self.config.same_room_bonus
            else:
                affinity = self.config.cross_room_penalty

            candidates.append(
                SpatialParentCandidate(
                    agent_id=other_id,
                    genome=genome,
                    fitness=fitness * affinity,
                    position=other_state.position,
                    distance=dist,
                    room_id=other_state.room_id,
                )
            )

        # Sort by fitness (highest first), take top k
        candidates.sort(key=lambda c: c.fitness, reverse=True)
        return candidates[:k]

    def select_trajectory_compatible_parents(
        self,
        agent_id: str,
        k: int = 3,
        genome_fn: Optional[Callable[[str], Any]] = None,
        fitness_fn: Optional[Callable[[str], float]] = None,
    ) -> List[SpatialParentCandidate]:
        """
        Select parents whose predicted trajectories don't collide
        with the given agent's predicted trajectory.
        """
        try:
            agent_pred = self.projector.predict_trajectory(
                agent_id, horizon=self.config.trajectory_horizon
            )
        except ValueError:
            return []

        all_agents = self._get_all_agent_ids(exclude=agent_id)
        candidates = []

        for other_id in all_agents:
            try:
                other_pred = self.projector.predict_trajectory(
                    other_id, horizon=self.config.trajectory_horizon
                )
            except ValueError:
                continue

            # Check for trajectory collision (simplified)
            collision_score = self._trajectory_collision_score(
                agent_pred.trajectory, other_pred.trajectory
            )

            if collision_score > 0:  # No collision
                other_state = self.projector.get_agent_state(other_id)
                genome = genome_fn(other_id) if genome_fn else None
                fitness = fitness_fn(other_id) if fitness_fn else 0.0

                candidates.append(
                    SpatialParentCandidate(
                        agent_id=other_id,
                        genome=genome,
                        fitness=fitness + collision_score,
                        position=other_state.position if other_state else (0.0,),
                        distance=self._distance(agent_id, other_id),
                        room_id=other_state.room_id if other_state else None,
                    )
                )

        candidates.sort(key=lambda c: c.fitness, reverse=True)
        return candidates[:k]

    # ── Breeding Daemon Integration ──

    def to_breeder_parents(
        self,
        candidates: List[SpatialParentCandidate],
    ) -> List[Tuple[Any, Any]]:
        """
        Convert spatial candidates to (genome, fitness) tuples
        suitable for BreederDaemon parent selection.
        """
        return [(c.genome, c.fitness) for c in candidates if c.genome is not None]

    def spatial_fitness_adjustment(self, agent_id: str, base_fitness: float) -> float:
        """
        Adjust fitness based on spatial diversity.
        Isolated agents (high diversity) get a bonus.
        Clustered agents (low diversity) get a penalty.
        """
        diversity = self.projector.get_spatial_diversity_score(agent_id)
        # Diversity bonus: up to +20% for isolated agents
        return base_fitness * (1.0 + 0.2 * diversity)

    # ── Spatial Metrics ──

    def population_spatial_entropy(self) -> float:
        """
        Compute spatial entropy of the population.
               High entropy = agents are well-distributed.
               Low entropy = agents are clustered.
        """
        agent_ids = self._get_all_agent_ids()
        if len(agent_ids) < 2:
            return 0.0

        # Compute average pairwise distance
        total_dist = 0.0
        count = 0
        for i, a1 in enumerate(agent_ids):
            for a2 in agent_ids[i + 1 :]:
                dist = self._distance(a1, a2)
                total_dist += dist
                count += 1

        avg_dist = total_dist / count if count > 0 else 0.0
        # Normalize by max expected distance (assume 100)
        return min(1.0, avg_dist / 100.0)

    def room_distribution(self) -> Dict[str, int]:
        """Count agents per room."""
        counts: Dict[str, int] = {}
        for agent_id in self._get_all_agent_ids():
            state = self.projector.get_agent_state(agent_id)
            if state and state.room_id:
                counts[state.room_id] = counts.get(state.room_id, 0) + 1
        return counts

    def detect_clusters(self, radius: float = 5.0) -> List[Set[str]]:
        """
        Detect spatial clusters of agents.
        Returns list of agent ID sets, each a cluster.
        """
        agent_ids = self._get_all_agent_ids()
        visited: Set[str] = set()
        clusters: List[Set[str]] = []

        for agent_id in agent_ids:
            if agent_id in visited:
                continue

            # BFS to find cluster
            cluster: Set[str] = {agent_id}
            queue = [agent_id]
            visited.add(agent_id)

            while queue:
                current = queue.pop(0)
                neighbors = self.projector.get_proximal_agents(current, radius=radius)
                for neighbor_id in neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        cluster.add(neighbor_id)
                        queue.append(neighbor_id)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def recommend_relocation(
        self, agent_id: str, target_entropy: float = 0.7
    ) -> Optional[WorldState]:
        """
        Recommend a new spatial position for an agent to improve
        population spatial entropy.
        """
        current_entropy = self.population_spatial_entropy()
        if current_entropy >= target_entropy:
            return None  # Already good enough

        # Find the largest empty region
        all_states = [
            self.projector.get_agent_state(aid)
            for aid in self._get_all_agent_ids()
            if self.projector.get_agent_state(aid) is not None
        ]

        if not all_states:
            return None

        # Simple heuristic: move to opposite quadrant
        current = self.projector.get_agent_state(agent_id)
        if current is None:
            return None

        # Compute centroid
        dim = len(current.position)
        centroid = tuple(
            sum(s.position[i] for s in all_states) / len(all_states) for i in range(dim)
        )

        # Recommend position opposite to centroid
        new_pos = tuple(2 * current.position[i] - centroid[i] for i in range(dim))

        return WorldState(
            position=new_pos,
            semantics=current.semantics.copy(),
            confidence=current.confidence,
            agent_id=agent_id,
            room_id=current.room_id,
        )

    # ── Helpers ──

    def _get_all_agent_ids(self, exclude: Optional[str] = None) -> List[str]:
        """Get all known agent IDs from the projector."""
        # Access the index's by_agent mapping
        ids = list(self.projector.index._by_agent.keys())
        if exclude:
            ids = [aid for aid in ids if aid != exclude]
        return ids

    def _distance(self, agent_id1: str, agent_id2: str) -> float:
        """Distance between two agents."""
        s1 = self.projector.get_agent_state(agent_id1)
        s2 = self.projector.get_agent_state(agent_id2)
        if s1 is None or s2 is None:
            return float("inf")
        return s1.distance_to(s2)

    def _adapt_radius(self, agent_id: str, base_radius: float) -> float:
        """Adapt radius based on local density."""
        # Count neighbors at base radius
        neighbors = self.projector.query_neighbors(agent_id, radius=base_radius)
        density = len(neighbors)

        if density < 2:
            # Sparse: expand radius
            return min(self.config.max_radius, base_radius * 2.0)
        elif density > 10:
            # Dense: shrink radius
            return max(self.config.min_radius, base_radius * 0.5)
        return base_radius

    def _trajectory_collision_score(
        self, traj1: List[WorldState], traj2: List[WorldState]
    ) -> float:
        """
        Simple trajectory collision detection.
        Returns positive score if no collision, negative if collision.
        """
        min_len = min(len(traj1), len(traj2))
        for i in range(min_len):
            dist = traj1[i].distance_to(traj2[i])
            if dist < 1.0:  # Collision threshold
                return self.config.collision_penalty
        return 1.0  # No collision
