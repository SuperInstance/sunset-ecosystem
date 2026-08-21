"""LineageSanityChecker — tamper-detection for agent parentage chains.

Verifies that agent genealogy hasn't been corrupted by adversarial breeding.
Checks for orphans, impossible vector jumps, generation mismatches, and cycles.

Reference: docs/RESEARCH_SECURITY.md — Experiment: LineageSanityChecker
"""

from __future__ import annotations

__all__ = ["LineageSanityChecker", "Agent"]

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """Minimal agent record for lineage verification.

    Attributes:
        id: Unique agent identifier.
        vector: Latent vector (list or ndarray of float).
        generation: Breeding generation (0 = root / seed).
        parent_a: First parent ID, or None for root agents.
        parent_b: Second parent ID, or None for asexual / single-parent.
    """

    id: int
    vector: Any
    generation: int
    parent_a: int | None = None
    parent_b: int | None = None


class LineageSanityChecker:
    """Verifies agent parentage chains for tampering.

    Args:
        max_depth: How many generations back to trace in ``build_lineage_tree``.
    """

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth

    # ── public API ──────────────────────────────────────────

    def verify_lineage(
        self, agent_id: int, population: list[Agent]
    ) -> tuple[bool, str]:
        """Verify an agent's parentage chain.

        Returns:
            (is_valid, reason_or_ok)

        Checks performed:
        1. Parents exist in population history.
        2. Parent-child vector distance is plausible (not impossible jump).
        3. No cycles in lineage graph.
        4. Birth generation makes sense (child.gen == max(parent.gen) + 1).
        """
        agent = self._find_agent(agent_id, population)
        if agent is None:
            return (False, f"Agent {agent_id} not found in population")

        # Check 1: Parents exist
        if agent.parent_a is not None:
            pa = self._find_agent(agent.parent_a, population)
            if pa is None:
                return (False, f"Orphan: parent_a {agent.parent_a} missing")
        if agent.parent_b is not None:
            pb = self._find_agent(agent.parent_b, population)
            if pb is None:
                return (False, f"Orphan: parent_b {agent.parent_b} missing")

        # Check 4: Generation consistency
        if agent.parent_a is not None or agent.parent_b is not None:
            parent_gens = []
            if agent.parent_a is not None:
                pa = self._find_agent(agent.parent_a, population)
                if pa:
                    parent_gens.append(pa.generation)
            if agent.parent_b is not None:
                pb = self._find_agent(agent.parent_b, population)
                if pb:
                    parent_gens.append(pb.generation)
            if parent_gens:
                expected_gen = max(parent_gens) + 1
                if agent.generation != expected_gen:
                    return (
                        False,
                        f"Generation mismatch: expected {expected_gen}, "
                        f"got {agent.generation}",
                    )
        else:
            # Root agent — generation should be 0
            if agent.generation != 0:
                return (
                    False,
                    f"Root agent {agent_id} has non-zero generation {agent.generation}",
                )

        # Check 3: Cycle detection
        if self._has_cycle(agent_id, population):
            return (False, f"Cycle detected in lineage of agent {agent_id}")

        # Check 2: Plausible vector jump
        jump_ok, jump_reason = self._check_vector_plausibility(agent, population)
        if not jump_ok:
            return (False, jump_reason)

        return (True, "Lineage valid")

    def detect_orphans(self, population: list[Agent]) -> list[int]:
        """Return agent IDs with missing or corrupted parent records."""
        orphans: list[int] = []
        for agent in population:
            if agent.parent_a is not None:
                if self._find_agent(agent.parent_a, population) is None:
                    orphans.append(agent.id)
                    continue
            if agent.parent_b is not None:
                if self._find_agent(agent.parent_b, population) is None:
                    orphans.append(agent.id)
                    continue
        return orphans

    def detect_impossible_jumps(
        self, population: list[Agent], threshold: float = 5.0
    ) -> list[int]:
        """Agents whose vector distance from nearest parent exceeds
        ``threshold × typical_jump``.

        Returns:
            List of flagged agent IDs.
        """
        # Compute typical jump = median parent-child distance across population
        jumps: list[float] = []
        for agent in population:
            if agent.parent_a is not None or agent.parent_b is not None:
                dist = self._nearest_parent_distance(agent, population)
                if dist is not None:
                    jumps.append(dist)

        if not jumps:
            return []

        typical_jump = float(np.median(jumps))
        if typical_jump == 0:
            # All identical — any non-zero distance is suspicious
            typical_jump = 1e-6

        flagged: list[int] = []
        for agent in population:
            if agent.parent_a is None and agent.parent_b is None:
                continue  # root agents have no parent distance
            dist = self._nearest_parent_distance(agent, population)
            if dist is not None and dist > threshold * typical_jump:
                flagged.append(agent.id)

        return flagged

    def build_lineage_tree(self, agent_id: int, population: list[Agent]) -> dict:
        """Build a nested dict of ancestors up to ``max_depth``.

        Returns:
            A dict with keys: ``agent_id``, ``generation``, ``parents``
            (list of recursively built parent trees).
        """
        return self._build_tree_recursive(agent_id, population, depth=0, visited=set())

    # ── helpers ─────────────────────────────────────────────

    def _find_agent(self, agent_id: int, population: list[Agent]) -> Agent | None:
        for a in population:
            if a.id == agent_id:
                return a
        return None

    def _has_cycle(self, agent_id: int, population: list[Agent]) -> bool:
        """DFS up the parent chain looking for a cycle back to agent_id."""
        visited: set[int] = set()
        frontier: list[int] = []

        # Start with this agent's parents
        agent = self._find_agent(agent_id, population)
        if agent is None:
            return False
        if agent.parent_a is not None:
            frontier.append(agent.parent_a)
        if agent.parent_b is not None:
            frontier.append(agent.parent_b)

        while frontier:
            current = frontier.pop()
            if current == agent_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            parent = self._find_agent(current, population)
            if parent is not None:
                if parent.parent_a is not None:
                    frontier.append(parent.parent_a)
                if parent.parent_b is not None:
                    frontier.append(parent.parent_b)
        return False

    def _check_vector_plausibility(
        self, agent: Agent, population: list[Agent]
    ) -> tuple[bool, str]:
        """Check that the agent's vector is a plausible distance from parents."""
        if agent.parent_a is None and agent.parent_b is None:
            return (True, "Root agent — no parent distance check")

        dist = self._nearest_parent_distance(agent, population)
        if dist is None:
            return (True, "No parent vectors available for comparison")

        # Compute typical jump across the whole population
        jumps: list[float] = []
        for a in population:
            if a.id == agent.id:
                continue
            if a.parent_a is not None or a.parent_b is not None:
                d = self._nearest_parent_distance(a, population)
                if d is not None:
                    jumps.append(d)

        if not jumps:
            return (True, "No reference jumps in population")

        typical = float(np.median(jumps))
        if typical == 0:
            typical = 1e-6

        # Use a conservative threshold (3× typical) for the plausibility gate
        if dist > 3.0 * typical:
            return (
                False,
                f"Impossible jump: distance={dist:.4f} > 3×typical={3 * typical:.4f}",
            )

        return (True, f"Jump distance {dist:.4f} within typical range")

    def _nearest_parent_distance(
        self, agent: Agent, population: list[Agent]
    ) -> float | None:
        """Cosine distance from agent to its nearest existing parent."""
        vec = self._to_numpy(agent.vector)
        if vec is None:
            return None

        distances: list[float] = []
        for parent_id in (agent.parent_a, agent.parent_b):
            if parent_id is None:
                continue
            parent = self._find_agent(parent_id, population)
            if parent is None:
                continue
            pvec = self._to_numpy(parent.vector)
            if pvec is None:
                continue
            distances.append(self._cosine_distance(vec, pvec))

        return min(distances) if distances else None

    def _build_tree_recursive(
        self,
        agent_id: int,
        population: list[Agent],
        depth: int,
        visited: set[int],
    ) -> dict:
        agent = self._find_agent(agent_id, population)
        if agent is None:
            return {"agent_id": agent_id, "generation": None, "parents": []}

        # Guard against cycles in the tree representation
        if agent_id in visited:
            return {
                "agent_id": agent_id,
                "generation": agent.generation,
                "parents": [{"cycle": True}],
            }

        visited = visited | {agent_id}
        parents: list[dict] = []

        if depth < self.max_depth:
            for parent_id in (agent.parent_a, agent.parent_b):
                if parent_id is not None:
                    parents.append(
                        self._build_tree_recursive(
                            parent_id, population, depth + 1, visited
                        )
                    )

        return {
            "agent_id": agent_id,
            "generation": agent.generation,
            "parents": parents,
        }

    @staticmethod
    def _to_numpy(vec: Any) -> np.ndarray | None:
        if vec is None:
            return None
        if isinstance(vec, np.ndarray):
            return vec.astype(np.float32)
        try:
            return np.asarray(vec, dtype=np.float32)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        an = float(np.linalg.norm(a))
        bn = float(np.linalg.norm(b))
        if an == 0 or bn == 0:
            return 1.0
        sim = float(np.dot(a, b) / (an * bn))
        sim = max(-1.0, min(1.0, sim))
        return 1.0 - sim
