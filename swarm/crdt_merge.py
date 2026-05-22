"""CRDT (Conflict-free Replicated Data Type) merge for divergent agent populations.

When network partitions occur, agent populations on different nodes breed
independently. When partitions heal, this engine merges the divergent
populations back into a single consistent state without coordination.

Reference: docs/RESEARCH_DISTRIBUTED.md — Experiment: CRDT Breeding Merge
"""

from __future__ import annotations

__all__ = [
    "Agent",
    "DivergenceReport",
    "CRDTMergeEngine",
    "LineageSanityError",
]

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from swarm.vector_table import FluxVectorTable, AgentVector

logger = logging.getLogger(__name__)


class LineageSanityError(ValueError):
    """Raised when a remote agent fails lineage sanity checks."""
    pass


@dataclass
class Agent:
    """Lightweight agent record for CRDT merge operations.

    Uses *agent_id* as an ``int`` to align with :class:`FluxVectorTable`.
    """

    agent_id: int
    fitness: float = 0.0
    generation: int = 0
    parent_a: int | None = None
    parent_b: int | None = None
    all_parents: list[int] = field(default_factory=list)
    vector: list[float] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    capability_mask: int = 0xFFFF

    def __post_init__(self) -> None:
        if not (0.0 <= self.fitness <= 1.0):
            raise ValueError(f"fitness must be in [0, 1], got {self.fitness}")
        if self.generation < 0:
            raise ValueError(f"generation must be >= 0, got {self.generation}")
        # Derive all_parents from parent_a/parent_b if not explicitly set
        if not self.all_parents and (self.parent_a is not None or self.parent_b is not None):
            parents: list[int] = []
            for p in (self.parent_a, self.parent_b):
                if p is not None and p not in parents:
                    parents.append(p)
            object.__setattr__(self, 'all_parents', parents)


@dataclass
class DivergenceReport:
    """Characterization of how two populations diverged."""

    local_only: list[int]  # agents only in local
    remote_only: list[int]  # agents only in remote
    common_diverged: list[int]  # agents in both but different
    lineage_conflicts: list[int]  # agents with conflicting parent records
    fitness_delta: float  # mean fitness difference for common agents


class CRDTMergeEngine:
    """Merge divergent agent populations after network partition.

    Args:
        vector_table: The local FluxVectorTable to sync merged embeddings into.
    """

    def __init__(self, vector_table: FluxVectorTable) -> None:
        self.vector_table = vector_table

    # ── public API ──────────────────────────────────────────

    def merge_populations(
        self, local: list[Agent], remote: list[Agent]
    ) -> list[Agent]:
        """Merge two divergent populations.

        Strategy:
        1. Union all agents (by unique ID)
        2. For agents existing in both: keep higher-fitness copy
        3. For new agents from remote: verify lineage sanity before accepting
        4. If both copies have valid but different lineages:
           create merged lineage with both parents noted
        5. Update vector table with merged embeddings
        """
        local_map = {a.agent_id: a for a in local}
        remote_map = {a.agent_id: a for a in remote}

        all_ids = set(local_map.keys()) | set(remote_map.keys())
        merged: dict[int, Agent] = {}
        rejected: list[int] = []

        for aid in all_ids:
            local_a = local_map.get(aid)
            remote_a = remote_map.get(aid)

            if local_a is not None and remote_a is not None:
                # Both sides have this agent — resolve conflict
                merged[aid] = self.resolve_conflict(local_a, remote_a)
            elif local_a is not None:
                # Local only
                merged[aid] = local_a
            else:
                # Remote only — sanity check before accepting
                assert remote_a is not None
                try:
                    self._verify_lineage_sanity(remote_a, local, remote)
                    merged[aid] = remote_a
                except LineageSanityError as exc:
                    logger.warning("Rejected remote agent %d: %s", aid, exc)
                    rejected.append(aid)

        # Sync vectors for every merged agent that has a vector
        for agent in merged.values():
            if agent.vector:
                self._upsert_vector(agent)

        logger.info(
            "Merged %d agents (%d rejected)", len(merged), len(rejected)
        )
        return list(merged.values())

    def detect_divergence(
        self, local: list[Agent], remote: list[Agent]
    ) -> DivergenceReport:
        """Characterize how two populations diverged."""
        local_ids = {a.agent_id for a in local}
        remote_ids = {a.agent_id for a in remote}

        local_only = sorted(local_ids - remote_ids)
        remote_only = sorted(remote_ids - local_ids)
        common = local_ids & remote_ids

        common_diverged: list[int] = []
        lineage_conflicts: list[int] = []
        fitness_diffs: list[float] = []

        local_map = {a.agent_id: a for a in local}
        remote_map = {a.agent_id: a for a in remote}

        for aid in common:
            la = local_map[aid]
            ra = remote_map[aid]

            if self._agents_differ(la, ra):
                common_diverged.append(aid)

            if self._lineage_conflicts(la, ra):
                lineage_conflicts.append(aid)

            fitness_diffs.append(abs(la.fitness - ra.fitness))

        fitness_delta = (
            sum(fitness_diffs) / len(fitness_diffs) if fitness_diffs else 0.0
        )

        return DivergenceReport(
            local_only=local_only,
            remote_only=remote_only,
            common_diverged=sorted(common_diverged),
            lineage_conflicts=sorted(lineage_conflicts),
            fitness_delta=fitness_delta,
        )

    def resolve_conflict(self, local_agent: Agent, remote_agent: Agent) -> Agent:
        """Pick or merge a single agent from conflicting copies.

        Rules:
        - Keep the copy with higher fitness.
        - If fitness is equal, prefer the one with the later *last_updated*.
        - If both have valid but different lineages, create a merged
          lineage with both parents noted.
        """
        if local_agent.fitness > remote_agent.fitness:
            winner = local_agent
        elif remote_agent.fitness > local_agent.fitness:
            winner = remote_agent
        else:
            # Tie-break on last_updated (LWW)
            winner = (
                local_agent
                if local_agent.last_updated >= remote_agent.last_updated
                else remote_agent
            )

        # If lineages differ and both are valid, merge them
        if self._lineage_conflicts(local_agent, remote_agent):
            merged_parents = self._merge_lineages(local_agent, remote_agent)
            # Return a new Agent with winner's body but merged lineage
            return Agent(
                agent_id=winner.agent_id,
                fitness=winner.fitness,
                generation=winner.generation,
                parent_a=merged_parents[0] if len(merged_parents) > 0 else None,
                parent_b=merged_parents[1] if len(merged_parents) > 1 else None,
                all_parents=merged_parents,
                vector=winner.vector,
                last_updated=max(local_agent.last_updated, remote_agent.last_updated),
                capability_mask=winner.capability_mask,
            )

        return winner

    def sync_vector_table(
        self, local_vt: FluxVectorTable, remote_vt: FluxVectorTable
    ) -> FluxVectorTable:
        """Merge vector tables using LWW (last-write-wins) on update timestamps.

        Returns a *new* FluxVectorTable containing the merged state.
        The returned table uses the same *dim* and *bit_width* as *local_vt*.
        """
        # Collect all agent IDs from both tables
        local_ids = set(local_vt._meta.keys())
        remote_ids = set(remote_vt._meta.keys())
        all_ids = local_ids | remote_ids

        merged = FluxVectorTable(dim=local_vt.dim, bit_width=local_vt.bit_width)

        for aid in all_ids:
            local_meta = local_vt._meta.get(aid)
            remote_meta = remote_vt._meta.get(aid)

            if local_meta is not None and remote_meta is not None:
                # LWW: compare timestamps stored in extra dict
                local_ts = float(local_meta.extra.get("last_updated", 0))
                remote_ts = float(remote_meta.extra.get("last_updated", 0))
                winner_vt = local_vt if local_ts >= remote_ts else remote_vt
            elif local_meta is not None:
                winner_vt = local_vt
            else:
                winner_vt = remote_vt

            # Reconstruct AgentVector from winner's internal storage
            vec = self._extract_vector(winner_vt, aid)
            meta = winner_vt._meta[aid]
            if vec is not None:
                merged.add(
                    AgentVector(
                        agent_id=aid,
                        vector=vec,
                        fitness=meta.fitness,
                        generation=meta.generation,
                        capability_mask=meta.capability_mask,
                        thermal_pressure=meta.thermal_pressure,
                    )
                )
                # Preserve timestamp in extra
                ts = float(meta.extra.get("last_updated", 0))
                merged._meta[aid].extra["last_updated"] = ts

        return merged

    # ── lineage sanity ──────────────────────────────────────

    def _verify_lineage_sanity(
        self,
        agent: Agent,
        local_pop: list[Agent],
        remote_pop: list[Agent],
    ) -> None:
        """Check that a remote agent's lineage is plausible.

        Raises LineageSanityError if anything looks impossible.
        """
        all_ids = {a.agent_id for a in local_pop} | {a.agent_id for a in remote_pop}

        # Fitness must be in valid range (already checked in __post_init__)
        # but double-check for impossible jumps relative to parents
        if agent.fitness > 1.0 or agent.fitness < 0.0:
            raise LineageSanityError(f"fitness {agent.fitness} out of [0, 1]")

        # Generation jump check: an agent's generation should not exceed
        # max(parent generations) + 1 by more than a reasonable margin.
        if agent.parent_a is not None or agent.parent_b is not None:
            parent_gens: list[int] = []
            for pop in (local_pop, remote_pop):
                for a in pop:
                    if a.agent_id in (agent.parent_a, agent.parent_b):
                        parent_gens.append(a.generation)

            if parent_gens:
                max_parent_gen = max(parent_gens)
                if agent.generation > max_parent_gen + 3:
                    raise LineageSanityError(
                        f"generation {agent.generation} too far from parents "
                        f"(max parent gen {max_parent_gen})"
                    )
            else:
                # Parents claimed but do not exist in either population
                raise LineageSanityError(
                    f"parents {agent.parent_a}, {agent.parent_b} not found in "
                    f"either population"
                )

        # If no parents, generation must be 0 (seed agent)
        if agent.parent_a is None and agent.parent_b is None and agent.generation != 0:
            raise LineageSanityError(
                f"seed agent (no parents) has non-zero generation {agent.generation}"
            )

    def _merge_lineages(
        self, local_agent: Agent, remote_agent: Agent
    ) -> list[int]:
        """Combine parent records from both copies into a single list.

        Preserves order: local parents first, then remote parents that
        are not already present.
        """
        parents: list[int] = []
        for p in (local_agent.parent_a, local_agent.parent_b):
            if p is not None and p not in parents:
                parents.append(p)
        for p in (remote_agent.parent_a, remote_agent.parent_b):
            if p is not None and p not in parents:
                parents.append(p)
        return parents

    # ── helpers ───────────────────────────────────────────

    def _agents_differ(self, a: Agent, b: Agent) -> bool:
        """True if two agents have meaningfully different state."""
        return (
            a.fitness != b.fitness
            or a.generation != b.generation
            or a.parent_a != b.parent_a
            or a.parent_b != b.parent_b
            or a.vector != b.vector
        )

    def _lineage_conflicts(self, a: Agent, b: Agent) -> bool:
        """True if both agents have valid but different parent records."""
        a_has_parents = a.parent_a is not None or a.parent_b is not None
        b_has_parents = b.parent_a is not None or b.parent_b is not None
        if not a_has_parents or not b_has_parents:
            return False
        return (a.parent_a, a.parent_b) != (b.parent_a, b.parent_b)

    def _upsert_vector(self, agent: Agent) -> None:
        """Add or overwrite an agent's vector in the local table."""
        if not agent.vector:
            return
        av = AgentVector(
            agent_id=agent.agent_id,
            vector=agent.vector,
            fitness=agent.fitness,
            generation=agent.generation,
            capability_mask=agent.capability_mask,
            thermal_pressure=0.0,
        )
        self.vector_table.add(av)
        # Store last_updated for future LWW merges
        self.vector_table._meta[agent.agent_id].extra["last_updated"] = (
            agent.last_updated
        )

    def _extract_vector(self, vt: FluxVectorTable, agent_id: int) -> list[float] | None:
        """Pull a raw vector out of a FluxVectorTable by ID.

        Uses the internal ``_vectors`` dict if available (mock), otherwise
        falls back to a zero-query search (which doesn't give us the raw
        vector for real turbovec, so we return None in that case).
        """
        if hasattr(vt._index, "_vectors"):
            vec = vt._index._vectors.get(agent_id)
            if vec is not None:
                return vec.tolist() if hasattr(vec, "tolist") else list(vec)
        return None
