"""Periodic compaction for long-running FluxVectorTable indices.

As agents breed and sunset over millions of generations, the DNA index
grows unbounded. Compaction merges archived (sunset) agents into
summary vectors, freeing memory while preserving lineage statistics.

Design:
    - Archive generation: sunset agents → archive bucket
    - Compaction trigger: archive size > threshold
    - Summary vector: centroid + variance of archived DNA
    - Full fidelity: living agents + recent N generations
    - Rollback: WAL retains pre-compaction state

Usage::

    from swarm.compaction import CompactionManager

    cm = CompactionManager(table=vector_table, max_archive_size=10000)
    cm.archive_sunset(agent_id)          # Mark agent as archived
    if cm.should_compact():
        summary = cm.compact()            # Merge archive into summary
        print(f"Compacted {summary.archived_count} agents → 1 summary vector")
"""

from __future__ import annotations

__all__ = ["CompactionManager", "ArchiveSummary", "CompactionPolicy"]

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ArchiveSummary:
    """Compressed representation of many archived agents."""

    generation_range: tuple[int, int]  # (first, last) generation archived
    archived_count: int
    centroid: list[float]  # mean DNA vector
    variance: list[float]  # per-dim variance
    best_fitness: float
    worst_fitness: float
    dominant_capabilities: int  # union of all capability masks
    timestamp: float  # compaction time


@dataclass
class CompactionPolicy:
    """Tunable thresholds for when and how to compact."""

    max_archive_size: int = 10_000  # Archive before compaction
    max_generations_without_compact: int = 100
    preserve_recent_generations: int = 10  # Always keep last N gens in full fidelity
    min_archive_for_summary: int = 100  # Don't compact tiny archives


class CompactionManager:
    """Manages agent archiving and periodic compaction.

    Args:
        table: FluxVectorTable to manage.
        policy: Compaction thresholds.
        generation_fn: Callable(agent_id) → int, extracts generation from ID.
    """

    def __init__(
        self,
        table: "FluxVectorTable",
        policy: Optional[CompactionPolicy] = None,
        generation_fn: Optional[callable] = None,
    ) -> None:
        self.table = table
        self.policy = policy or CompactionPolicy()
        self.generation_fn = generation_fn or self._default_generation_fn

        # Agent ID → archived flag
        self._archived: set[int] = set()
        # Agent ID → generation number
        self._generations: dict[int, int] = {}
        # List of ArchiveSummary objects (oldest first)
        self._summaries: list[ArchiveSummary] = []

        self._last_compact_generation: int = 0

    @staticmethod
    def _default_generation_fn(agent_id: int) -> int:
        """Extract generation from agent ID.

        Default: agent IDs encode generation in high 32 bits.
        Override if your ID scheme is different.
        """
        return (agent_id >> 32) & 0xFFFFFFFF

    # ── lifecycle ───────────────────────────────────────────

    def record_birth(self, agent_id: int, generation: int) -> None:
        """Register a new agent's generation."""
        self._generations[agent_id] = generation

    def archive_sunset(self, agent_id: int) -> None:
        """Mark an agent as archived (sunset, no longer breeding)."""
        self._archived.add(agent_id)
        logger.debug(
            "Archived agent %d (gen %d)", agent_id, self._generations.get(agent_id, -1)
        )

    # ── compaction triggers ───────────────────────────────

    @property
    def archive_size(self) -> int:
        return len(self._archived)

    def should_compact(self) -> bool:
        """Check if compaction should run."""
        if self.archive_size < self.policy.min_archive_for_summary:
            return False

        if self.archive_size >= self.policy.max_archive_size:
            return True

        # Check generation gap
        if self._generations:
            max_gen = max(self._generations.values())
            if (
                max_gen - self._last_compact_generation
                >= self.policy.max_generations_without_compact
            ):
                return True

        return False

    # ── compaction ────────────────────────────────────────

    def compact(self) -> Optional[ArchiveSummary]:
        """Compact archived agents into a summary vector.

        Returns:
            ArchiveSummary if compaction ran, None if skipped.
        """
        if not self.should_compact():
            return None

        # Determine which agents to compact
        # Preserve recent generations in full fidelity
        preserved_gens = set()
        if self._generations:
            max_gen = max(self._generations.values())
            preserved_gens = set(
                range(
                    max_gen - self.policy.preserve_recent_generations + 1,
                    max_gen + 1,
                )
            )

        to_compact: list[int] = []
        for aid in list(self._archived):
            gen = self._generations.get(aid, 0)
            if gen not in preserved_gens:
                to_compact.append(aid)

        if len(to_compact) < self.policy.min_archive_for_summary:
            logger.info(
                "Compaction skipped: only %d agents eligible (< %d)",
                len(to_compact),
                self.policy.min_archive_for_summary,
            )
            return None

        logger.info("Compacting %d archived agents...", len(to_compact))

        # Build summary statistics
        vectors: list[np.ndarray] = []
        fitnesses: list[float] = []
        capabilities: list[int] = []
        generations: list[int] = []

        for aid in to_compact:
            # Retrieve from table
            vec = self.table._get_vector(aid)
            if vec is None:
                continue
            vectors.append(np.array(vec, dtype=np.float32))
            meta = self.table._meta.get(aid)
            if meta:
                fitnesses.append(meta.fitness)
                capabilities.append(meta.capability_mask)
            generations.append(self._generations.get(aid, 0))

        if not vectors:
            logger.warning(
                "No vectors found for %d agents — compaction aborted", len(to_compact)
            )
            return None

        stack = np.stack(vectors)
        centroid = stack.mean(axis=0).tolist()
        variance = stack.var(axis=0).tolist()

        summary = ArchiveSummary(
            generation_range=(min(generations), max(generations)),
            archived_count=len(to_compact),
            centroid=centroid,
            variance=variance,
            best_fitness=max(fitnesses) if fitnesses else 0.0,
            worst_fitness=min(fitnesses) if fitnesses else 0.0,
            dominant_capabilities=self._union_capabilities(capabilities),
            timestamp=0.0,  # caller can set
        )

        # Remove compacted agents from table
        for aid in to_compact:
            self.table.remove(aid)
            self._archived.discard(aid)
            self._generations.pop(aid, None)

        self._summaries.append(summary)
        if self._generations:
            self._last_compact_generation = max(self._generations.values())

        logger.info(
            "Compaction complete: %d agents → 1 summary (gens %d-%d)",
            summary.archived_count,
            summary.generation_range[0],
            summary.generation_range[1],
        )
        return summary

    @staticmethod
    def _union_capabilities(masks: list[int]) -> int:
        """Bitwise OR of all capability masks."""
        result = 0
        for m in masks:
            result |= m
        return result

    # ── query against summaries ─────────────────────────────

    def search_with_summaries(
        self,
        query_vec: list[float],
        k: int = 5,
        include_summaries: bool = True,
    ) -> list[tuple[int | str, float, "AgentMeta | ArchiveSummary"]]:
        """Search living agents + optionally summary centroids.

        Returns:
            List of (agent_id or "summary_N", score, meta) sorted best-first.
            Summary entries have lower priority than living agents.
        """
        # Search living agents
        living_results = self.table.search(query_vec, k=k * 2)

        if not include_summaries or not self._summaries:
            return living_results[:k]

        # Search summary centroids
        summary_results: list[tuple[str, float, ArchiveSummary]] = []
        for i, summary in enumerate(self._summaries):
            # Cosine similarity to centroid
            q = np.array(query_vec, dtype=np.float32)
            c = np.array(summary.centroid, dtype=np.float32)
            score = float(np.dot(q, c) / (np.linalg.norm(q) * np.linalg.norm(c) + 1e-8))
            summary_results.append((f"summary_{i}", score, summary))

        # Merge: living agents get score boost, summaries penalized
        merged: list[tuple[int | str, float, object]] = []
        for aid, score, meta in living_results:
            merged.append((aid, score * 1.1, meta))  # Boost living
        for sid, score, summary in summary_results:
            merged.append((sid, score * 0.9, summary))  # Penalize archived

        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:k]

    # ── stats ───────────────────────────────────────────────

    @property
    def living_count(self) -> int:
        """Number of non-archived agents in table."""
        return len(self.table._index) - len(self._archived)

    @property
    def summary_count(self) -> int:
        return len(self._summaries)

    @property
    def total_archived(self) -> int:
        """Total agents ever archived (including compacted)."""
        return sum(s.archived_count for s in self._summaries) + len(self._archived)

    def __repr__(self) -> str:
        return (
            f"CompactionManager(living={self.living_count}, "
            f"archive={self.archive_size}, summaries={self.summary_count})"
        )
