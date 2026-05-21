"""BreedingDaemon — orchestrates AutoBreeder with vector tables and compaction.

Wires together:
    - AutoBreeder (tournament + breed + rebirth)
    - FluxVectorTable (embedding-driven parent selection)
    - CompactionManager (archive sunset agents, periodic compaction)
"""

from __future__ import annotations

__all__ = ["FluxVectorTable", "CompactionManager", "BreedingDaemon"]

import logging
import threading
import time
from typing import Optional

import numpy as np

from nerve.room_grid import RoomGrid
from swarm.breeder_daemon import AutoBreeder
from swarm.thermal import DeviceType, ThermalBudget
from swarm.tournament import AgentScore, sunset_candidates

logger = logging.getLogger(__name__)


class FluxVectorTable:
    """Vector store for agent embeddings — enables similarity-based parent selection.

    Each agent gets a vector (e.g. from its latent output).  The table supports
    cosine-similarity queries for parent selection.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.vectors: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def add(self, agent_id: str, vector: np.ndarray) -> None:
        """Store or update an agent's vector."""
        with self._lock:
            self.vectors[agent_id] = np.asarray(vector, dtype=np.float32).copy()

    def remove(self, agent_id: str) -> bool:
        """Remove an agent's vector. Returns True if removed."""
        with self._lock:
            return self.vectors.pop(agent_id, None) is not None

    def query(
        self,
        query_vector: np.ndarray,
        k: int = 3,
        exclude: set[str] | None = None,
    ) -> list[str]:
        """Return top-k most similar agent_ids by cosine similarity."""
        with self._lock:
            if not self.vectors:
                return []
            q = np.asarray(query_vector, dtype=np.float32)
            qn = q / (np.linalg.norm(q) + 1e-8)
            sims: dict[str, float] = {}
            for aid, vec in self.vectors.items():
                if exclude and aid in exclude:
                    continue
                vn = vec / (np.linalg.norm(vec) + 1e-8)
                sims[aid] = float(np.dot(qn, vn))
            return sorted(sims, key=sims.get, reverse=True)[:k]

    def top_k(self, k: int = 3) -> list[str]:
        """Return top-k agents by vector L2 norm."""
        with self._lock:
            if not self.vectors:
                return []
            norms = {
                aid: float(np.linalg.norm(vec))
                for aid, vec in self.vectors.items()
            }
            return sorted(norms, key=norms.get, reverse=True)[:k]


class CompactionManager:
    """Archives sunset agents and periodically compacts storage."""

    def __init__(self, max_archive_size: int = 1000) -> None:
        self.archive: list[dict] = []
        self.max_archive_size = max_archive_size
        self._lock = threading.Lock()
        self._compaction_count = 0

    def archive_sunset(
        self,
        agent_id: str,
        scores: AgentScore | None = None,
        tick: int = 0,
    ) -> None:
        """Archive an agent that has been sunset."""
        with self._lock:
            record: dict = {
                "agent_id": agent_id,
                "tick": tick,
                "timestamp": time.time(),
            }
            if scores is not None:
                record["ethos"] = scores.ethos
                record["pathos"] = scores.pathos
                record["logos"] = scores.logos
            self.archive.append(record)

    def compact(self, dedupe: bool = True) -> int:
        """Compact the archive: dedupe and trim to max size.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            before = len(self.archive)
            if dedupe:
                seen: set[str] = set()
                deduped: list[dict] = []
                for rec in self.archive:
                    key = rec["agent_id"]
                    if key not in seen:
                        seen.add(key)
                        deduped.append(rec)
                self.archive = deduped

            if len(self.archive) > self.max_archive_size:
                # Keep most recent
                self.archive = self.archive[-self.max_archive_size :]

            removed = before - len(self.archive)
            self._compaction_count += 1
            logger.info(
                "Compaction #%d: removed %d records",
                self._compaction_count,
                removed,
            )
            return removed

    @property
    def size(self) -> int:
        with self._lock:
            return len(self.archive)


class BreedingDaemon:
    """Orchestrates the breeding loop with optional vector tables and compaction.

    Wraps AutoBreeder and adds:
    - FluxVectorTable for embedding-driven parent selection
    - CompactionManager for archiving sunset agents
    - Periodic compaction calls
    """

    def __init__(
        self,
        grid: RoomGrid,
        thermal: ThermalBudget,
        vector_table: Optional[FluxVectorTable] = None,
        compaction: Optional[CompactionManager] = None,
        interval: int = 10,
        cold_threshold: int = 3,
        n_winners: int = 3,
        device: DeviceType = DeviceType.GPU,
        compaction_interval: int = 5,
    ) -> None:
        self.auto_breeder = AutoBreeder(
            grid=grid,
            thermal=thermal,
            interval=interval,
            cold_threshold=cold_threshold,
            n_winners=n_winners,
            device=device,
        )
        self.vector_table = vector_table
        self.compaction = compaction or CompactionManager()
        self.compaction_interval = compaction_interval
        self._cycle_count = 0

    def cycle(self, n_winners: Optional[int] = None) -> list[tuple[int, str]]:
        """Run one breeding cycle with vector-aware parent selection.

        Archives sunset candidates via CompactionManager after breeding.
        Runs compaction every *compaction_interval* cycles.

        Returns:
            List of (reborn_room_id, parent_agent_id) tuples.
        """
        self._cycle_count += 1

        # Identify dominated agents as sunset candidates BEFORE rebirth
        all_scores = [
            AgentScore(
                agent_id=f"room_{rid}",
                ethos=float(self.auto_breeder.grid.activity[rid])
                / max(1, int(self.auto_breeder.grid.activity.max())),
                pathos=float(self.auto_breeder.grid.activity[rid])
                / max(1, int(self.auto_breeder.grid.activity.max())),
                logos=float(self.auto_breeder.grid.activity[rid])
                / max(1, int(self.auto_breeder.grid.activity.max())),
            )
            for rid in range(self.auto_breeder.grid.n)
        ]
        to_sunset = sunset_candidates(all_scores)
        for candidate in to_sunset:
            self.compaction.archive_sunset(
                agent_id=candidate.agent_id,
                scores=candidate,
                tick=self.auto_breeder._tick_count,
            )

        # Run breeding with vector table if available
        results = self.auto_breeder.auto_breed(
            n_winners=n_winners,
            vector_table=self.vector_table,
        )

        # Remove vectored entries for rebirthed rooms (old occupant is gone)
        if self.vector_table is not None:
            for room_id, _parent_id in results:
                self.vector_table.remove(f"room_{room_id}")

        # Periodic compaction
        if self._cycle_count % self.compaction_interval == 0:
            self.compaction.compact()

        return results

    def start(self) -> None:
        """Delegate to AutoBreeder daemon thread."""
        self.auto_breeder.start()

    def stop(self) -> None:
        """Delegate to AutoBreeder daemon thread."""
        self.auto_breeder.stop()

    @property
    def running(self) -> bool:
        return self.auto_breeder.running

    @property
    def log(self) -> list:
        return self.auto_breeder.log
