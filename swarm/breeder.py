"""BreedingDaemon — orchestrates AutoBreeder with compaction and sunset archiving.

Wires together:
    - AutoBreeder (tournament + breed + rebirth, optional vector table)
    - CompactionManager (archive sunset agents, periodic compaction)
    - sunset_candidates (identifies dominated agents for archiving)

This module is additive — it does not replace AutoBreeder but wraps it
with fleet-level lifecycle management.
"""

from __future__ import annotations

__all__ = ["BreedingDaemon"]

import logging
from typing import Optional

from nerve.room_grid import RoomGrid
from swarm.breeder_daemon import AutoBreeder
from swarm.compaction import CompactionManager
from swarm.thermal import DeviceType, ThermalBudget
from swarm.tournament import AgentScore, sunset_candidates

logger = logging.getLogger(__name__)


class BreedingDaemon:
    """Orchestrates the breeding loop with optional vector tables and compaction.

    Wraps AutoBreeder and adds:
    - Pre-breeding sunset archiving via CompactionManager
    - Periodic compaction calls every N cycles
    - Public ``select_parents()`` passthrough for introspection

    Args:
        grid: RoomGrid instance.
        thermal: ThermalBudget instance.
        vector_table: Optional FluxVectorTable for embedding-driven parent
            selection (passed directly to AutoBreeder).
        compaction: Optional CompactionManager for archiving sunset agents.
        interval: Daemon tick interval in seconds.
        cold_threshold: Activity threshold for cold rooms.
        n_winners: Tournament winners to use as parent pool.
        device: Default device type for thermal allocation.
        compaction_interval: Run compaction every N cycles.
    """

    def __init__(
        self,
        grid: RoomGrid,
        thermal: ThermalBudget,
        vector_table=None,
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
            vector_table=vector_table,
        )
        self.compaction = compaction
        self.compaction_interval = compaction_interval
        self._cycle_count = 0

    # ── public API ──────────────────────────────────────────

    def cycle(self, n_winners: Optional[int] = None) -> list[tuple[int, str]]:
        """Run one breeding cycle with archiving and periodic compaction.

        Steps:
            1. Identify dominated agents as sunset candidates and archive them.
            2. Run AutoBreeder.auto_breed() (vector-aware if table provided).
            3. Trigger compaction every *compaction_interval* cycles.

        Returns:
            List of (reborn_room_id, parent_agent_id) tuples.
        """
        self._cycle_count += 1

        # Step 1: Archive sunset candidates (dominated agents)
        if self.compaction is not None:
            self._archive_sunset_candidates()

        # Step 2: Run breeding (vector table handled internally by AutoBreeder)
        results = self.auto_breeder.auto_breed(n_winners=n_winners)

        # Step 3: Periodic compaction
        if (
            self.compaction is not None
            and self._cycle_count % self.compaction_interval == 0
        ):
            summary = self.compaction.compact()
            if summary is not None:
                logger.info(
                    "Compaction cycle %d: archived %d agents",
                    self._cycle_count,
                    summary.archived_count,
                )

        return results

    def select_parents(
        self,
        n_winners: Optional[int] = None,
        use_vector: bool = True,
    ) -> list[AgentScore]:
        """Passthrough to AutoBreeder.select_parents().

        Returns tournament winners (or vector-selected parents when a table
        is configured and *use_vector* is True).
        """
        return self.auto_breeder.select_parents(
            n_winners=n_winners, use_vector=use_vector
        )

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

    # ── internals ───────────────────────────────────────────

    def _archive_sunset_candidates(self) -> None:
        """Find dominated agents and archive them via CompactionManager."""
        grid = self.auto_breeder.grid
        max_activity = max(1, int(grid.activity.max()))

        all_scores = [
            AgentScore(
                agent_id=f"room_{rid}",
                ethos=float(grid.activity[rid]) / max_activity,
                pathos=float(grid.activity[rid]) / max_activity,
                logos=float(grid.activity[rid]) / max_activity,
            )
            for rid in range(grid.n)
        ]

        to_sunset = sunset_candidates(all_scores)
        for candidate in to_sunset:
            numeric_id = self._agent_id_to_numeric(candidate.agent_id)
            self.compaction.archive_sunset(numeric_id)
            logger.debug(
                "Archived sunset agent %s (numeric=%d)",
                candidate.agent_id,
                numeric_id,
            )

    @staticmethod
    def _agent_id_to_numeric(agent_id: str) -> int:
        """Convert 'room_N' to numeric ID for CompactionManager.

        Uses the same convention as AutoBreeder._agent_id_to_numeric.
        """
        if agent_id.startswith("room_"):
            return int(agent_id.split("_")[1])
        try:
            return int(agent_id, 16)
        except ValueError:
            import hashlib
            digest = hashlib.blake2b(agent_id.encode(), digest_size=8).digest()
            return int.from_bytes(digest, "big") % (2 ** 64)
