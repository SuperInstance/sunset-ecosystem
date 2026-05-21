"""AutoBreeder — connects tournament + breed + rebirth.

Periodically finds cold rooms, runs tournaments among hot rooms,
breeds winners, and rebirths cold rooms with cloned winner weights.
Thread-safe. Respects ThermalBudget (parent-sacrifice-before-child-spawn).
"""

from __future__ import annotations

__all__ = ["AutoBreeder"]

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from nerve.room_grid import RoomGrid
from swarm.thermal import DeviceType, ThermalBudget
from swarm.tournament import AgentScore, TournamentRound, breed

logger = logging.getLogger(__name__)


@dataclass
class RebirthRecord:
    """Log entry for a single rebirth event."""

    room_id: int
    parent_agent_id: str
    parent_ethos: float
    parent_pathos: float
    parent_logos: float
    tick: int
    child_config: Optional[dict] = None

    def __repr__(self) -> str:
        return (
            f"RebirthRecord(room={self.room_id}, "
            f"parent={self.parent_agent_id!r}, "
            f"tick={self.tick})"
        )


class AutoBreeder:
    """Automatically breeds agents from hot rooms into cold rooms.

    Wires together:
        - RoomGrid (room activity / rebirth)
        - TournamentRound (selection)
        - breed() (crossover)
        - ThermalBudget (slot management)

    Usage::

        grid = RoomGrid(250)
        thermal = ThermalBudget()
        breeder = AutoBreeder(grid, thermal, interval=10)
        results = breeder.auto_breed()
        breeder.start()          # daemon thread
        # ... later:
        breeder.stop()
    """

    def __init__(
        self,
        grid: RoomGrid,
        thermal: ThermalBudget,
        interval: int = 10,
        cold_threshold: int = 3,
        n_winners: int = 3,
        device: DeviceType = DeviceType.GPU,
    ) -> None:
        self.grid = grid
        self.thermal = thermal
        self.interval = interval
        self.cold_threshold = cold_threshold
        self.n_winners = n_winners
        self.device = device

        self._log: list[RebirthRecord] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tick_count = 0

    # ── public API ──────────────────────────────────────────

    def auto_breed(
        self,
        n_winners: Optional[int] = None,
    ) -> list[tuple[int, str]]:
        """Run one breeding cycle.

        1. Find cold rooms (grid.cold(threshold)).
        2. Score hot rooms as tournament agents (ethos=pathos=logos=normalized activity).
        3. Run tournament, take top N winners.
        4. Breed children from winners.
        5. Rebirth cold rooms using cloned winner weights.
        6. Respect thermal budget (parent-sacrifice-before-child-spawn).

        Args:
            n_winners: Override for how many tournament winners to use.
                Defaults to self.n_winners.

        Returns:
            List of (reborn_room_id, parent_agent_id) tuples.
        """
        n_winners = n_winners or self.n_winners

        cold_rooms = self.grid.cold(thresh=self.cold_threshold)
        if not cold_rooms:
            return []

        # Build scores for hot rooms
        hot_rooms = self.grid.top(k=max(20, n_winners * 2))
        if not hot_rooms:
            return []

        max_activity = max(a for _, a in hot_rooms) or 1.0
        population = [
            AgentScore(
                agent_id=f"room_{rid}",
                ethos=activity / max_activity,
                pathos=activity / max_activity,
                logos=activity / max_activity,
            )
            for rid, activity in hot_rooms
        ]

        # Run tournament
        tournament = TournamentRound(population)
        ranked = tournament.run()
        winners = [r.scores for r in ranked[:n_winners] if r.scores is not None]

        if not winners:
            return []

        # Breed children
        num_children = min(len(cold_rooms), len(winners))
        children = breed(winners, num_children=num_children)

        # Rebirth cold rooms with cloned winner weights
        results: list[tuple[int, str]] = []

        with self._lock:
            self._tick_count += 1
            tick = self._tick_count

        for idx, child in enumerate(children):
            if idx >= len(cold_rooms):
                break

            room_id = cold_rooms[idx]

            # Pick parent: first winner whose weights we clone
            parent_id = child.get("parent_a") or child.get("parent_b")
            if parent_id is None:
                continue

            # Parse parent room number from "room_N"
            parent_room = int(parent_id.split("_")[1]) if "_" in parent_id else 0

            # Check thermal budget — sacrifice parent slot for child
            if not self.thermal.can_spawn(self.device):
                ok = self.thermal.parent_sacrifice_before_spawn(
                    parent_id=parent_id,
                    child_device=self.device,
                )
                if not ok:
                    logger.warning(
                        "No thermal headroom for room %d, skipping", room_id
                    )
                    continue

            # Clone parent weights for rebirth (instead of random init)
            self._rebirth_with_clone(room_id, parent_room)

            # Allocate child in thermal budget
            child_id = child["id"]
            self.thermal.allocate(child_id, self.device)

            record = RebirthRecord(
                room_id=room_id,
                parent_agent_id=parent_id,
                parent_ethos=child.get("ethos", 0.0),
                parent_pathos=child.get("pathos", 0.0),
                parent_logos=child.get("logos", 0.0),
                tick=tick,
                child_config=child,
            )
            with self._lock:
                self._log.append(record)

            results.append((room_id, parent_id))
            logger.info(
                "Rebirthed room %d from parent %s (tick %d)",
                room_id,
                parent_id,
                tick,
            )

        return results

    def start(self) -> None:
        """Start the background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="auto-breeder",
            daemon=True,
        )
        self._thread.start()
        logger.info("AutoBreeder daemon started (interval=%d)", self.interval)

    def stop(self) -> None:
        """Stop the daemon thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("AutoBreeder daemon stopped")

    @property
    def log(self) -> list[RebirthRecord]:
        """Read-only copy of the rebirth log."""
        with self._lock:
            return list(self._log)

    @property
    def running(self) -> bool:
        """Whether the daemon thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ── internals ───────────────────────────────────────────

    def _rebirth_with_clone(self, target_room: int, source_room: int) -> None:
        """Rebirth target_room using cloned weights from source_room.

        Instead of random init, copies source room's weights and applies
        small noise (mutation).
        """
        with self._lock:
            for key in ("w1", "w2", "w3"):
                self.grid.w[key][target_room] = (
                    self.grid.w[key][source_room].copy()
                    + np.random.randn(*self.grid.w[key][source_room].shape).astype(
                        np.float32
                    )
                    * 0.005  # small mutation
                )
            for key in ("b1", "b2", "b3"):
                self.grid.w[key][0, target_room] = (
                    self.grid.w[key][0, source_room].copy()
                )
            self.grid.activity[target_room] = 0
            self.grid.chaos[target_room] = 0.3
            self.grid.history[target_room] = []

    def _run_loop(self) -> None:
        """Background loop: auto_breed every N ticks."""
        while not self._stop_event.is_set():
            try:
                self.auto_breed()
            except Exception:
                logger.exception("AutoBreeder cycle failed")
            self._stop_event.wait(self.interval)
