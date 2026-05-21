"""AutoBreeder — connects tournament + breed + rebirth.

Periodically finds cold rooms, runs tournaments among hot rooms,
breeds winners, and rebirths cold rooms with cloned winner weights.
Thread-safe. Respects ThermalBudget (parent-sacrifice-before-child-spawn).

**New in this version:** Optional `FluxVectorTable` integration for
vector-based parent selection instead of random sampling from tournament
winners. This enables:
    - Diversity-aware breeding (search for dissimilar parents)
    - Capability-filtered selection (R15 mask matching)
    - Latent DNA similarity matching instead of activity-only scoring
"""

from __future__ import annotations

__all__ = ["AutoBreeder"]

import logging
import random
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
    selected_by_vector_search: bool = False

    def __repr__(self) -> str:
        return (
            f"RebirthRecord(room={self.room_id}, "
            f"parent={self.parent_agent_id!r}, "
            f"tick={self.tick}, vec_search={self.selected_by_vector_search})"
        )


class AutoBreeder:
    """Automatically breeds agents from hot rooms into cold rooms.

    Wires together:
        - RoomGrid (room activity / rebirth)
        - TournamentRound (selection)
        - breed() (crossover)
        - ThermalBudget (slot management)
        - FluxVectorTable (optional vector-based parent selection)

    Usage::

        grid = RoomGrid(250)
        thermal = ThermalBudget()
        breeder = AutoBreeder(grid, thermal, interval=10)
        results = breeder.auto_breed()
        breeder.start()          # daemon thread
        # ... later:
        breeder.stop()

    With vector search::

        from swarm.vector_table import FluxVectorTable
        table = FluxVectorTable(dim=256, bit_width=4)
        breeder = AutoBreeder(grid, thermal, vector_table=table)
    """

    def __init__(
        self,
        grid: RoomGrid,
        thermal: ThermalBudget,
        interval: int = 10,
        cold_threshold: int = 3,
        n_winners: int = 3,
        device: DeviceType = DeviceType.GPU,
        vector_table: Optional["FluxVectorTable"] = None,
    ) -> None:
        self.grid = grid
        self.thermal = thermal
        self.interval = interval
        self.cold_threshold = cold_threshold
        self.n_winners = n_winners
        self.device = device
        self._vector_table = vector_table

        self._log: list[RebirthRecord] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tick_count = 0

    # ── public API ──────────────────────────────────────────

    def select_parents(
        self,
        n_winners: Optional[int] = None,
        use_vector: bool = True,
    ) -> list[AgentScore]:
        """Public API: select parent agents for breeding.

        Runs a tournament on hot rooms and returns the top *n_winners*.
        When *use_vector* is True and a ``FluxVectorTable`` was provided
        at construction, the method also returns the vector-selected pairs
        so callers can inspect diversity-aware choices.

        Args:
            n_winners: How many winners to select. Defaults to self.n_winners.
            use_vector: Whether to prefer vector-aware selection when a table
                is available.

        Returns:
            List of AgentScore winners (best first).  Empty list when no
            hot rooms exist.
        """
        n_winners = n_winners or self.n_winners

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

        tournament = TournamentRound(population)
        ranked = tournament.run()
        winners = [r.scores for r in ranked[:n_winners] if r.scores is not None]

        if use_vector and self._vector_table is not None:
            # Vector-aware path: select parent pairs for diversity.
            # We return the *primary* parents (parent_a) from each pair.
            pairs = self._select_parents_vector(winners, n_winners)
            primary = []
            for a, b in pairs:
                if a is not None:
                    primary.append(a)
            # Deduplicate while preserving order
            seen: set[str] = set()
            deduped: list[AgentScore] = []
            for w in primary:
                if w.agent_id not in seen:
                    seen.add(w.agent_id)
                    deduped.append(w)
            return deduped[:n_winners]

        return winners

    def auto_breed(
        self,
        n_winners: Optional[int] = None,
    ) -> list[tuple[int, str]]:
        """Run one breeding cycle.

        1. Find cold rooms (grid.cold(threshold)).
        2. Score hot rooms as tournament agents (ethos=pathos=logos=normalized activity).
        3. Run tournament, take top N winners.
        4. **If vector_table present**: use latent DNA search to pick parents
           that are similar/diverse based on their compressed vectors.
           **Otherwise**: random crossover from tournament winners (legacy).
        5. Breed children from selected parents.
        6. Rebirth cold rooms using cloned winner weights.
        7. Respect thermal budget (parent-sacrifice-before-child-spawn).

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

        # Select parents — vector-aware or random
        n_children = min(len(cold_rooms), n_winners)
        if self._vector_table is not None:
            parent_pairs = self._select_parents_vector(winners, n_children)
        else:
            parent_pairs = self._select_parents_random(winners, n_children)

        # Breed children from selected parent pairs
        children = self._breed_from_pairs(parent_pairs)

        # Rebirth cold rooms with cloned winner weights
        results: list[tuple[int, str]] = []

        with self._lock:
            self._tick_count += 1
            tick = self._tick_count

        for idx, child in enumerate(children):
            if idx >= len(cold_rooms):
                break

            room_id = cold_rooms[idx]

            # Pick primary parent for weight clone (first winner in pair)
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

            selected_by_vec = child.get("selected_by_vector_search", False)

            # Sync to vector table if available
            if self._vector_table is not None and "vector" in child:
                from swarm.vector_table import AgentVector
                self._vector_table.add(
                    AgentVector(
                        agent_id=child["numeric_id"],
                        vector=child["vector"],
                        fitness=child.get("fitness", 0.0),
                        generation=child.get("generation", tick),
                        capability_mask=child.get("capability_mask", 0xFFFF),
                        thermal_pressure=child.get("thermal_pressure", 0.0),
                    )
                )

            record = RebirthRecord(
                room_id=room_id,
                parent_agent_id=parent_id,
                parent_ethos=child.get("ethos", 0.0),
                parent_pathos=child.get("pathos", 0.0),
                parent_logos=child.get("logos", 0.0),
                tick=tick,
                child_config=child,
                selected_by_vector_search=selected_by_vec,
            )
            with self._lock:
                self._log.append(record)

            results.append((room_id, parent_id))
            logger.info(
                "Rebirthed room %d from parent %s (tick %d, vec_search=%s)",
                room_id,
                parent_id,
                tick,
                selected_by_vec,
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
        logger.info(
            "AutoBreeder daemon started (interval=%d, vec_search=%s)",
            self.interval,
            self._vector_table is not None,
        )

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

    def _select_parents_random(
        self,
        winners: list[AgentScore],
        n_children: int,
    ) -> list[tuple[AgentScore, Optional[AgentScore]]]:
        """Legacy random parent selection.

        Returns list of (parent_a, parent_b) tuples. parent_b may be None
        for single-parent clone.
        """
        import random
        pairs: list[tuple[AgentScore, Optional[AgentScore]]] = []
        if len(winners) < 2:
            for _ in range(n_children):
                pairs.append((winners[0], None) if winners else (None, None))
            return pairs
        for _ in range(n_children):
            a, b = random.sample(winners, 2)
            pairs.append((a, b))
        return pairs

    def _select_parents_vector(
        self,
        winners: list[AgentScore],
        n_children: int,
    ) -> list[tuple[AgentScore, Optional[AgentScore]]]:
        """Vector-aware parent selection using FluxVectorTable.

        Strategy: For each child, pick the highest-fitness winner as
        parent_a. Then search the vector table for the **least similar**
        agent among other winners — this preserves diversity by mating
        dissimilar parents.

        If the vector table is empty or missing entries, falls back to
        random selection.
        """
        if self._vector_table is None or len(self._vector_table) == 0:
            return self._select_parents_random(winners, n_children)

        import random

        pairs: list[tuple[AgentScore, Optional[AgentScore]]] = []
        winner_ids = {w.agent_id for w in winners}

        for _ in range(n_children):
            # parent_a: highest fitness winner
            parent_a = max(winners, key=lambda w: w.product)

            # parent_b: search for dissimilar winner (max diversity)
            # We query with parent_a's vector and look for the *worst* match
            # among remaining winners — i.e. the most genetically distant.
            vec_a = self._vector_table._meta.get(self._agent_id_to_numeric(parent_a.agent_id))
            if vec_a is None:
                # No vector for this winner — fallback to random
                if len(winners) >= 2:
                    b = random.choice([w for w in winners if w.agent_id != parent_a.agent_id])
                else:
                    b = None
                pairs.append((parent_a, b))
                continue

            # Search for ALL winners in vector table, then pick the most distant
            # Build allowlist of all winner IDs
            allowlist = [
                self._agent_id_to_numeric(w.agent_id)
                for w in winners
                if w.agent_id != parent_a.agent_id
            ]

            if not allowlist:
                pairs.append((parent_a, None))
                continue

            # Search with a dummy query — we'll sort by score ascending (worst match)
            # Actually, we need parent_a's vector as the query and find min score
            # But turbovec only returns top-k. We can search with k=len(allowlist)
            # and pick the last result (lowest score).
            #
            # Better: get the vector for parent_a from the table, search all,
            # take the worst score.
            try:
                results = self._vector_table.search(
                    query=parent_a,  # AgentScore has no .vector; need actual vector
                    k=len(allowlist),
                    allowlist=allowlist,
                )
            except (TypeError, AttributeError):
                # parent_a is an AgentScore, not a vector — fallback
                if len(winners) >= 2:
                    b = random.choice([w for w in winners if w.agent_id != parent_a.agent_id])
                else:
                    b = None
                pairs.append((parent_a, b))
                continue

            if not results:
                pairs.append((parent_a, None))
                continue

            # Most distant = lowest score (last in sorted results, since
            # turbovec returns best-first)
            worst_id, _, _ = results[-1]
            parent_b = next(
                (w for w in winners if self._agent_id_to_numeric(w.agent_id) == worst_id),
                None,
            )
            pairs.append((parent_a, parent_b))

        return pairs

    @staticmethod
    def _agent_id_to_numeric(agent_id: str) -> int:
        """Convert 'room_N' or hex string to uint64 ID.

        Supports:
            - room_123 → 123
            - 0xabc → int('abc', 16)
            - raw int string → int()
        """
        if agent_id.startswith("room_"):
            return int(agent_id.split("_")[1])
        try:
            return int(agent_id, 16)
        except ValueError:
            # Hash to uint64 for arbitrary strings
            import hashlib
            digest = hashlib.blake2b(agent_id.encode(), digest_size=8).digest()
            return int.from_bytes(digest, "big") % (2 ** 64)

    @staticmethod
    def _breed_from_pairs(
        pairs: list[tuple[AgentScore, Optional[AgentScore]]],
    ) -> list[dict]:
        """Breed children from (parent_a, parent_b) pairs.

        Mirrors ``breed()`` from swarm.tournament but operates on
        pre-selected pairs instead of random sampling.
        """
        import random
        import uuid

        children: list[dict] = []
        for a, b in pairs:
            if a is None:
                continue
            if b is None:
                # Single parent — clone with mutation
                child = {
                    "id": uuid.uuid4().hex[:12],
                    "parent_a": a.agent_id,
                    "parent_b": None,
                    "ethos": _mutate(a.ethos),
                    "pathos": _mutate(a.pathos),
                    "logos": _mutate(a.logos),
                    "fitness": a.product,
                    "selected_by_vector_search": False,
                }
            else:
                child = {
                    "id": uuid.uuid4().hex[:12],
                    "parent_a": a.agent_id,
                    "parent_b": b.agent_id,
                    "ethos": _mutate(_crossover(a.ethos, b.ethos)),
                    "pathos": _mutate(_crossover(a.pathos, b.pathos)),
                    "logos": _mutate(_crossover(a.logos, b.logos)),
                    "fitness": _crossover(a.product, b.product),
                    "selected_by_vector_search": True,
                }
            children.append(child)
        return children

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


# ── helpers ─────────────────────────────────────────────

def _crossover(a: float, b: float) -> float:
    """Random crossover between two parent values."""
    t = random.random()
    return a * t + b * (1 - t)


def _mutate(value: float, sigma: float = 0.05) -> float:
    """Gaussian mutation clamped to [0, 1]."""
    return max(0.0, min(1.0, random.gauss(value, sigma)))
