"""CognitionLoop — Perception-reasoning-action cycle for RoomGrid agents.

Provides:
  - AgentConfig: configuration dataclass for agent behavior
  - CognitionLoop: observe → reason → act cycle that can be wired into RoomGrid.tick()

The loop runs at the end of each tick when ``enable_cognition=True`` in
AgentConfig. It observes the grid state, reasons about which rooms need
attention, and acts by breeding, rebirthing, or adjusting chaos.
"""

from __future__ import annotations

__all__ = ["AgentConfig", "CognitionLoop", "CognitionState"]

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent operating inside a RoomGrid.

    Parameters
    ----------
    enable_cognition : bool
        If True, RoomGrid.tick() invokes CognitionLoop.loop() after
        the forward pass.  Default False (no overhead).
    cognition_interval : int
        Run cognition only every N ticks to amortise cost.
        Default 1 (every tick).
    cold_threshold : int
        Rooms with activity below this are considered "cold".
        Default 1.
    hot_threshold : int
        Rooms with activity above this are considered "hot".
        Default 50.
    breed_candidates : int
        Max number of breed actions per cognition cycle.
        Default 3.
    rebirth_candidates : int
        Max number of rebirth actions per cognition cycle.
        Default 2.
    chaos_boost : float
        Amount to boost chaos for cold rooms.
        Default 0.1 (clamped so chaos stays ≤ 1.0).
    chaos_decay : float
        Amount to decay chaos for over-active rooms.
        Default 0.05 (clamped so chaos stays ≥ 0.01).
    top_k_observed : int
        Number of top-active rooms to include in observations.
        Default 10.
    """

    enable_cognition: bool = False
    cognition_interval: int = 1
    cold_threshold: int = 1
    hot_threshold: int = 50
    breed_candidates: int = 3
    rebirth_candidates: int = 2
    chaos_boost: float = 0.1
    chaos_decay: float = 0.05
    top_k_observed: int = 10


@dataclass
class CognitionState:
    """Snapshot of grid state produced by CognitionLoop.observe()."""

    tick: int
    n_rooms: int
    active_count: int
    cold_count: int
    top_active: List[Tuple[int, int]]  # (room_id, activity)
    top_novel: List[Tuple[int, float]]  # (room_id, novelty_score)
    mean_chaos: float
    mean_novelty: float
    fired_ids: List[int]
    latents_shape: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dict for logging / JSON."""
        return {
            "tick": self.tick,
            "n_rooms": self.n_rooms,
            "active_count": self.active_count,
            "cold_count": self.cold_count,
            "top_active": self.top_active,
            "top_novel": self.top_novel,
            "mean_chaos": round(float(self.mean_chaos), 4),
            "mean_novelty": round(float(self.mean_novelty), 4),
            "fired_ids": self.fired_ids,
            "latents_shape": self.latents_shape,
        }


class CognitionLoop:
    """Observe → Reason → Act cycle for autonomous RoomGrid management.

    Usage
    -----
    config = AgentConfig(enable_cognition=True)
    loop = CognitionLoop(config)
    grid = RoomGrid(100, agent_config=config)
    grid.tick(signal)   # cognition runs automatically

    Or manually::

        state = loop.observe(grid)
        decisions = loop.reason(state)
        loop.act(grid, decisions)
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self._tick_counter = 0
        self._last_observation: Optional[CognitionState] = None
        self._last_decisions: Optional[Dict[str, Any]] = None

    # ── Observe ─────────────────────────────────────────────

    def observe(self, grid: Any) -> CognitionState:
        """Extract a structured snapshot of grid state.

        Parameters
        ----------
        grid : RoomGrid (or duck-typed)
            Must expose: ``n``, ``ticks``, ``activity``, ``chaos``,
            ``latents``, ``cold(thresh)``.

        Returns
        -------
        CognitionState
        """
        n = getattr(grid, "n", 0)
        ticks = getattr(grid, "ticks", 0)
        activity = getattr(grid, "activity", np.zeros(n, dtype=np.int32))
        chaos = getattr(grid, "chaos", np.full(n, 0.3, dtype=np.float32))
        latents = getattr(grid, "latents", None)

        active_count = int((activity > 0).sum())
        cold_thresh = self.config.cold_threshold
        cold_count = int((activity < cold_thresh).sum())

        # Top-K active rooms
        top_k = min(self.config.top_k_observed, n)
        if n > 0:
            top_idx = np.argsort(activity)[::-1][:top_k]
            top_active = [(int(i), int(activity[i])) for i in top_idx]
        else:
            top_active = []

        # Top-K novel rooms (if latents available and history exists)
        top_novel: List[Tuple[int, float]] = []
        if (
            latents is not None
            and hasattr(grid, "_hist")
            and hasattr(grid, "_hist_count")
        ):
            from nerve.room_grid import batch_novelty

            hist = grid._hist
            hist_count = grid._hist_count
            hist_idx = getattr(grid, "_hist_idx", 0)
            hist_max = getattr(grid, "_hist_max", 20)
            try:
                nv = batch_novelty(latents, hist, hist_count, hist_idx, hist_max)
                novel_idx = np.argsort(nv)[::-1][:top_k]
                top_novel = [(int(i), float(nv[i])) for i in novel_idx]
                mean_novelty = float(nv.mean())
            except Exception as e:
                log.debug("Novelty computation in observe() failed: %s", e)
                mean_novelty = 0.5
        else:
            mean_novelty = 0.5

        fired_ids = []
        if hasattr(grid, "_last_fired_ids"):
            fired_ids = list(grid._last_fired_ids)

        state = CognitionState(
            tick=ticks,
            n_rooms=n,
            active_count=active_count,
            cold_count=cold_count,
            top_active=top_active,
            top_novel=top_novel,
            mean_chaos=float(chaos.mean()),
            mean_novelty=mean_novelty,
            fired_ids=fired_ids,
            latents_shape=latents.shape if latents is not None else (),
        )
        self._last_observation = state
        return state

    # ── Reason ──────────────────────────────────────────────

    def reason(self, state: CognitionState) -> Dict[str, Any]:
        """Turn observations into a decision dict.

        Decisions
        ---------
        breed_pairs : List[Tuple[int, int]]
            (src, dst) room pairs to breed.
        rebirth_ids : List[int]
            Rooms to rebirth (reset weights).
        chaos_adjustments : List[Tuple[int, float]]
            (room_id, new_chaos) overrides.
        """
        decisions: Dict[str, Any] = {
            "breed_pairs": [],
            "rebirth_ids": [],
            "chaos_adjustments": [],
        }

        if state.n_rooms == 0:
            self._last_decisions = decisions
            return decisions

        # ── Chaos adjustments ──────────────────────────────
        # Boost cold rooms, decay over-active rooms
        cold_ids = []
        hot_ids = []
        for room_id, activity in state.top_active:
            if activity < self.config.cold_threshold:
                cold_ids.append(room_id)
            elif activity > self.config.hot_threshold:
                hot_ids.append(room_id)

        for rid in cold_ids[: self.config.rebirth_candidates]:
            new_chaos = min(1.0, 0.3 + self.config.chaos_boost)
            decisions["chaos_adjustments"].append((rid, new_chaos))

        for rid in hot_ids[: self.config.breed_candidates]:
            new_chaos = max(0.01, 0.3 - self.config.chaos_decay)
            decisions["chaos_adjustments"].append((rid, new_chaos))

        # ── Rebirth ────────────────────────────────────────
        # Rebirth the coldest rooms that have been cold for a while
        if state.cold_count > 0 and self.config.rebirth_candidates > 0:
            # Pick from the bottom of top_active (least active)
            candidates = [
                rid
                for rid, act in reversed(state.top_active)
                if act < self.config.cold_threshold
            ]
            decisions["rebirth_ids"] = candidates[: self.config.rebirth_candidates]

        # ── Breed ──────────────────────────────────────────
        # Breed top-active rooms into cold rooms to spread good patterns
        if (
            state.cold_count > 0
            and state.active_count > 0
            and self.config.breed_candidates > 0
        ):
            src_pool = [
                rid
                for rid, act in state.top_active
                if act >= self.config.cold_threshold
            ]
            dst_pool = [
                rid for rid, act in state.top_active if act < self.config.cold_threshold
            ]
            pairs: List[Tuple[int, int]] = []
            for src, dst in zip(
                src_pool[: self.config.breed_candidates],
                dst_pool[: self.config.breed_candidates],
            ):
                pairs.append((int(src), int(dst)))
            decisions["breed_pairs"] = pairs

        self._last_decisions = decisions
        return decisions

    # ── Act ─────────────────────────────────────────────────

    def act(self, grid: Any, decisions: Dict[str, Any]) -> None:
        """Execute decisions on the grid.

        Parameters
        ----------
        grid : RoomGrid (or duck-typed)
            Must expose ``breed(src, dst)``, ``rebirth(i)``, and
            have a mutable ``chaos`` array.
        decisions : dict
            Output of ``reason()``.
        """
        # Apply chaos adjustments first (no side effects beyond the array)
        for rid, new_chaos in decisions.get("chaos_adjustments", []):
            if 0 <= rid < getattr(grid, "n", 0):
                grid.chaos[rid] = float(new_chaos)

        # Rebirth before breeding so we don't overwrite newborn rooms
        for rid in decisions.get("rebirth_ids", []):
            if hasattr(grid, "rebirth"):
                try:
                    grid.rebirth(rid)
                except Exception as e:
                    log.warning("rebirth(%d) failed: %s", rid, e)
            else:
                log.debug("grid has no rebirth() method; skipping")

        # Breed top performers into cold rooms
        for src, dst in decisions.get("breed_pairs", []):
            if hasattr(grid, "breed"):
                try:
                    grid.breed(src, dst)
                except Exception as e:
                    log.warning("breed(%d, %d) failed: %s", src, dst, e)
            else:
                log.debug("grid has no breed() method; skipping")

    # ── Loop ────────────────────────────────────────────────

    def loop(self, grid: Any) -> Dict[str, Any]:
        """One full cognition cycle: observe → reason → act.

        Returns
        -------
        dict
            The decisions dict for introspection / logging.
        """
        self._tick_counter += 1
        interval = max(1, self.config.cognition_interval)
        if (self._tick_counter % interval) != 0:
            return {}

        state = self.observe(grid)
        decisions = self.reason(state)
        self.act(grid, decisions)
        log.debug("CognitionLoop tick=%d decisions=%s", state.tick, decisions)
        return decisions

    def reset(self) -> None:
        """Reset internal counters (useful for deterministic tests)."""
        self._tick_counter = 0
        self._last_observation = None
        self._last_decisions = None

    @property
    def last_observation(self) -> Optional[CognitionState]:
        """Most recent observation (or None if loop never ran)."""
        return self._last_observation

    @property
    def last_decisions(self) -> Optional[Dict[str, Any]]:
        """Most recent decisions (or None if loop never ran)."""
        return self._last_decisions
