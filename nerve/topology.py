"""NerveTopology — The living network.

Wires nerve fibers to RoomGrid rooms via Hebbian routing.
One tick = perceive → route → fire rooms → tiles flow back → reinforce.

The COLLECT → SELECT → COMPILE lifecycle:
  COLLECT:  Fiber perceives raw signal (PERCEIVING state)
  SELECT:   Grid rooms fire based on novelty (tournament-like selection)
  COMPILE:  Repeated patterns compile routes (Hebbian strengthening → automatic)
  FEEDBACK: Reception signal reinforces strong routes, weakens poor ones
  REGULATE: Chaos probability decays as routes compile
"""
from __future__ import annotations

__all__ = ["NerveTopology", "TickResult"]

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from nerve.room_grid import RoomGrid
from nerve.fiber import NerveFiber, FiberState, SensoryTile
from nerve.routing import RoutingLayer

log = logging.getLogger(__name__)


@dataclass
class TickResult:
    """Snapshot of one topology tick."""
    tick: int
    fibers_perceived: int
    rooms_fired: int
    routes_activated: int
    routes_compiled: int
    novel_signals: int
    latency_ms: float


class NerveTopology:
    """Orchestrates the full nerve fiber → grid → routing → feedback cycle.

    Architecture:
        Signal → Fiber.perceive() → SensoryTile
               → RoutingLayer.fire(fiber, rooms)
               → Rooms that fire: receive tile as grid input
               → RoomGrid.tick(encoded_tile)
               → Results feed back through RoutingLayer.feedback()
               → Hebbian channels strengthen between co-firing rooms
    """

    def __init__(
        self,
        n_fibers: int = 8,
        n_rooms: int = 250,
        chaos: float = 0.3,
        adapt_threshold: float = 0.95,
        learning_rate: float = 0.05,
        signal_dim: int = 64,
    ):
        self.n_fibers = n_fibers
        self.n_rooms = n_rooms
        self.signal_dim = signal_dim

        # Components
        self.grid = RoomGrid(n_rooms, d=signal_dim)
        self.routing = RoutingLayer(chaos=chaos, learning_rate=learning_rate)

        # Fibers: each is a unique perception channel
        self.fibers: dict[str, NerveFiber] = {}
        for i in range(n_fibers):
            fid = f"fiber-{i}"
            self.fibers[fid] = NerveFiber(
                fiber_id=fid,
                model_type="jepa",
                adapt_threshold=adapt_threshold,
                novelty_threshold=0.3,
                epsilon=0.05,
            )

        # Wire routing: every fiber → every room
        for fid in self.fibers:
            for room_idx in range(n_rooms):
                rid = f"room-{room_idx}"
                self.routing.add_route(fid, rid, strength=0.1)

        # Hebbian channels between adjacent rooms (Penrose neighbors)
        try:
            from swarm.penrose import assign_positions
            positions = assign_positions([f"room-{i}" for i in range(n_rooms)])
            for i in range(len(positions) - 1):
                a = positions[i].agent_id
                b = positions[i + 1].agent_id
                self.routing.add_channel(a, b, weight=0.1)
        except Exception as exc:
            log.warning("Could not create Penrose channels: %s", exc)

        self.tick_count = 0
        self._lock = threading.Lock()
        self._results: deque[TickResult] = deque(maxlen=1000)

        # Adaptive chaos: decays with compilation progress
        self._base_chaos = chaos

    def __repr__(self) -> str:
        return (
            f"NerveTopology(fibers={self.n_fibers}, rooms={self.n_rooms}, "
            f"tick={self.tick_count})"
        )

    @property
    def stats(self) -> dict[str, Any]:
        compiled_fibers = sum(
            1 for f in self.fibers.values()
            if f.state == FiberState.COMPILED
        )
        return {
            "tick": self.tick_count,
            "fibers": self.n_fibers,
            "rooms": self.n_rooms,
            "rooms_active": int((self.grid.activity > 0).sum()),
            "rooms_cold": len(self.grid.cold()),
            "fibers_compiled": compiled_fibers,
            "fibers_perceiving": self.n_fibers - compiled_fibers,
            "routes": len(self.routing._routes),
            "channels": len(self.routing._channels),
            "chaos": self.routing.chaos,
        }

    def _encode_tile(self, tile: SensoryTile) -> np.ndarray:
        """Encode a SensoryTile into a signal vector for the grid.

        PERFORMANCE: Cached by pattern_id — avoids repeated RandomState creation.
        The old implementation created a RandomState per call (0.5ms each),
        which dominated tick() latency at 150+ calls per tick.
        """
        # Cache key: pattern_id + state (deterministic encoding)
        cache_key = (tile.pattern_id, tile.state.value)
        if hasattr(self, '_tile_cache') and cache_key in self._tile_cache:
            signal = self._tile_cache[cache_key].copy()
        else:
            if not hasattr(self, '_tile_cache'):
                self._tile_cache = {}
            rng = np.random.RandomState(hash(tile.pattern_id) % (2**31))
            signal = rng.randn(self.signal_dim).astype(np.float32) * 0.1
            self._tile_cache[cache_key] = signal.copy()

        # Energy scaling by fiber state
        if tile.state == FiberState.COMPILED:
            signal *= 0.3
        elif tile.state == FiberState.NOVELTY_ALERT:
            signal *= 2.0
        elif tile.state == FiberState.PERCEIVING:
            signal *= 1.0
        elif tile.state == FiberState.ADAPTING:
            signal *= 0.7

        signal *= (0.5 + tile.confidence * 0.5)
        return signal

    def tick(self, signals: dict[str, Any] | None = None) -> TickResult:
        """One full topology tick — optimized for speed."""
        t0 = time.perf_counter()
        self.tick_count += 1

        # ── COLLECT: All fibers perceive ──────────────────────
        tiles: dict[str, SensoryTile] = {}
        novel_count = 0

        for fid, fiber in self.fibers.items():
            raw = signals.get(fid, np.random.randn(self.signal_dim)) if signals else None
            if raw is None:
                raw = np.random.randn(self.signal_dim).astype(np.float32)

            tile = fiber.perceive(raw)
            tiles[fid] = tile
            if tile.state in (FiberState.NOVELTY_ALERT, FiberState.PERCEIVING):
                novel_count += 1

        # ── SELECT: Route tiles to rooms (FAST PATH) ──────────
        room_signals: dict[str, list[tuple[str, SensoryTile]]] = {}
        routes_activated = 0

        for fid, tile in tiles.items():
            fired = self.routing.fire_fast(fid)  # ← vectorized
            routes_activated += len(fired)
            for rid in fired:
                room_signals.setdefault(rid, []).append((fid, tile))

        # ── COMPILE: Grid processes combined signal ────────────
        combined = np.zeros(self.signal_dim, dtype=np.float32)
        for rid, sources in room_signals.items():
            for fid, tile in sources:
                combined += self._encode_tile(tile)

        if np.any(combined != 0):
            combined /= max(1, len(room_signals))
            grid_result = self.grid.tick(combined)
        else:
            grid_result = {"fired": 0, "ids": [], "tick": self.tick_count}

        # ── FEEDBACK: Batch reinforce routes ──────────────────
        routes_compiled = 0
        fired_room_ids = {f"room-{i}" for i in grid_result.get("ids", [])}
        feedback_batch: list[tuple[str, str, bool]] = []

        # Collect all feedback into batch
        for fid in self.fibers:
            for rid in fired_room_ids:
                feedback_batch.append((fid, rid, True))

        # Penalize routes to cold rooms (cached)
        cold_rooms = self.grid.cold()
        for room_idx in cold_rooms:
            rid = f"room-{room_idx}"
            for fid in self.fibers:
                feedback_batch.append((fid, rid, False))

        # Apply batch feedback
        self.routing.feedback_batch(feedback_batch)

        # Count compiled routes
        for fid in self.fibers:
            for rid in fired_room_ids:
                route_key = self.routing._route_key(fid, rid)
                route = self.routing._routes.get(route_key)
                if route and route.strength > 0.9:
                    routes_compiled += 1

        # ── REGULATE: Adaptive chaos decay ──────────────────────
        compiled_fraction = routes_compiled / max(1, len(self.routing._routes))
        self.routing.chaos = max(
            0.01,
            self._base_chaos * (1.0 - compiled_fraction) * 0.99 ** self.tick_count,
        )

        # ── Periodic: Decay Hebbian channels ──────────────────
        if self.tick_count % 100 == 0:
            self.routing.decay_all(factor=0.999)

        latency = (time.perf_counter() - t0) * 1000

        result = TickResult(
            tick=self.tick_count,
            fibers_perceived=len(tiles),
            rooms_fired=grid_result.get("fired", 0),
            routes_activated=routes_activated,
            routes_compiled=routes_compiled,
            novel_signals=novel_count,
            latency_ms=latency,
        )
        self._results.append(result)
        return result

    def run(self, ticks: int = 1000, signals=None) -> list[TickResult]:
        """Run multiple ticks. Returns results for monitoring."""
        results = []
        for t in range(ticks):
            r = self.tick(signals(t) if callable(signals) else signals)
            results.append(r)
            if t % 100 == 0:
                log.info("Topology tick %d: %s", t, r)
        return results

    def compiled_pathways(self) -> list[dict[str, Any]]:
        """Return routes that have compiled (strength > 0.9)."""
        return [
            {
                "source": r.source,
                "destination": r.destination,
                "strength": r.strength,
                "fires": r.fires,
                "reception": r.reception,
            }
            for r in self.routing._routes.values()
            if r.strength > 0.9
        ]

    def active_fibers(self) -> dict[str, dict]:
        """Return fiber states for monitoring."""
        return {
            fid: {
                "state": f.state.value,
                "confidence": f.confidence,
                "compiled_patterns": len(f._compiled_patterns),
                "total_signals": f._total_signals,
            }
            for fid, f in self.fibers.items()
        }

    def rebirth_cold_rooms(self) -> int:
        """Rebirth all cold rooms with new random weights."""
        cold = self.grid.cold()
        for i in cold:
            self.grid.rebirth(i)
            for fid in self.fibers:
                key = self.routing._route_key(fid, f"room-{i}")
                if key in self.routing._routes:
                    self.routing._routes[key].strength = 0.3
        return len(cold)
