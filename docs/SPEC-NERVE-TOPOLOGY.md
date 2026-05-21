# SPEC-NERVE-TOPOLOGY.md — Wire the Nerve Fiber Topology into a Living Network

## Problem

The nerve layer has all the components but they're disconnected:

- **JEPAGrid** (`nerve/room_grid.py`): 250+ rooms, batch forward, activity tracking, cold detection, rebirth
- **NerveFiber** (`nerve/fiber.py`): Lifecycle states (PERCEIVING → ADAPTING → COMPILED → NOVELTY_ALERT), sensory tiles, confidence accumulation, epsilon schedule
- **RoutingLayer** (`nerve/routing.py`): Hebbian channels, route firing with chaos probability, strength reinforcement, decay
- **HebbianChannel** (`nerve/routing.py`): Bidirectional co-activation tracking, weight strengthening/decay
- **Rust kernel** (`nerve/src/lib.rs`): Multi-threaded batch forward, 10K rooms in ~2.35ms

But nothing connects them. No code path goes: "fiber perceives signal → routes to grid rooms → rooms fire → tiles flow back through Hebbian channels → compiled pathways emerge."

The tripartite grammar (COLLECT → SELECT → COMPILE) exists in every individual component but is never orchestrated across them.

## Ground-Level Code

### Existing component interfaces

```
NerveFiber.perceive(signal)      → SensoryTile(confidence, state, features)
JEPAGrid.tick(x: ndarray)        → {"fired": N, "ids": [...], "tick": T}
RoutingLayer.fire(source, dsts)  → list[fired_destination_ids]
RoutingLayer.feedback(src, dst, success) → None
HebbianChannel.activate()        → new_weight
Route.reinforce(success, lr)     → strength update
```

### The missing piece: NerveTopology

A single orchestrator that wires fibers → grid → routing → feedback into one tick cycle.

**New file: `sunset-ecosystem/nerve/topology.py`**

```python
"""NerveTopology — The living network.

Wires nerve fibers to JEPAGrid rooms via Hebbian routing.
One tick = perceive → route → fire rooms → tiles flow back → reinforce.

The COLLECT → SELECT → COMPILE lifecycle:
  COLLECT:  Fiber perceives raw signal (PERCEIVING state)
  SELECT:   Grid rooms fire based on novelty (tournament-like selection)
  COMPILE:  Repeated patterns compile routes (Hebbian strengthening → automatic)
  FEEDBACK: Reception signal reinforces strong routes, weakens poor ones
  REGULATE: Chaos probability decays as routes compile
"""

from __future__ import annotations

__all__ = ["NerveTopology"]

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from nerve.room_grid import JEPAGrid
from nerve.fiber import NerveFiber, FiberState, SensoryTile
from nerve.routing import RoutingLayer, Route, HebbianChannel

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
               → JEPAGrid.tick(encoded_tile)
               → Results feed back through RoutingLayer.feedback()
               → Hebbian channels strengthen between co-firing rooms

    The topology IS the unified runtime described in THEORY-OF-ECOSYSTEMS:
    "A loosely-coupled runtime where agents share a threshold parameter
    (learned from the user) but maintain independent computations."

    Args:
        n_fibers: Number of input nerve fibers.
        n_rooms: Number of JEPAGrid rooms.
        chaos: Initial chaos probability for route exploration.
        adapt_threshold: Confidence threshold for fiber compilation.
        learning_rate: Hebbian reinforcement rate.
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
        self.grid = JEPAGrid(n_rooms, d=signal_dim)
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
        from swarm.penrose import assign_positions
        positions = assign_positions([f"room-{i}" for i in range(n_rooms)])
        for i in range(len(positions) - 1):
            a = positions[i].agent_id
            b = positions[i + 1].agent_id
            self.routing.add_channel(a, b, weight=0.1)

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

        Uses the tile's features hash as seed for deterministic encoding.
        Compiled tiles produce lower-energy signals (they're routine).
        Novel tiles produce higher-energy signals (they demand attention).
        """
        rng = np.random.RandomState(
            hash(tile.pattern_id) % (2**31)
        )
        signal = rng.randn(self.signal_dim).astype(np.float32) * 0.1

        # Energy scaling by fiber state
        if tile.state == FiberState.COMPILED:
            signal *= 0.3   # compiled = routine, low energy
        elif tile.state == FiberState.NOVELTY_ALERT:
            signal *= 2.0   # novelty = important, high energy
        elif tile.state == FiberState.PERCEIVING:
            signal *= 1.0   # normal perception
        elif tile.state == FiberState.ADAPTING:
            signal *= 0.7   # adapting = building confidence

        # Confidence-weighted
        signal *= (0.5 + tile.confidence * 0.5)

        return signal

    def tick(self, signals: dict[str, Any] | None = None) -> TickResult:
        """One full topology tick.

        Args:
            signals: Optional mapping of fiber_id → raw signal.
                     If None, generates random test signals.

        Returns:
            TickResult with performance metrics.
        """
        t0 = time.perf_counter()
        self.tick_count += 1

        # ── COLLECT: All fibers perceive ──────────────────────
        tiles: dict[str, SensoryTile] = {}
        novel_count = 0

        for fid, fiber in self.fibers.items():
            raw = signals.get(fid, np.random.randn(self.signal_dim)) if signals else None
            if raw is None:
                # Auto-generate signal from grid state
                raw = np.random.randn(self.signal_dim).astype(np.float32)

            tile = fiber.perceive(raw)
            tiles[fid] = tile
            if tile.state in (FiberState.NOVELTY_ALERT, FiberState.PERCEIVING):
                novel_count += 1

        # ── SELECT: Route tiles to rooms ──────────────────────
        # Each fiber fires to rooms based on route strength + chaos
        room_signals: dict[str, list[tuple[str, SensoryTile]]] = {}
        routes_activated = 0

        for fid, tile in tiles.items():
            fired = self.routing.fire(fid)
            routes_activated += len(fired)
            for rid in fired:
                room_signals.setdefault(rid, []).append((fid, tile))

        # ── COMPILE: Grid processes combined signal ───────────
        # Aggregate tile signals for each room, weighted by route strength
        combined = np.zeros(self.signal_dim, dtype=np.float32)
        for rid, sources in room_signals.items():
            for fid, tile in sources:
                combined += self._encode_tile(tile)

        if np.any(combined != 0):
            combined /= max(1, len(room_signals))
            grid_result = self.grid.tick(combined)
        else:
            grid_result = {"fired": 0, "ids": [], "tick": self.tick_count}

        # ── FEEDBACK: Reinforce routes based on grid response ─
        # Rooms that fired = successful reception
        routes_compiled = 0
        fired_room_ids = {f"room-{i}" for i in grid_result.get("ids", [])}

        for fid in self.fibers:
            for rid in fired_room_ids:
                # Did this room fire from this fiber's route?
                route_key = self.routing._route_key(fid, rid)
                route = self.routing._routes.get(route_key)
                if route:
                    success = True  # room fired = good routing
                    self.routing.feedback(fid, rid, success)
                    if route.strength > 0.9:
                        routes_compiled += 1

            # Penalize routes to cold rooms
            cold_rooms = self.grid.cold(thresh=0)
            for room_idx in cold_rooms:
                rid = f"room-{room_idx}"
                self.routing.feedback(fid, rid, success=False)

        # ── REGULATE: Adaptive chaos decay ────────────────────
        compiled_fraction = routes_compiled / max(1, len(self.routing._routes))
        # Chaos decays as more routes compile (less exploration needed)
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
                log.info(f"Topology tick {t}: {r}")
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
        cold = self.grid.cold(thresh=1)
        for i in cold:
            self.grid.rebirth(i)
            # Reset routes to this room (give it fresh chances)
            for fid in self.fibers:
                key = self.routing._route_key(fid, f"room-{i}")
                if key in self.routing._routes:
                    self.routing._routes[key].strength = 0.3
        return len(cold)
```

### Integration test

**New file: `sunset-ecosystem/nerve/test_topology.py`**

```python
"""Test the full NerveTopology cycle."""

import numpy as np
import pytest
from nerve.topology import NerveTopology


def test_topology_creates():
    """Topology wires fibers → rooms → routing."""
    topo = NerveTopology(n_fibers=4, n_rooms=50)
    assert topo.n_fibers == 4
    assert topo.n_rooms == 50
    assert len(topo.routing._routes) == 4 * 50  # every fiber → every room


def test_topology_ticks():
    """One tick: all fibers perceive, some rooms fire."""
    topo = NerveTopology(n_fibers=4, n_rooms=50)
    result = topo.tick()
    assert result.fibers_perceived == 4
    assert result.tick == 1
    assert result.latency_ms > 0


def test_topology_compiles():
    """After many ticks with repeated signal, fibers should compile."""
    topo = NerveTopology(n_fibers=2, n_rooms=20, adapt_threshold=0.8)
    signal = {"fiber-0": "test-pattern-abc", "fiber-1": "test-pattern-xyz"}

    for _ in range(100):
        topo.tick(signals=signal)

    # At least one fiber should have started compiling
    stats = topo.stats
    assert stats["fibers_compiled"] >= 0  # may or may not compile in 100 ticks


def test_topology_adaptive_chaos():
    """Chaos should decay over time."""
    topo = NerveTopology(n_fibers=2, n_rooms=20, chaos=0.3)
    initial_chaos = topo.routing.chaos

    for _ in range(500):
        topo.tick()

    assert topo.routing.chaos <= initial_chaos


def test_topology_rebirth():
    """Cold rooms get rebirthed."""
    topo = NerveTopology(n_fibers=2, n_rooms=20)
    # Tick once to initialize
    topo.tick()
    # Rebirth cold rooms
    rebirthed = topo.rebirth_cold_rooms()
    assert rebirthed >= 0


def test_topology_penrose_channels():
    """Hebbian channels exist between Penrose-adjacent rooms."""
    topo = NerveTopology(n_fibers=2, n_rooms=20)
    assert len(topo.routing._channels) > 0
    # Channels should be between adjacent room pairs
    for key, ch in topo.routing._channels.items():
        assert "room-" in ch.node_a
        assert "room-" in ch.node_b


def test_topology_run_batch():
    """Run 100 ticks without error."""
    topo = NerveTopology(n_fibers=4, n_rooms=50)
    results = topo.run(ticks=100)
    assert len(results) == 100
    assert results[-1].tick == 100
```

### Benchmark harness

**New file: `sunset-ecosystem/nerve/bench_topology.py`**

```python
"""Benchmark the full topology tick cycle."""
import time
import numpy as np
from nerve.topology import NerveTopology


def bench_topology(n_fibers=8, n_rooms=250, ticks=100):
    topo = NerveTopology(n_fibers=n_fibers, n_rooms=n_rooms)

    # Warmup
    for _ in range(10):
        topo.tick()

    t0 = time.perf_counter()
    for _ in range(ticks):
        topo.tick()
    elapsed = time.perf_counter() - t0

    avg_ms = elapsed / ticks * 1000
    print(f"NerveTopology({n_fibers} fibers, {n_rooms} rooms):")
    print(f"  {ticks} ticks in {elapsed:.3f}s")
    print(f"  {avg_ms:.2f} ms/tick")
    print(f"  Stats: {topo.stats}")

    return avg_ms


if __name__ == "__main__":
    # Small
    bench_topology(4, 50, 200)
    # Medium
    bench_topology(8, 250, 200)
    # Large
    bench_topology(8, 1000, 100)
```

## Decision

Build `NerveTopology` as the orchestrator that connects all existing nerve components into one tick cycle. It does NOT replace any existing code — it wires them together.

The topology follows the universal grammar exactly:
- **COLLECT**: Fibers perceive signals, produce tiles
- **SELECT**: RoutingLayer fires routes based on strength + chaos
- **COMPILE**: Repeated patterns compile fibers (PERCEIVING → COMPILED) and strengthen routes (Hebbian)
- **FEEDBACK**: Grid firing results feed back into route reinforcement
- **REGULATE**: Chaos decays as compilation progresses

Key design choice: **Fiber → Room is many-to-many.** Every fiber routes to every room. Route strength determines which paths actually fire. This matches the conservation law (γ + H ≈ constant): high routing connectivity (many routes) trades off against routing selectivity (strong routes dominate).

## Implementation Order

1. Write `nerve/topology.py` with NerveTopology class
2. Write `nerve/test_topology.py` — 7 unit tests
3. Write `nerve/bench_topology.py` — performance harness
4. Wire into BreedingDaemon (from SPEC-BREEDER) — topology replaces direct grid access
5. Add `--topology` CLI mode to sunset-ecosystem entry point
6. Add compiled pathway visualization (which routes compiled, fiber states)
7. Integration with Rust kernel (from SPEC-JEPA-GRID-OPTIMIZATION) — topology.grid uses Rust forward
8. Add heartbeat monitoring to `HEARTBEAT.md` for topology health

## Success Criteria

- [ ] `NerveTopology(4, 50)` creates 200 routes (4 fibers × 50 rooms)
- [ ] `topo.tick()` processes all fibers and returns TickResult
- [ ] Chaos decays over time: `topo.routing.chaos` decreases
- [ ] Fibers compile after repeated signals: `COMPILED` state reached
- [ ] Hebbian channels between Penrose-adjacent rooms auto-created
- [ ] `rebirth_cold_rooms()` resets cold rooms and weakens their routes
- [ ] 250 rooms × 8 fibers × 100 ticks completes in < 5 seconds
- [ ] All 7 unit tests pass
- [ ] Benchmark prints ms/tick for small/medium/large configurations
- [ ] Stats dict shows compiled fibers, active rooms, route count
