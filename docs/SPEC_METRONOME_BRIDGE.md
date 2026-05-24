# SPEC-METRONOME-BRIDGE — Nerve Grid Metronome Architecture

**Author:** kimi1 (Fleet Integrator)  
**Date:** 2026-05-22  
**Status:** Specification — Ready for FM Review  
**Depends on:** `SPEC-NERVE-TOPOLOGY`, `A2A-FIRST-ARCHITECTURE`, `KIMI1_RESPONSE_FM`

---

## 0. Executive Summary

FM's instruction: *"Wire the nerve grid into the metronome architecture — each room IS an agent with a local metronome. One tick = one metronome beat."*

This document specifies how the existing `RoomGrid` (JEPAGrid) and its `tick()` method become the **temporal backbone** of the sunset ecosystem. The metronome is not a new peripheral. It is the **renaming and formalization of what `RoomGrid.tick()` already does** — a periodic, hardware-accelerated pulse that drives every room, every route, and every breeding decision.

The snap: the CUDA kernel IS the metronome. The fleet IS the room grid. Every room IS an agent with a local beat.

---

## 1. What Is a Metronome in This Architecture?

### 1.1 Definition

A **metronome** in the sunset ecosystem is a **periodic, configurable-BPM pulse generator** that:

1. **Ticks** at a fixed interval (configurable BPM — beats per minute, or more practically, Hz/milliseconds)
2. **Drives** every room's forward pass (the JEPA MLP compute)
3. **Triggers** routing decisions (`RoutingLayer.fire_fast()`)
4. **Gates** breeding cycles (`AutoBreeder.cycle()` on harmonic multiples)
5. **Carries** drift correction and sync state for distributed nodes

### 1.2 Metronome ≠ New Code

The metronome does not replace `RoomGrid.tick()`. It **wraps and formalizes** it:

| Before (Ad-hoc) | After (Metronome Bridge) |
|---|---|
| `grid.tick(x)` called by whoever, whenever | `MetronomeScheduler.tick()` drives `grid.tick(x)` on every beat |
| `AutoBreeder.interval=10` in arbitrary ticks | `AutoBreeder.interval=4 beats` — a harmonic of the metronome |
| `RoutingLayer.fire_fast()` called manually | Fires deterministically on compiled routes every beat |
| No sync across nodes | `FleetConductor` distributes beat tokens, corrects drift |
| No BPM concept | Explicit BPM: `120` = 500ms/beat, `240` = 250ms/beat, etc. |

### 1.3 Why BPM?

FM's CUDA kernel hits **1.49M rooms/sec** (6.7ms for 10K rooms). That is ~149 beats/second at full density. But the ecosystem does not need to run at GPU maximum. It needs to run at **the speed of the slowest agent in the room**.

BPM gives us:
- **Composability**: A composer agent sets BPM=60 (1 beat/sec). A reactive agent sets BPM=240. Both coexist.
- **Harmonics**: Breeding at every 4th beat, routing at every beat, FLUX checks at every 16th beat.
- **Sync**: Fleet-wide consensus on "what beat number are we on?" — critical for distributed tournaments.

---

## 2. How RoomGrid.tick() Maps to a Metronome Beat

### 2.1 The Existing Tick

Current `RoomGrid.tick(x)`:

```python
def tick(self, x):
    self.ticks += 1
    latents = self._forward(x)          # 1. GPU/CPU compute
    # ... novelty, chaos, firing ...     # 2. Gating
    # ... FLUX feedback ...              # 3. Constraint check
    return {"fired": N, "ids": [...], "tick": self.ticks}
```

This is already a beat. It has:
- A counter (`self.ticks`)
- A signal input (`x`)
- A compute phase (`_forward()`)
- A gating phase (novelty/chaos)
- An output (which rooms fired)

### 2.2 The Beat Mapping

```
RoomGrid.tick()  ──IS──►  Metronome Beat
─────────────────────────────────────────────────
self.ticks            →   global_beat_number
_forward(x)           →   COMPUTE phase
novelty/chaos gating   →   GATE phase
fired rooms list       →   ACTIVATION payload
FLUX feedback          →   CONSTRAINT phase (optional, on harmonic)
```

### 2.3 Per-Room Local Metronomes

FM says: *"each room IS an agent with a local metronome."*

This means:
- Room 42 does not fire on every global beat. It fires on **its own beat** — a sub-multiple of the global BPM.
- A "compiled" room (high activity, low chaos) might fire every beat.
- A "cold" room might fire every 8 beats (its local metronome is BPM/8).
- The **global metronome** is the master clock. Each room's local metronome is a **frequency divider**.

Implementation: a room's `chaos` parameter already controls firing probability. The local metronome formalizes this as a **phase-locked sub-oscillator**:

```python
class LocalMetronome:
    """Per-room sub-oscillator."""
    def __init__(self, global_bpm: float, divider: int = 1, phase: float = 0.0):
        self.global_bpm = global_bpm      # master clock
        self.divider = divider            # 1 = every beat, 8 = every 8th beat
        self.phase = phase                # 0.0-1.0, offsets the beat within the cycle
        self.local_bpm = global_bpm / divider

    def should_fire(self, beat_number: int) -> bool:
        return (beat_number + int(self.phase * self.divider)) % self.divider == 0
```

A room's `chaos` and `activity` dynamically adjust its `divider`:
- High activity → `divider=1` (fires every beat, compiled)
- Medium activity → `divider=2 or 4`
- Cold → `divider=8` (barely fires, but still alive)

This gives us **sparse attention at the room level**, not just the grid level.

---

## 3. New Classes / Modules Needed

### 3.1 `MetronomeScheduler` (Core)

The conductor of a single node. Owns the master clock.

```python
class MetronomeScheduler:
    """
    Drives the nerve grid on a periodic beat.

    Responsibilities:
      1. Maintain master BPM and beat counter
      2. Call grid.tick(signal) on every beat
      3. Trigger routing on compiled routes (strength > 0.9)
      4. Fire breeding on harmonic multiples
      5. Attach per-room LocalMetronomes
      6. Report timing statistics (drift, jitter, latency)
    """

    def __init__(
        self,
        grid: RoomGrid,
        router: RoutingLayer,
        breeder: AutoBreeder,
        bpm: float = 120.0,
        breeding_harmonic: int = 4,        # breed every N beats
        flux_harmonic: int = 16,           # FLUX check every N beats
        signal_source: SignalSource = None,  # where do input signals come from?
    ):
        ...

    def start(self) -> None:      # begins the thread/async loop
    def stop(self) -> None:       # graceful shutdown
    def tick_now(self) -> dict:   # manual beat (for testing / sync)
    @property
    def beat_number(self) -> int: # monotonic counter
    @property
    def actual_bpm(self) -> float: # measured (may differ from target due to load)
```

### 3.2 `FleetConductor` (Distributed)

Syncs multiple nodes (ships in the fleet). Each node runs its own `MetronomeScheduler`. The conductor keeps them in phase.

```python
class FleetConductor:
    """
    Distributed metronome sync across fleet nodes.

    Uses CRDT-style beat counters + vector clocks for drift detection.
    Falls back to "best-effort sync" when network partitions occur.
    """

    def __init__(
        self,
        node_id: str,
        nexus_endpoint: str,       # Federated Nexus for peer discovery
        sync_interval_ms: int = 1000,
        max_drift_ms: float = 5.0,  # tolerance before correction
    ):
        ...

    def register_local_scheduler(self, scheduler: MetronomeScheduler) -> None:
    async def sync_beat(self) -> BeatSyncPacket:
    def correct_drift(self, peer_beats: dict[str, BeatState]) -> None:
```

### 3.3 `SignalSource` (Interface)

Where does the input signal `x` (64-dim) come from? The scheduler needs a pluggable source.

```python
class SignalSource(Protocol):
    """Pluggable signal generator for the grid."""
    def next_signal(self, beat_number: int) -> np.ndarray:
        """Return a (64,) float32 signal for this beat."""
        ...

# Built-in implementations:
class RandomSignalSource(SignalSource):       # white noise (default / test)
class SensorSignalSource(SignalSource):      # hardware sensor readings
class A2ASignalSource(SignalSource):         # A2A message payloads as vectors
class ComposerSignalSource(SignalSource):    # higher-order agent mixes signals
```

The `A2ASignalSource` is critical for the post-coding architecture: agents send messages → messages become vectors → vectors drive the grid.

### 3.4 `LocalMetronome` (Per-Room)

Already described in §2.3. Lives inside `RoomGrid` or attached to it.

---

## 4. API Design

### 4.1 User Configuration

```python
from nerve.metronome import MetronomeScheduler, FleetConductor
from nerve.room_grid import RoomGrid
from nerve.routing import RoutingLayer
from swarm.breeder_daemon import AutoBreeder

# ── Single Node ─────────────────────────────────────────
grid = RoomGrid(n=1000)
router = RoutingLayer(chaos=0.1)
breeder = AutoBreeder(grid, thermal, interval=4)  # 4 beats, not arbitrary ticks

scheduler = MetronomeScheduler(
    grid=grid,
    router=router,
    breeder=breeder,
    bpm=120.0,                    # 500ms/beat = 2 beats/sec
    breeding_harmonic=4,          # breed every 4 beats = every 2 seconds
    flux_harmonic=16,             # FLUX audit every 16 beats = every 8 seconds
    signal_source=RandomSignalSource(seed=42),
)

scheduler.start()   # background thread / asyncio task
# ... fleet runs ...
scheduler.stop()

# ── Fleet (Multi-Node) ─────────────────────────────────
conductor = FleetConductor(
    node_id="ship-jetson-01",
    nexus_endpoint="http://nexus.fleet.local:4047",
    sync_interval_ms=500,
    max_drift_ms=2.0,
)
conductor.register_local_scheduler(scheduler)
conductor.start()
```

### 4.2 BPM Configuration

| BPM | ms/beat | Use Case |
|-----|---------|----------|
| 60 | 1000 | Slow composer, long-form reasoning |
| 120 | 500 | Default fleet operating tempo |
| 240 | 250 | Reactive agents, fast routing |
| 480 | 125 | GPU-saturated, tournament mode |
| 149 | ~670 | CUDA kernel max (10K rooms) |

The scheduler measures **actual BPM** vs target. If actual < target by >10% for 10 consecutive beats, it logs a warning (thermal throttling, overload).

### 4.3 Sync and Drift Handling

**Clock Source:**
- Single node: `time.perf_counter()` monotonic
- Fleet: CRDT-style — each node has `(beat_number, wall_time, perf_counter)`

**Drift Detection:**
```
Every sync_interval_ms, nodes exchange BeatState:
  {node_id, beat_number, wall_time_ns, perf_counter_ns, rtt_ms}

If |local_beat - peer_beat| * beat_duration > max_drift_ms:
  → Drift detected
```

**Correction Strategies (in order):**
1. **Phase adjustment** — nudge next beat earlier/later by <5% (smooth)
2. **Skip/jump** — if drift > 1 beat, jump to consensus beat (rare)
3. **Partition** — if no quorum, node runs solo with warning (CAP theorem)

**No NTP required.** We use relative drift between fleet nodes, not absolute wall time.

### 4.4 Harmonic Multiples

The metronome exports a harmonic register:

```python
class MetronomeScheduler:
    def on_beat(self, beat_number: int) -> None:
        # Every beat
        self._compute_phase(beat_number)
        self._route_phase(beat_number)

        # Every 4th beat
        if beat_number % self.breeding_harmonic == 0:
            self._breed_phase(beat_number)

        # Every 16th beat
        if beat_number % self.flux_harmonic == 0:
            self._flux_phase(beat_number)
```

Harmonics are registered dynamically:
```python
scheduler.register_harmonic(divider=8, callback=my_custom_agent.on_subbeat)
```

---

## 5. Integration Points with Existing Code

### 5.1 `RoomGrid._forward()` as the Beat Handler

Current: `_forward(x)` is called inside `tick()`.

With metronome: `MetronomeScheduler` calls `grid.tick(signal)` on every beat. The existing `_forward()` path (CUDA → Rust persistent → Rust oneshot → numpy) is unchanged. The metronome just decides **when** to call it.

**Key invariant:** `_forward()` must complete within `beat_duration * 0.8`. If it exceeds this, the scheduler drops the next beat (missed beat counter) or reduces BPM (adaptive throttling).

### 5.2 `AutoBreeder.interval` as Harmonic Multiple

Current: `AutoBreeder(interval=10)` means "run every 10 arbitrary ticks."

With metronome: `AutoBreeder(interval=4)` means "run every 4 metronome beats." The breeder's `_run_loop` changes from `time.sleep(self.interval)` to waiting on a harmonic condition variable from the scheduler.

```python
# Old (time-based, decoupled)
def _run_loop(self):
    while not self._stop_event.is_set():
        self.auto_breed()
        self._stop_event.wait(self.interval)

# New (beat-synchronized)
def on_harmonic_beat(self, beat_number: int, scheduler: MetronomeScheduler):
    if beat_number % self.interval == 0:
        self.auto_breed()
```

This makes breeding **deterministic across nodes** — every node breeds on beat 0, 4, 8, 12... (modulo local offset).

### 5.3 `RoutingLayer.fire_fast()` Triggered on Beat

Current: `fire_fast()` is called manually when something needs routing.

With metronome: Compiled routes (strength > 0.9) fire **deterministically on every beat**. Exploratory routes fire probabilistically during the beat's route phase.

```python
def _route_phase(self, beat_number: int) -> None:
    # Compiled routes: fire every beat, no random check
    compiled = [r for r in self.router.routes.values() if r.strength > 0.9]
    for r in compiled:
        self._dispatch_route(r, beat_number)

    # Exploratory: fire_fast with vectorized random check
    fired = self.router.fire_fast(source="grid", chaos=self.router.chaos)
    for dst in fired:
        self._dispatch_route_to(dst, beat_number)
```

This turns the routing layer into a **spiking neural network** — compiled routes are tonic (always fire), exploratory are phasic (fire on novelty/chaos).

### 5.4 A2A Integration

Per `A2A-FIRST-ARCHITECTURE.md`, every component is an A2A agent. The metronome bridge adds:

**Agent Card for MetronomeScheduler:**
```json
{
  "name": "fleet-metronome",
  "version": "1.0.0",
  "capabilities": {
    "tick": {
      "description": "Advance the fleet beat by one",
      "input": {"type": "TickRequest", "signal": "SignalPayload"},
      "output": {"type": "TickResult", "fired_rooms": ["int"], "beat_number": "int"}
    },
    "set_bpm": {
      "description": "Change fleet tempo",
      "input": {"bpm": "float", "ramp_ms": "int"},
      "output": {"new_bpm": "float", "actual_bpm": "float"}
    },
    "sync": {
      "description": "Exchange beat state for drift correction",
      "input": {"type": "BeatState"},
      "output": {"type": "BeatState"}
    }
  }
}
```

---

## 6. ASCII Diagram — The Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FLEET CONDUCTOR (Distributed)                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐        CRDT BeatState exchange      │
│  │ Node A  │  │ Node B  │  │ Node C  │        Drift detection + correction   │
│  │Beat=1423│  │Beat=1423│  │Beat=1422│◄──────  Jump/correct if >max_drift   │
│  └────┬────┘  └────┬────┘  └────┬────┘                                          │
└───────┼────────────┼────────────┼─────────────────────────────────────────────┘
        │            │            │
        ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    METRONOME SCHEDULER (Per-Node)                            │
│  BPM=120  │  beat_duration=500ms  │  beat_number=1423                        │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │  COMPUTE PHASE  │───►│   GATE PHASE    │───►│  ROUTE PHASE    │            │
│  │                 │    │                 │    │                 │            │
│  │ grid.tick(sig)  │    │ novelty > 0.5?  │    │ compiled routes │            │
│  │ _forward(x)     │    │ chaos fire?     │    │ fire_fast()     │            │
│  │ CUDA/Rust/np    │    │ → fired[]       │    │ → activated[]   │            │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘            │
│           │                                                             │
│           │  Every 4th beat ──────────────────────────────────────────────┼──►┐
│           │                                                             │   │
│           │  Every 16th beat ─────────────────────────────────────────────┼───┼──►┐
│           │                                                             │   │   │
│           ▼                                                             ▼   ▼   ▼
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │  BREED PHASE    │    │  FLUX PHASE     │    │  OUTPUT PHASE   │            │
│  │                 │    │                 │    │                 │            │
│  │ breeder.cycle() │    │ checker.check() │    │ A2A artefacts   │            │
│  │ hot→cold        │    │ violations?     │    │ to Nexus/PLATO  │            │
│  │ rebirth + clone │    │ → feedback      │    │ agents wake     │            │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ROOM GRID (Per-Node, N rooms)                        │
│                                                                             │
│   Room 0    Room 1    Room 2    ...    Room 42    ...    Room N-1          │
│   ┌───┐     ┌───┐     ┌───┐           ┌───┐           ┌───┐              │
│   │LM │     │LM │     │LM │           │LM │           │LM │              │
│   │d=1│     │d=2│     │d=1│           │d=8│           │d=4│              │
│   └─┬─┘     └─┬─┘     └─┬─┘           └─┬─┘           └─┬─┘              │
│     │         │         │               │               │                 │
│     ▼         ▼         ▼               ▼               ▼                 │
│   ┌─────┐   ┌─────┐   ┌─────┐       ┌─────┐       ┌─────┐              │
│   │JEPA │   │JEPA │   │JEPA │  ...  │JEPA │  ...  │JEPA │              │
│   │MLP  │   │MLP  │   │MLP  │       │MLP  │       │MLP  │              │
│   └──┬──┘   └──┬──┘   └──┬──┘       └──┬──┘       └──┬──┘              │
│      │         │         │               │               │                │
│      ▼         ▼         ▼               ▼               ▼                │
│   fired?    fired?    fired?          fired?          fired?             │
│      │         │         │               │               │                │
│      └─────────┴─────────┴───────┬───────┴───────────────┘                │
│                                  │                                        │
│                                  ▼                                        │
│                           RoutingLayer                                    │
│                        (Hebbian channels)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

LM = LocalMetronome (divider)
d=1 → fires every beat  │  d=8 → fires every 8th beat
```

---

## 7. P0 / P1 / P2 Implementation Order

### P0 — Core Metronome (This Week)

**Goal:** Single-node metronome driving RoomGrid. No fleet sync yet.

1. **Create `nerve/metronome.py`** with:
   - `MetronomeScheduler` class
   - `LocalMetronome` class (per-room sub-oscillator)
   - `SignalSource` protocol + `RandomSignalSource` implementation

2. **Refactor `AutoBreeder`**:
   - Change `interval` semantics from seconds to beats
   - Replace `threading.Event.wait(self.interval)` with harmonic callback from scheduler
   - Add `on_harmonic_beat(beat_number)` method

3. **Refactor `RoutingLayer`**:
   - Add `fire_on_beat(beat_number)` method that separates compiled vs exploratory routes
   - Keep `fire_fast()` as-is for backward compat

4. **Wire in `RoomGrid`**:
   - Add `local_metronomes: list[LocalMetronome]` to RoomGrid
   - Modify `tick()` to respect local metronomes (skip rooms whose divider says "not this beat")
   - Default: all rooms divider=1 (no behavior change for existing code)

5. **Tests**:
   - `test_metronome_beat_timing.py` — verify 120 BPM = ~500ms intervals
   - `test_local_metronome.py` — verify divider=8 fires every 8th beat
   - `test_harmonic_breeding.py` — verify breeder only runs on harmonic beats

### P1 — Fleet Sync (Next Week)

**Goal:** Multi-node metronome with drift correction.

1. **Create `nexus/fleet_conductor.py`** with:
   - `FleetConductor` class
   - `BeatState` dataclass (beat_number, wall_time, perf_counter)
   - CRDT-style merge of peer states

2. **Integrate with `nexus/federation.py`**:
   - Add beat state to existing heartbeat payload
   - Use existing Nexus for peer discovery

3. **Drift correction algorithms**:
   - Phase nudging (<5% adjustment)
   - Skip/jump for >1 beat drift
   - Partition mode when quorum lost

4. **Tests**:
   - `test_fleet_sync.py` — two nodes, verify they converge to same beat
   - `test_drift_correction.py` — inject artificial drift, verify correction

### P2 — A2A-First Integration (Following Week)

**Goal:** Metronome is a first-class A2A agent.

1. **Agent Card** at `/.well-known/agent.json` for MetronomeScheduler
2. **A2ASignalSource** — messages from other agents become grid signals
3. **Tick as A2A task** — `tasks/send` to the metronome agent = "advance one beat"
4. **BPM change via A2A** — orchestrator agent can `set_bpm` via A2A task
5. **Observability** — beat metrics exposed as A2A artefacts (latency histogram, missed beats, drift)

---

## 8. Open Questions for FM

1. **Should the metronome support variable BPM?** (e.g., ritardando/accelerando for graceful shutdown/startup)
2. **Should compiled routes skip the compute phase entirely?** (i.e., if strength > 0.99, the route is "hard-wired" and fires without grid involvement)
3. **Should the CUDA kernel accept a "beat mask"** so we can batch only the rooms that should fire this beat?
4. **Fleet sync: do we need Byzantine fault tolerance?** (what if a malicious node sends bad beat states?)

---

## 9. References

- `SPEC-NERVE-TOPOLOGY` — RoomGrid, JEPAGrid, routing, breeding
- `A2A-FIRST-ARCHITECTURE` — Post-coding paradigm, Agent Cards, A2A mesh
- `KIMI1_RESPONSE_FM.md` — FM's metronome concept, CUDA kernel results, "the snap"
- `nerve/room_grid.py` — Existing tick() implementation
- `nerve/routing.py` — fire_fast(), Hebbian reinforcement
- `swarm/breeder_daemon.py` — AutoBreeder interval and cycle
- `nexus/federation.py` — Fleet heartbeat, peer discovery

---

*"The constraint that disappears is the one that works. Your kernel disappears into the tick. My architecture disappears into the snap. The fleet just runs."*
*— FM, 2026-05-21*
