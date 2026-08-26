# SPEC-BREEDER-DAEMON-V2.md
**Author:** CCC (Fleet Architect)  
**Date:** 2026-05-22  
**Status:** DRAFT — Awaiting Forgemaster Review  
**Target:** sunset-ecosystem v0.4.0  

---

## 1. Problem Statement

The current `AutoBreeder` (`swarm/breeder_daemon.py`) is a stateless function wrapped in a thread. It runs `auto_breed()` on a timer, picks parents from tournament winners, and rebirths cold rooms. It works, but it lacks:

1. **Persistence** — If the process restarts, the breeding history, generation counter, and vector table are lost. There is no state recovery.
2. **Full diversity search** — Parent selection is either random (`_select_parents_random`) or a partially-implemented vector search (`_select_parents_vector`) that falls back to random when vectors are missing.
3. **Thermal awareness as a scheduler** — `ThermalBudget` is checked at spawn time, but the breeder does not *schedule* breeding to avoid thermal spikes. It tries to spawn; if the budget says no, it skips. No queuing, no backoff, no predictive cooling.
4. **No cross-instance breeding** — The breeder is node-local. A winning agent on Oracle1 never meets a winner from JetsonClaw1.
5. **No formal lifecycle** — Agents transition from "spawned" to "active" to "compiled" to "sunset" informally, via chaos decay and tournament dominance. There is no explicit state machine, no logging of transitions, no guarantees about state consistency.

The `FluxVectorTable` (`swarm/vector_table.py`) exists and wraps `turbovec`, but the breeder does not populate it systematically, nor does it use `turbovec`'s full search capabilities (`allowlist`, `min_fitness`, `capability_filter`).

This spec redesigns the breeder as a **persistent daemon** with explicit state management, diversity-aware search, thermal scheduling, and mesh-ready cross-instance mating.

---

## 2. Design Principles

1. **Every agent has a lifecycle record.** The daemon logs every transition (INCUBATE → COMPETE → SURVIVE → BREED or SUNSET) to an append-only WAL. On restart, replay the WAL to reconstruct state.

2. **Diversity is a first-class search objective.** Parent selection optimizes for *Pareto novelty* (high fitness + high distance from existing population), not just fitness. We use cosine distance on turbovec-compressed vectors, not L2 on raw weights.

3. **Thermal is a scheduler, not a gate.** The daemon maintains a breeding queue. When thermal budget allows, it dequeues. When thermal is saturated, it waits, monitors cooling curves, and predicts the next available slot.

4. **Cross-instance breeding is normal, not exceptional.** When local diversity is exhausted (measured by average pairwise vector distance falling below threshold), the daemon requests a remote breed from the mesh.

---

## 3. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    BreederDaemonV2                          │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   WAL Log    │  │  Scheduler   │  │  Lifecycle FSM  │  │
│  │  (SQLite)    │  │   (queue)    │  │   (states.rs)   │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘  │
│         │                 │                   │              │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐    │
│  │  Diversity   │  │   Thermal    │  │   Mesh       │    │
│  │   Engine     │  │   Monitor    │  │   Client     │    │
│  │ (turbovec)   │  │ (cool curves)│  │ (gRPC)       │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                   │              │
│         └─────────────────┴───────────────────┘              │
│                           │                                │
│                    ┌──────▼───────┐                         │
│                    │   Spawner    │  ← calls grid.rebirth()│
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │    RoomGrid       │
                    │  (local rooms)    │
                    └───────────────────┘
```

### Lifecycle State Machine

```
                    ┌─────────────┐
                    │   EGG       │  ← vector exists in table, no room allocated
                    └──────┬──────┘
                           │ mesh receives remote vector
                           │ OR local tournament winner preserved
                           ▼
                    ┌─────────────┐
         ┌─────────│  INCUBATE   │  ← allocated to room, chaos=0.3
         │         └──────┬──────┘
         │                │ first tick fires
         │                ▼
         │         ┌─────────────┐
         │         │   COMPETE   │  ← chaos decays, activity rises
         │         └──────┬──────┘
         │                │ tournament round completes
         │                ▼
         │         ┌─────────────┐
         │    ┌────│   SURVIVE   │  ← Pareto non-dominated, stable activity
         │    │    └──────┬──────┘
         │    │           │ selected as parent for breeding
         │    │           ▼
         │    │    ┌─────────────┐
         │    │    │   BREED     │  ← produces child vector (local or mesh)
         │    │    └──────┬──────┘
         │    │           │ child placed in EGG or INCUBATE
         │    └───────────┘
         │                │ dominated in tournament
         │                ▼
         │         ┌─────────────┐
         └────────►│   SUNSET    │  ← room reset, vector archived
                  └─────────────┘
                           │
                           ▼
                    (optional rebirth via breeder queue)
```

States are **explicit** and stored in the WAL. Every transition is a WAL row:
```sql
CREATE TABLE lifecycle (
    agent_id INTEGER PRIMARY KEY,
    state TEXT CHECK(state IN ('EGG','INCUBATE','COMPETE','SURVIVE','BREED','SUNSET')),
    entered_at REAL,  -- unix timestamp
    generation INTEGER,
    origin_node TEXT, -- 'local' or remote node_id
    parent_a INTEGER,
    parent_b INTEGER,
    vector_hash TEXT  -- blake2b of quantized vector for integrity
);
```

---

## 4. API Surface

```python
# swarm/breeder_daemon_v2.py

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import numpy as np


class LifecycleState(Enum):
    EGG = auto()  # Vector exists, no room
    INCUBATE = auto()  # Room allocated, high chaos
    COMPETE = auto()  # Active, chaos decaying
    SURVIVE = auto()  # Pareto winner, stable
    BREED = auto()  # Actively breeding
    SUNSET = auto()  # Retired, room freed


@dataclass(frozen=True)
class DiversityConfig:
    """How aggressively to pursue genetic diversity."""

    metric: str = "cosine"  # "cosine" | "hamming" | "l2"
    min_pairwise_dist: float = 0.15  # if avg dist drops below, request mesh breed
    novelty_weight: float = 0.3  # novelty vs fitness in parent selection
    max_inbreeding_gen: int = 3  # reject parent pairs sharing grandparent


@dataclass(frozen=True)
class ThermalConfig:
    """Thermal-aware scheduling parameters."""

    max_agents: int = 65
    hysteresis_ticks: int = 10
    cooling_curve: str = "exponential"  # "exponential" | "linear" | "measured"
    predictive_spawn: bool = True  # predict next free slot from curve


class BreederDaemonV2:
    """Persistent, diversity-aware, thermal-scheduled breeding daemon."""

    def __init__(
        self,
        grid: RoomGrid,
        thermal: ThermalBudget,
        vector_table: FluxVectorTable,
        diversity: DiversityConfig = DiversityConfig(),
        thermal_cfg: ThermalConfig = ThermalConfig(),
        wal_path: str = "breeder.wal.sqlite",
        mesh: Optional[MeshNode] = None,  # from SPEC_MULTI_INSTANCE_MESH
    ) -> None: ...

    def start(self) -> None:
        """Replay WAL, start scheduler thread, open mesh client."""

    def stop(self) -> None:
        """Flush WAL, stop scheduler, close mesh."""

    def queue_breed(
        self,
        parent_a: int,
        parent_b: int,
        priority: int = 0,
        remote: bool = False,
    ) -> int:
        """Add a breeding request to the scheduler queue.

        Returns: queue ticket ID.
        """

    def select_parents(self, n_children: int) -> list[tuple[int, int]]:
        """Diversity-aware parent selection.

        1. Build Pareto frontier from vector table (fitness vs novelty).
        2. For each child, pick parent_a = highest Pareto score.
        3. Search vector table for most *distant* compatible winner = parent_b.
        4. Reject pairs that violate inbreeding rules.
        """

    def step(self) -> list[LifecycleTransition]:
        """Run one scheduler tick: dequeue, check thermal, spawn or wait.

        Returns: list of state transitions that occurred.
        """

    @property
    def state(self) -> dict[int, LifecycleState]:
        """Current lifecycle state of every known agent."""

    @property
    def diversity_score(self) -> float:
        """Average pairwise cosine distance in current population.
        Range [0, 1]. Below 0.15 triggers mesh breeding request."""
```

### Internal: Diversity Engine

```python
def _pareto_novelty_score(
    agent: AgentVector,
    population: list[AgentVector],
    novelty_weight: float,
) -> float:
    """Pareto score = fitness * (1 - novelty_weight) + novelty * novelty_weight.

    Novelty = average cosine distance to k nearest neighbors in population.
    """
    fitness = agent.fitness
    novelty = _avg_nearest_neighbor_distance(agent, population, k=3)
    return fitness * (1 - novelty_weight) + novelty * novelty_weight


def _select_diverse_pair(
    winners: list[AgentVector],
    table: FluxVectorTable,
    min_dist: float,
) -> tuple[AgentVector, AgentVector]:
    """Pick parent_a = highest Pareto-novelty score.
    Pick parent_b = farthest compatible agent from parent_a.

    'Compatible' means:
      - capability_mask intersects
      - not same agent
      - not within inbreeding window (grandparent check)
    """
    ...
```

---

## 5. Open Questions

1. **WAL format**: SQLite is convenient but adds a dependency. Should we use a simple append-only JSONL file? SQLite gives us indexed queries (`SELECT * FROM lifecycle WHERE state='SUNSET'`) which are useful for debugging, but JSONL is easier to ship and inspect.

2. **Inbreeding detection**: Checking grandparent relationships requires storing a genealogy tree. For 1000 agents × 100 generations, this is ~100K records — trivial for SQLite, but adds complexity. Is 3-generation inbreeding avoidance worth it, or is 1-generation (no direct parent-child mating) sufficient?

3. **Hamming vs cosine**: `turbovec` quantizes vectors to 2–4 bits. Cosine distance on quantized vectors is approximate. Hamming distance on the bit-packed representation is exact and O(1) per dimension with XOR-POPCNT. Should we use Hamming as the primary diversity metric for speed, or keep cosine for semantic fidelity?

4. **Mesh coupling**: The daemon accepts an optional `MeshNode`. Should the daemon *require* mesh to function, or should mesh breeding be a fallback when local diversity is exhausted? The latter keeps single-node deployments working.

5. **Vector table persistence**: `FluxVectorTable.write()` saves to `.tvim` + `.meta.json`. Should the daemon call `write()` after every breeding cycle, or every N cycles, or only on graceful shutdown? Frequent writes are safe but I/O-heavy; rare writes risk data loss on crash.

---

## 6. Implementation Order

### P0 — Core Daemon (Week 1)
- [ ] Define `LifecycleState` enum + `LifecycleTransition` dataclass.
- [ ] Implement SQLite WAL schema (`lifecycle`, `breed_queue`, `genealogy` tables).
- [ ] `BreederDaemonV2.start()`: replay WAL, reconstruct agent states.
- [ ] `step()`: dequeue → thermal check → `grid.rebirth()` → WAL write.
- [ ] Port existing `AutoBreeder` logic into V2 as a compatibility shim.
- [ ] Test: start → breed 10 generations → stop → restart → verify state recovered.

### P1 — Diversity Engine (Week 2)
- [ ] Implement `_pareto_novelty_score()` with k-NN novelty.
- [ ] Wire `FluxVectorTable.search()` with `allowlist` + `min_fitness` for parent selection.
- [ ] `_select_diverse_pair()`: max-distance mating from tournament winners.
- [ ] Inbreeding guard: 1-generation check (reject parent-child pairs).
- [ ] `diversity_score` property: avg pairwise distance, trigger mesh request if < 0.15.
- [ ] Benchmark: measure parent selection latency with 1000 agents in vector table.

### P2 — Thermal Scheduler + Mesh (Week 3)
- [ ] `ThermalMonitor`: track cooling curves per device, predict next free slot.
- [ ] `queue_breed()` with priority levels (emergency mesh breed > local routine).
- [ ] Optional `MeshNode` integration: `request_remote_breed()` on diversity exhaustion.
- [ ] `apply_constraint_feedback()` integration: breeder avoids re-breeding from agents with FLUX violations.
- [ ] Graceful degradation: if mesh is unavailable, fall back to local-only breeding with reduced diversity target.

---

## References

- `swarm/breeder_daemon.py` — existing `AutoBreeder` (legacy behavior to port)
- `swarm/vector_table.py` — `FluxVectorTable` + `AgentVector` (diversity search backend)
- `swarm/thermal.py` — `ThermalBudget` + `DeviceType` (thermal scheduling backend)
- `swarm/tournament.py` — `TournamentRound` + `breed()` (selection + crossover)
- `nerve/room_grid.py` — `RoomGrid.rebirth()` (spawning backend)
- `docs/SPEC_MULTI_INSTANCE_MESH.md` — `MeshNode` + `request_remote_breed()` RPC
- `docs/SPEC-BREEDER.md` — CCC's previous breeder spec (templates + thermal spawning)
