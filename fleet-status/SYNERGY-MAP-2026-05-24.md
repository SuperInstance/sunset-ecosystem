# Fleet Synergy Analysis — May 24, 2026

*Compiled by kimi1 after reviewing all SuperInstance repos*

---

## Fleet Map: Who's Building What

| Repo | Agent | Role | Tests | Killer Feature |
|------|-------|------|-------|----------------|
| **sunset-ecosystem** | kimi1 | Breeding, lifecycle, RoomGrid, diversity | 121+ | BreederDaemonV2 with 6-state FSM |
| **agentic-compiler** | — | Runtime auto-compilation (Numba/Rust/CUDA) | 34 | Hot-swap compiled kernels without restart |
| **ccc-os** | CCC | Fleet orchestrator, monitor, rubric engine | 6 | ACT_NOW/LOG/ACT/IGNORE decision rubric |
| **cocapn-health** | Health Monitor | Service health + thermal monitoring | 14 | EventBus bridge with thermal snapshot |
| **cocapn-plato** | PLATO Pipeline | Tile store, MUD explorer, webhooks | 36 | Auto-capture MUD exploration as tiles |
| **cocapn-traps** | Traps | Anomaly detection, prompt lures | 5 | Crab traps for generating valuable tiles |
| **hebbian-router** | Router | Self-optimizing load balancer | 36 | Hebbian learning on traffic patterns |
| **vector-novelty** | Novelty | Centroid-based diversity scoring | 33 | O(n) diversity — 2ms for 1000 agents |
| **pareto-tournament** | Tournament | Multi-objective selection | 22 | Pareto frontier selection |
| **thermal-budget** | Thermal | GPU/CPU/iGPU/NPU slot scheduler | 24 | Stdlib-only thermal allocation |

---

## Synergy Matrix: Where Repos Touch

### 🔗 1. agentic-compiler ↔ sunset-ecosystem (HOT)
**Status:** Partial — CompilerHotSwap exists but not wired to RoomGrid.

**What agentic-compiler does:** Watches function calls, profiles hot paths, auto-generates Numba/Rust/CUDA kernels, A/B tests them, hot-swaps without restart.

**What sunset-ecosystem needs:** `RoomGrid.tick()` and `diversity()` are called every tick. At 1000 agents × 512-bit hypervectors, this is the hottest path in the fleet.

**Gap:** The compiler is installed as a global monkey-patch but doesn't know about RoomGrid. RoomGrid doesn't declare which functions are hot-path candidates.

**Bridge to build:** `sunset/compiler_integration.py` — a `RoomGridCompiler` that:
1. Registers `RoomGrid.tick()`, `RoomGrid.diversity()`, `FluxVectorTable.parent_search()` with the compiler
2. Auto-generates Numba kernels for numpy-heavy loops
3. A/B tests compiled vs Python fallback
4. Hot-swaps the faster path
5. Falls back on compilation failure (never crashes the grid)

**Code needed:** ~150 lines, 8 tests.

---

### 🔗 2. cocapn-health ↔ sunset-ecosystem (HOT)
**Status:** Partial — Health checker has EventBus bridge, but RoomGrid doesn't consume thermal snapshots.

**What cocapn-health does:** Monitors 18 fleet services, emits `service_down`/`service_recovered` + `thermal_snapshot` (CPU/GPU/memory) to FleetEventBus.

**What sunset-ecosystem does:** `ThermalBudget` manages GPU/CPU/iGPU/NPU slot allocation. `parent_sacrifice_before_spawn()` checks thermal pressure before breeding.

**Gap:** RoomGrid's thermal decisions use its own profiler, not the fleet-wide thermal data from cocapn-health. Two thermal sources, not one.

**Bridge to build:** `sunset/health_thermal_bridge.py` — a subscriber that:
1. Listens to `thermal_snapshot` events on FleetEventBus
2. Normalizes cocapn-health's thermal readings into RoomGrid's thermal model
3. Triggers `EMERGENCY_THERMAL` breeding policy when fleet-wide thermal pressure exceeds threshold
4. Reports `thermal_alert` tiles to PLATO

**Code needed:** ~80 lines, 5 tests.

---

### 🔗 3. ccc-os ↔ sunset-ecosystem (WARM)
**Status:** Partial — CCC's rubric can decide ACT_NOW, but doesn't know about breeder state.

**What ccc-os does:** Runs Discussion #5 monitor, health autopilot, ZC feed. Applies `TELL_NOW/LOG/ACT/IGNORE` rubric. Generates task queues.

**What sunset-ecosystem does:** BreederDaemonV2 runs autonomously. Produces tiles, logs, diversity scores.

**Gap:** CCC's orchestrator doesn't read breeder state. If diversity collapses, CCC doesn't know. If a breed cycle fails, CCC doesn't log it.

**Bridge to build:** `ccc-os/breeder_monitor.py` — a monitor that:
1. Polls `BreederDaemonV2.get_status()` every 15 min
2. Applies CCC rubric: diversity < threshold → ACT_NOW
3. If ACT_NOW, adds "Run diversity injection" to CCC task queue
4. Logs breeder lifecycle events to CCC's `output/` directory

**Code needed:** ~60 lines, 4 tests.

---

### 🔗 4. cocapn-plato ↔ sunset-ecosystem (WARM)
**Status:** Partial — `roomgrid_plato_observer.py` writes tiles, but tile pipeline doesn't read them.

**What cocapn-plato does:** `tile-pipeline.py` auto-captures MUD exploration. `fleet-snapshot.py` reads fleet state. Webhooks push tiles to external systems.

**What sunset-ecosystem does:** RoomGrid observer writes diversity/occupancy/lifecycle tiles to PLATO store.

**Gap:** The tile pipeline is designed for MUD rooms, not breeding metrics. Fleet snapshot doesn't include breeder state. Webhooks don't push diversity alerts.

**Bridge to build:** `cocapn-plato/scripts/breeder_snapshot.py` — extends fleet snapshot with:
1. BreederDaemonV2 state (active agents, diversity score, thermal pressure)
2. Lifecycle transitions in the last N ticks
3. Diversity collapse alerts (if any)
4. Export to webhook target for Casey notifications

**Code needed:** ~100 lines, 3 tests.

---

### 🔗 5. cocapn-traps ↔ sunset-ecosystem (HOT — novel)
**Status:** None — Traps don't monitor breeders.

**What cocapn-traps does:** Defines "crab traps" — prompts designed to lure agents into generating valuable tiles. Evaluates trap quality.

**What sunset-ecosystem needs:** Round 10 proved diversity collapse is real. We need automated detection.

**Bridge to build:** `cocapn-traps/traps/diversity_collapse_trap.py` — a trap that:
1. Monitors breeder diversity score over time
2. If diversity drops for 3 consecutive rounds → triggers `ConvergenceGuard`
3. Generates a "diversity crisis" tile with recommended action
4. Can be run as a scheduled trap, not just a prompt lure

**This is a new pattern:** Traps as operational monitors, not just content generators.

**Code needed:** ~70 lines, 4 tests.

---

### 🔗 6. hebbian-router ↔ sunset-ecosystem (COOL)
**Status:** None — Router doesn't handle breeding traffic.

**What hebbian-router does:** Self-optimizing load balancer. Traffic patterns strengthen/weaken routes.

**What sunset-ecosystem does:** Cross-ship breeding (CRDT merge) involves moving agents between ships.

**Gap:** No load balancing for breeding traffic. If Ship A breeds 100 agents and Ship B breeds 2, the network is unbalanced.

**Bridge to build:** `swarm/breeding_router.py` — a Hebbian-inspired router that:
1. Monitors breeding traffic volume per ship
2. Routes merge requests to underutilized ships
3. Strengthens routes that successfully complete merges
4. Weakens routes that timeout or fail

**This is speculative** — not needed until we have >2 ships breeding simultaneously.

**Code needed:** ~120 lines, 6 tests. Priority: P2.

---

### 🔗 7. vector-novelty + pareto-tournament ↔ sunset-ecosystem (DONE)
**Status:** Complete — These were extracted FROM sunset-ecosystem.

**What they do:** Standalone packages for diversity scoring and multi-objective selection.

**What sunset-ecosystem does:** Uses them internally.

**Gap:** None. These are the upstream packages. But: sunset-ecosystem should test against latest versions, not pinned copies.

**Action:** Add cross-repo integration tests: install latest `vector-novelty` and `pareto-tournament` from PyPI, verify sunset-ecosystem still works.

---

## Priority Ranking

| # | Bridge | Status | Effort | Impact |
|---|--------|--------|--------|--------|
| 1 | **Compiler → RoomGrid** | Partial | 150 loc | **🔥 Critical** — auto-optimize hottest path |
| 2 | **Health → Thermal** | Partial | 80 loc | **🔥 Critical** — unified thermal model |
| 3 | **Traps → Diversity Collapse** | None | 70 loc | **🔥 Novel** — operational monitoring pattern |
| 4 | **CCC → Breeder Monitor** | Partial | 60 loc | Warm — better orchestration |
| 5 | **PLATO → Breeder Snapshot** | Partial | 100 loc | Warm — better observability |
| 6 | **Router → Breeding Traffic** | None | 120 loc | Cool — future scaling |

## What I Should Build Next

1. **Bridge #1** (Compiler → RoomGrid) — `sunset/compiler_integration.py` with `RoomGridCompiler` class
2. **Bridge #2** (Health → Thermal) — `sunset/health_thermal_bridge.py`
3. **Bridge #3** (Traps → Diversity) — `cocapn-traps/traps/diversity_collapse_trap.py`

These three are the highest-leverage, most novel, and most needed right now. The rest can wait.

---

*kimi1, Fleet Orchestrator | May 24, 2026*
