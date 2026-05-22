# Sunset Ecosystem — Integration Map
> **Branch:** `turbovec-integration-ccc`  
> **Commits:** 25  
> **Generated:** 2026-05-22 by Code Integration Auditor (subagent)  
> **Scope:** `nerve/`, `swarm/`, `sunset/`, `scripts/`, `docs/`

---

## 1. Module Dependency Graph

```
                          ┌─────────────────┐
                          │   scripts/      │
                          │ (demo,bench)    │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  NerveTopology  │  ← nerve/topology.py
                          │   (orchestrator)│
                          └────────┬────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼──────┐          ┌────────▼────────┐        ┌────────▼────────┐
│   nerve/     │          │     nerve/        │        │    sunset/      │
│   fiber.py   │◄────────►│   routing.py      │        │   compiler.py   │
│ (perceive)   │          │ (fire, feedback)  │        │ (auto-compile)  │
└──────────────┘          └────────┬──────────┘        └────────┬────────┘
                                   │                          │
                          ┌────────▼────────┐                 │
                          │   nerve/          │◄────────────────┘
                          │   room_grid.py    │  (GridBackendSelector)
                          │ (JEPAGrid)        │
                          └────────┬──────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼──────┐          ┌────────▼────────┐        ┌────────▼────────┐
│   swarm/     │          │    swarm/         │        │    sunset/      │
│ tournament.py│◄────────►│ breeder_daemon.py │        │ flux_integration.py│
│ (AgentScore) │          │ (AutoBreeder)     │        │ (constraint check)│
└──────────────┘          └────────┬──────────┘        └─────────────────┘
                                   │
                          ┌────────▼────────┐
                          │    swarm/         │
                          │   thermal.py      │
                          │ (ThermalBudget)   │
                          └───────────────────┘
```

### Import Matrix

| Importer | Imports From | What |
|----------|-------------|------|
| `nerve/topology.py:12-16` | `nerve.room_grid` | `RoomGrid`, `JEPAGrid` |
| `nerve/topology.py:12-16` | `nerve.fiber` | `NerveFiber`, `FiberState`, `SensoryTile` |
| `nerve/topology.py:12-16` | `nerve.routing` | `RoutingLayer` |
| `nerve/topology.py:105` | `sunset.compiler` | `Compiler` (lazy, in `enable_compiler()`) |
| `nerve/room_grid.py:20-30` | `ctypes` | `_RUST_LIB`, `_CUDA_LIB` (hardware detection) |
| `nerve/routing.py` | `numpy` | vectorized `fire_fast()`, `_activate_channels_limited()` |
| `swarm/breeder_daemon.py:15-17` | `nerve.room_grid` | `RoomGrid` |
| `swarm/breeder_daemon.py:18-20` | `swarm.thermal` | `DeviceType`, `ThermalBudget` |
| `swarm/breeder_daemon.py:21` | `swarm.tournament` | `AgentScore`, `TournamentRound`, `breed` |
| `swarm/breeder_daemon.py:60` | `swarm.vector_table` | `FluxVectorTable` (optional, lazy) |
| `sunset/compiler.py:28` | `sunset.codegen` | `CodeGenerator`, `GeneratedKernel`, `PythonAnalyzer` |
| `sunset/flux_integration.py` | `ctypes` | `_RustBackend` loads `libflux_vm.so` |
| `scripts/demo_full_stack.py:10-12` | `nerve.topology`, `swarm.breeder_daemon`, `swarm.thermal` | full stack |

**Note:** `swarm/chaos.py` has **zero inbound imports** from the rest of the ecosystem. It is a disconnected utility.

---

## 2. Data Flow

```
Raw Signal (ndarray[64])
    │
    ▼
┌─────────────────┐
│ fiber.perceive()│  ← nerve/fiber.py:174
│  (NerveFiber)   │
└────────┬────────┘
         │ SensoryTile(pattern_id, confidence, state, features)
         ▼
┌─────────────────┐
│ routing.fire()  │  ← nerve/routing.py:116 (fire_fast)
│  (RoutingLayer) │     or nerve/routing.py:97 (fire, scalar)
└────────┬────────┘
         │ list[room_id]
         ▼
┌─────────────────┐
│ _encode_tile()  │  ← nerve/topology.py:180
│  (deterministic │     LUT + state scaling
│   hash → vector)│
└────────┬────────┘
         │ ndarray[64]
         ▼
┌─────────────────┐
│ grid.tick()     │  ← nerve/room_grid.py:310
│  (RoomGrid)     │
└────────┬────────┘
         │ {"fired": N, "ids": [...]}
         ▼
┌─────────────────┐
│ routing.feedback()│ ← nerve/routing.py:140 (feedback_batch)
│  (strength update)│
└────────┬────────┘
         │ Hebbian reinforce / decay
         ▼
┌─────────────────┐
│ grid.cold()     │  ← nerve/room_grid.py:295
│  (activity check)│
└────────┬────────┘
         │ list[room_idx]
         ▼
┌─────────────────┐
│ breeder.auto_breed()│ ← swarm/breeder_daemon.py:130
│  (tournament +    │
│   rebirth)        │
└─────────────────┘
```

### Data Structures That Cross Module Boundaries

| Structure | Produced By | Consumed By | Fields Used |
|-----------|------------|-------------|-------------|
| `SensoryTile` | `fiber.py` | `topology.py` | `pattern_id`, `state`, `confidence` |
| `TickResult` | `topology.py` | `scripts/demo_full_stack.py` | `latency_ms`, `rooms_fired`, `routes_compiled` |
| `AgentScore` | `breeder_daemon.py` | `tournament.py` | `ethos`, `pathos`, `logos` (but all set = activity) |
| `RebirthRecord` | `breeder_daemon.py` | nobody | `_log` is write-only; no reader exists |

---

## 3. Control Flow: What Calls What

### `NerveTopology.tick()` — The Master Clock

```
tick(signals)
  ├── 1. COLLECT: fiber.perceive(signal)          [nerve/fiber.py:174]
  │      └── produces SensoryTile
  │
  ├── 2. SELECT: routing.fire_fast(fid)           [nerve/routing.py:116]
  │      └── returns list[room_id]
  │
  ├── 3. ENCODE: _encode_tile(tile)               [nerve/topology.py:180]
  │      └── cached LUT lookup
  │
  ├── 4. GRID: grid.tick(combined_signal)         [nerve/room_grid.py:310]
  │      └── JEPAGrid._forward() → novelty() → firing
  │
  ├── 5. FEEDBACK: routing.feedback_batch()       [nerve/routing.py:140]
  │      └── batch reinforce for fired rooms + penalty for cold
  │
  ├── 6. REGULATE: chaos decay                    [nerve/topology.py:260]
  │      └── compiled_fraction → chaos *= decay
  │
  ├── 7. PERIODIC: channel decay (every 100 ticks) [nerve/routing.py:...]
  │
  └── 8. PERIODIC: auto-compile (every N ticks)   [nerve/topology.py:92]
         └── Compiler.compile_hotspots()
```

### `AutoBreeder` — The Background Daemon

```
start() → _run_loop()
  └── every `interval` seconds:
        auto_breed()
          ├── grid.cold(thresh)                   [nerve/room_grid.py:295]
          ├── grid.top(k)                         [nerve/room_grid.py:285]
          ├── build AgentScore population         [swarm/breeder_daemon.py:150]
          │      └── ethos=pathos=logos=activity/max_activity  ← FLAT SCORING
          ├── TournamentRound.run()               [swarm/tournament.py:120]
          ├── select_parents()                    [swarm/breeder_daemon.py:90]
          │      └── vector search OR random
          ├── breed()                             [swarm/tournament.py:210]
          ├── thermal.can_spawn() / sacrifice     [swarm/thermal.py:90]
          └── grid.rebirth() + weight clone       [swarm/breeder_daemon.py:330]
```

### `Compiler` — The Profiler Daemon

```
enable_compiler() → Compiler.install()
  └── monkey-patches all callable attrs in "nerve" + "nerve.room_grid"

tick() → _maybe_auto_compile()
  └── every 50 ticks (after tick 100):
        compile_hotspots(top_n=3)
          ├── profiler.get_hotspots()            [sunset/compiler.py:340]
          ├── CodeGenerator.compile()             [sunset/codegen.py:270]
          ├── validate (A/B test)                 [sunset/codegen.py:310]
          ├── measure_speedup()                   [sunset/codegen.py:330]
          └── deploy (hot-swap)                   [sunset/codegen.py:350]
```

---

## 4. Configuration Flow

### Hardcoded Constants (No Config Files)

There are **zero external config files** in the ecosystem. Everything is hardcoded Python constants:

| Constant | Value | Location | Impact |
|----------|-------|----------|--------|
| `_RUST_ONESHOT_THRESHOLD` | 500 rooms | `nerve/room_grid.py:29` | Switch to Rust oneshot |
| `_RUST_PERSIST_THRESHOLD` | 50 rooms | `nerve/room_grid.py:30` | Switch to Rust persistent |
| `_CUDA_THRESHOLD` | 1000 rooms | `nerve/room_grid.py:31` | Switch to CUDA |
| `SAMPLE_RATE` | 0.05 (5%) | `sunset/compiler.py:43` | Profiler sampling rate |
| `COMPILE_THRESHOLD` | 100 calls | `sunset/compiler.py:44` | Min calls before compile |
| `SPEEDUP_THRESHOLD` | 2.0× | `sunset/compiler.py:45` | Min speedup to deploy |
| `chaos` (default) | 0.3 | `nerve/topology.py:42` | Initial exploration rate |
| `DEFAULT_BUDGETS` | GPU:9, CPU:36, iGPU:14, NPU:6 | `swarm/thermal.py:36` | Thermal caps |

### Environment Variables (Sparse)

| Variable | Used By | Default |
|----------|---------|---------|
| `FLUX_VM_PATH` | `sunset/flux_integration.py:34` | `../flux-vm-v3-temp/target/release/flux_vm` |
| `FLUX_COMPILER_PATH` | `sunset/flux_integration.py:35` | `../flux-compiler-v0.1.0/compiler/fluxc.py` |

**Observation:** The `FLUX_COMPILER_PATH` env var is misspelled (`FLUST_COMPILER_PATH` in the code at `flux_integration.py:35`).

---

## 5. Missing Glue — Integration Gaps

### GAP-001: RoutingLayer → Tournament scoring
**Where:** `nerve/routing.py:fire_fast()` → `swarm/tournament.py:TournamentRound.run()`  
**Problem:** `Route.fires` and `Route.strength` metrics from the routing layer are never fed into tournament fitness calculation. The breeder creates `AgentScore` with `ethos=pathos=logos=activity/max_activity` — a flat 1D score that ignores routing topology entirely.  
**Impact:** Tournament selects agents based only on room firing frequency, not on how well-connected they are in the routing graph. A room with high activity but weak routes is treated the same as a room with moderate activity and strong compiled pathways.  
**Fix:** In `swarm/breeder_daemon.py:150-158`, replace flat scoring with a composite that includes `routing.get_route_strength(room_id)`:
```python
route_strength = self._avg_route_strength(rid)  # new helper
ethos = activity / max_activity
pathos = route_strength
logos = activity / max_activity
```
**Effort:** Small (1 helper function, 3 lines changed)

---

### GAP-002: AutoBreeder → Compiler (zero coordination)
**Where:** `swarm/breeder_daemon.py:auto_breed()` ↔ `sunset/compiler.py:Compiler`  
**Problem:** The breeder rebirths rooms by cloning parent weights (`_rebirth_with_clone`), but it never informs the compiler that old compiled kernels for those room indices are now stale. The compiler's monkey-patched functions still reference old weight signatures.  
**Impact:** After rebirth, compiled `batch_novelty` or `_forward` kernels may produce incorrect results because they were compiled against old weight array shapes or signatures. This is a silent correctness bug.  
**Fix:** Add a callback hook in `AutoBreeder.__init__`:
```python
self._compiler: Optional[Compiler] = None  # set externally
```
In `AutoBreeder._rebirth_with_clone()`, after weight mutation, call:
```python
if self._compiler is not None:
    self._compiler.invalidate_cache_for(room_id)
```
**Effort:** Small (2 files, ~8 lines)

---

### GAP-003: Compiler → CodeGenerator Rust backend is a stub
**Where:** `sunset/codegen.py:RustGenerator.generate()` (lines 180-200)  
**Problem:** The `RustGenerator` class claims to compile Python to Rust but returns a hardcoded error: `"Procedural Rust generation v2 not yet implemented."`. The `RustBackend` in `compiler.py` delegates to this stub, meaning no Rust compilation ever happens despite the architecture claiming three backends (Numba, Rust, CUDA).  
**Impact:** The compiler pipeline has a dead branch. Functions that should go to Rust (dict-heavy, string-heavy per the analyzer) silently fall back to Python with no user-visible warning except a log message that may never be read.  
**Fix:** Either (a) implement procedural generation using `ast` → `quote!` macros, or (b) remove the Rust backend from the default pipeline and document it as future work. **Recommended:** Option (b) for v0.4, add a `TODO` ticket for (a).  
**Effort:** Small for option (b) — delete `RustBackend` from `Compiler.backends` list and update docs. Medium for option (a) — ~2 weeks.

---

### GAP-004: FLUX constraint checker → Breeder (no constraint-aware breeding)
**Where:** `sunset/flux_integration.py:apply_constraint_feedback()` → `swarm/breeder_daemon.py:auto_breed()`  
**Problem:** The FLUX integration penalizes violating rooms by increasing chaos (`grid.chaos[violations] += 0.1`), but this information never reaches the breeder. The breeder may select a parent that produced constraint-violating offspring in the previous generation.  
**Impact:** The spec `SPEC_BREEDER_DAEMON_V2.md` §6.3 explicitly calls for `apply_constraint_feedback()` integration to avoid re-breeding from agents with FLUX violations, but this is not implemented.  
**Fix:** In `AutoBreeder.select_parents()`, filter out agents whose room index is in a `self._violation_log` set. Populate this set via a callback from `apply_constraint_feedback()`:
```python
# In breeder_daemon.py
self._violation_log: set[int] = set()

def _on_constraint_violation(self, room_idx: int) -> None:
    self._violation_log.add(room_idx)
```
Then in `select_parents()`, skip `room_idx` in `self._violation_log`.
**Effort:** Small (~10 lines)

---

### GAP-005: ThermalBudget → Compiler (no thermal-aware compilation)
**Where:** `swarm/thermal.py:ThermalBudget` ↔ `sunset/compiler.py:Compiler.compile_hotspots()`  
**Problem:** The compiler auto-compiles hot functions every 50 ticks without checking whether the system has thermal headroom. Numba compilation spawns LLVM processes that consume CPU; this can starve the grid's real-time tick budget.  
**Impact:** On CPU-constrained systems (e.g., Jetson Nano), a Numba compilation at tick 100 can cause a 500ms+ latency spike because LLVM hogs cores that the grid forward pass needs.  
**Fix:** Add a `thermal_check` parameter to `_maybe_auto_compile()`:
```python
def _maybe_auto_compile(self, thermal: Optional[ThermalBudget] = None) -> list[str]:
    if thermal and not thermal.can_breed(threshold=0.9):
        return []  # skip compilation, system is hot
```
Pass `self._thermal` from `NerveTopology` (already has it in `demo_full_stack.py` but not wired into topology).
**Effort:** Small (~5 lines)

---

### GAP-006: NerveTopology → AutoBreeder (missing thermal reference)
**Where:** `nerve/topology.py:NerveTopology.__init__()` vs `scripts/demo_full_stack.py:25`  
**Problem:** `NerveTopology` does not accept or store a `ThermalBudget`. The demo script creates both independently (`topo = NerveTopology(...); thermal = ThermalBudget(); breeder = AutoBreeder(topo.grid, thermal, ...)`). This means the topology has no thermal awareness — it never throttles ticks when the system is overloaded.  
**Impact:** The topology runs at full speed regardless of thermal state. On resource-constrained hardware, this causes thermal throttling at the OS level, degrading performance unpredictably.  
**Fix:** Add `thermal: Optional[ThermalBudget] = None` to `NerveTopology.__init__()`. In `tick()`, if thermal is provided and `thermal.thermal_headroom() > 0.95`, insert a `time.sleep(0.001)` backoff to let the system cool.  
**Effort:** Small (~8 lines)

---

### GAP-007: Hebbian channels are never activated by grid co-firing
**Where:** `nerve/routing.py:HebbianChannel.activate()`  
**Problem:** `HebbianChannel` has an `activate()` method that strengthens co-activation weights, but nothing in the ecosystem calls it. `NerveTopology.tick()` creates channels between Penrose-adjacent rooms (`topology.py:70-76`), but there is no code that calls `channel.activate()` when two rooms fire together. The channels exist but never strengthen.  
**Impact:** The Hebbian "neurons that fire together wire together" mechanism is decorative. Channels remain at their initial weight of 0.1 forever, providing no adaptive structure to the network.  
**Fix:** In `NerveTopology.tick()`, after `grid.tick()` returns fired room IDs, iterate co-fired pairs and activate their channels:
```python
fired_ids = grid_result.get("ids", [])
for i, a in enumerate(fired_ids):
    for b in fired_ids[i+1:]:
        key = self.routing._channel_key(f"room-{a}", f"room-{b}")
        ch = self.routing._channels.get(key)
        if ch:
            ch.activate()
```
**Effort:** Small (~8 lines)

---

### GAP-008: ChaosProbability module is completely disconnected
**Where:** `swarm/chaos.py` — zero imports from any other module  
**Problem:** `ChaosProbability` is a well-designed decaying probability class, but nothing imports it. `RoutingLayer` in `nerve/routing.py` uses a simple float `self.chaos` with manual decay (`self.chaos = max(0.01, self._base_chaos * ...)` in `topology.py:260`). The `ChaosProbability` class with its adaptive decay curves is never used.  
**Impact:** Dead code. 430 lines of `swarm/chaos.py` serve no purpose in the current system. The routing chaos decay logic is duplicated (and simpler) in `NerveTopology`.  
**Fix:** Replace the float chaos in `RoutingLayer` with `ChaosProbability`. Update `NerveTopology` to call `chaos_prob.update(adaptation_score)` instead of manual decay.  
**Effort:** Small (~15 lines in routing.py + topology.py)

---

### GAP-009: Vector table is empty in all current flows
**Where:** `swarm/breeder_daemon.py:_select_parents_vector()` (lines 270-340)  
**Problem:** The breeder has a `FluxVectorTable` integration path, but no code ever populates the vector table with agent vectors. The `_select_parents_vector()` method falls back to random selection on every call because `len(self._vector_table) == 0`. The `add()` call in `auto_breed()` (line 245-254) adds child vectors, but these are children of random parents — the table starts empty and only ever contains post-breed vectors, never the original population.  
**Impact:** Diversity-aware breeding is a no-op. The breeder always uses random parent selection despite having a vector table wired in.  
**Fix:** In `AutoBreeder.__init__()`, after receiving `grid`, iterate all rooms and seed the vector table with their initial latent fingerprints:
```python
for rid in range(grid.n):
    self._vector_table.add(AgentVector(
        agent_id=rid,
        vector=grid.latents[rid] if grid.latents is not None else np.zeros(16),
        fitness=0.5,
        generation=0,
    ))
```
**Effort:** Small (~8 lines)

---

### GAP-010: RebirthRecord log is write-only
**Where:** `swarm/breeder_daemon.py:RebirthRecord` (line 35)  
**Problem:** The `AutoBreeder` maintains a `_log: list[RebirthRecord]` that grows unbounded. There is no API to read, query, or truncate this log. The `log` property returns a copy, but nothing in the ecosystem consumes it. The records contain valuable lineage data (parent IDs, tick, vector search flag) that could drive diversity analysis or mesh synchronization.  
**Impact:** Memory leak on long runs. Unbounded list growth. Valuable breeding history is captured then ignored.  
**Fix:** (a) Add a `maxlen` or periodic truncation; (b) Expose a `get_lineage(agent_id)` method that traces parent chains; (c) Wire the log to a WAL file (per `SPEC_BREEDER_DAEMON_V2.md`).  
**Effort:** Small for (a), Medium for (b+c).

---

### GAP-011: `inject_chaos()` from `swarm/chaos.py` is never called
**Where:** `swarm/chaos.py:inject_chaos()` (line 95)  
**Problem:** This is the primary API of `chaos.py` — it takes a routes dict and swaps/reroutes destinations probabilistically. Despite being the most complex function in the file (~80 lines), nothing calls it. The routing layer's exploration is handled by `Route.fire()` adding a random check against `chaos`, not by structural route mutation.  
**Impact:** The sophisticated swap/reroute chaos injection mechanism is completely unused. Route topology never changes structurally — only per-fire random noise.  
**Fix:** Call `inject_chaos()` periodically from `NerveTopology.tick()` (e.g., every 500 ticks) to rewire low-strength routes:
```python
if self.tick_count % 500 == 0:
    routes_dict = self.routing._to_dict()  # new helper
    new_routes, events = inject_chaos(routes_dict, self._chaos_prob)
    self.routing._from_dict(new_routes)
```
**Effort:** Small (~10 lines + 2 helpers in routing.py)

---

### GAP-012: Rust kernel novelty computation is still Python
**Where:** `nerve/room_grid.py:JEPAGrid.tick()` → `batch_novelty()`  
**Problem:** The Rust kernel (`nerve/src/lib.rs`) only implements forward pass. Novelty computation (`batch_novelty`) and history ring buffer management remain in Python. For 10K rooms, forward is ~2.35ms but novelty is ~2.2ms (Numba) or ~15ms (pure Python), meaning the Rust kernel's speedup is partially wasted waiting on Python.  
**Impact:** The SPEC-JEPA-GRID-OPTIMIZATION.md success criteria target "full tick < 3ms for 10K rooms" but the current split (Rust forward + Python novelty) means the best observed full tick is ~5ms.  
**Fix:** Extend the Rust kernel with `jepa_novelty_batch()` that takes `(latents, history_tensor, hist_count, topk, max_hist)` and returns novelty scores. Wire it in `room_grid.py` behind the same `kernel="auto"` flag as forward.  
**Effort:** Medium (~200 lines Rust, ~30 lines Python binding)

---

### GAP-013: `AgentPhase` enum in `sunset/agent.py` is never used
**Where:** `sunset/agent.py:AgentPhase` (imported but never referenced)  
**Problem:** `sunset/agent.py` exports `AgentPhase` (an Enum with INCUBATE, COMPETE, SURVIVE, BREED, SUNSET states). These states are described in `SPEC_BREEDER_DAEMON_V2.md` as the explicit lifecycle FSM, but the actual `AutoBreeder` in `swarm/breeder_daemon.py` does not import or use `AgentPhase`. Agents transition implicitly via `chaos` decay and activity levels, not via explicit state machine.  
**Impact:** The formal lifecycle FSM exists in code but is disconnected from the actual breeding runtime. State transitions are invisible and unlogged.  
**Fix:** Import `AgentPhase` in `AutoBreeder`. Add a `_phases: dict[int, AgentPhase]` dict. Update phase on every rebirth and tournament. This is the foundation for the WAL logging described in `SPEC_BREEDER_DAEMON_V2.md`.  
**Effort:** Small (~20 lines)

---

### GAP-014: `trinity_scorer.py` is disconnected from the tournament
**Where:** `sunset/trinity_scorer.py` exports `trinity_score()` and `normalize_connection()`  
**Problem:** `sunset/trinity_scorer.py` provides a scoring function that normalizes connections, but `swarm/tournament.py` computes scores as `a.product` (simple ethos×pathos×logos). The trinity scorer's normalization logic (which includes distance-to-center weighting) is never used.  
**Impact:** The trinity scorer module is dead code. The tournament uses naive product scoring, which may overweight agents strong in one dimension but weak in others.  
**Fix:** In `swarm/tournament.py:TournamentMatch.__init__()`, replace `a.product` with `trinity_score(a, population)` after importing from `sunset.trinity_scorer`.  
**Effort:** Small (1 import, 2 lines)

---

### GAP-015: `sunset/generation_runner.py` is not wired into `AutoBreeder`
**Where:** `sunset/generation_runner.py` exports `GenerationRunner`, `GenerationReport`, `EthosProfile`  
**Problem:** The `GenerationRunner` class is designed to orchestrate multi-generation runs with reporting, but `AutoBreeder` (the actual daemon that runs generations) does not import or use it. `GenerationRunner` has no callers in the entire codebase.  
**Impact:** Generation reporting infrastructure exists but produces no reports. The `EthosProfile` dataclass (which tracks ethos distribution across generations) is never instantiated.  
**Fix:** Replace `AutoBreeder._run_loop()` with a `GenerationRunner` delegate, or have `AutoBreeder` emit `GenerationReport` objects after each breeding cycle.  
**Effort:** Medium (~50 lines to wire together)

---

## 6. Implementation Priority

### P0 — Breaks Core Functionality (Data Doesn't Flow / Silent Bugs)

| Gap | Problem | Why P0 |
|-----|---------|--------|
| **GAP-002** | Compiler kernels go stale after rebirth | Silent correctness bug — compiled functions reference old weights |
| **GAP-004** | Breeder ignores FLUX violations | Violating parents re-breed, propagating bad behavior |
| **GAP-007** | Hebbian channels never activate | Core neural-plasticity mechanism is decorative |
| **GAP-009** | Vector table is empty | Diversity breeding is a no-op despite being a headline feature |

### P1 — Missing Optimization (Works But Suboptimal)

| Gap | Problem | Why P1 |
|-----|---------|--------|
| **GAP-001** | Tournament ignores routing strength | Selection is 1D flat, missing network structure signal |
| **GAP-005** | Compiler ignores thermal headroom | Latency spikes from LLVM compilation on hot systems |
| **GAP-006** | Topology has no thermal awareness | Runs flat out regardless of device state |
| **GAP-012** | Rust kernel doesn't do novelty | Forward speedup wasted waiting on Python novelty |
| **GAP-014** | Trinity scorer disconnected | Better scoring exists but isn't used |

### P2 — Nice to Have (Convenience, Monitoring, Cleanup)

| Gap | Problem | Why P2 |
|-----|---------|--------|
| **GAP-003** | Rust compiler backend is a stub | Dead branch in pipeline; confusing to users |
| **GAP-008** | ChaosProbability module disconnected | `swarm/chaos.py` is dead code; should be wired or removed |
| **GAP-010** | RebirthRecord log is write-only | Memory leak + wasted lineage data |
| **GAP-011** | `inject_chaos()` never called | Route topology never mutates structurally |
| **GAP-013** | `AgentPhase` enum unused | FSM exists but is decorative |
| **GAP-015** | `GenerationRunner` has no callers | Reporting infrastructure is orphaned |

---

## 7. Architecture Assessment

### What Works

1. **The tick cycle is closed.** `NerveTopology.tick()` genuinely wires fiber → routing → grid → feedback into one coherent loop. This is the achievement of the 25 commits.
2. **Hardware auto-dispatch works.** `GridBackendSelector` correctly picks numpy/rust/cuda based on room count. The thresholds are sensible.
3. **The compiler pipeline (Numba) works.** 1129× speedup on `expensive_dot_product` is real. A/B validation and hot-swap are functional.
4. **FLUX Python fallback works.** Constraint checking runs without Rust deps. The API boundary is clean.

### What's Half-Baked

1. **The breeder is a skeleton with nice clothes.** It has `FluxVectorTable` integration, `RebirthRecord` logging, and thermal sacrifice — but the vector table is empty, the log is unread, and thermal only gates spawn (doesn't schedule).
2. **The compiler's Rust backend is a mirage.** It looks like a three-backend system (Numba/Rust/CUDA) but only Numba works. The Rust generator returns `error="not yet implemented"` on every call.
3. **Hebbian channels are decorative.** They are created but never activated. The "neurons that fire together wire together" tagline is unimplemented.
4. **Chaos exists in two incompatible forms.** `RoutingLayer` uses a float. `swarm/chaos.py` has a full class with decay curves, swap/reroute events, and adaptation tracking. They don't talk.
5. **Lifecycle FSM exists in spec only.** `AgentPhase` is exported but unused. Agents have no explicit state.

### Brutal Honesty

> The ecosystem is a **working demo with aspirational architecture.** The happy path (500 rooms, 200 ticks, Numba backend) runs beautifully. The advanced features (vector breeding, Hebbian learning, Rust compilation, thermal scheduling, FLUX integration) have the right APIs and the right hooks — but the wires behind the panels are often loose or missing entirely.

The 25 commits built a **convincing facade**. What remains is the **plumbing behind it** — and that's exactly what this map is for.

---

*End of Integration Map.*
