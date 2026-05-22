# Integration Map — Sunset Ecosystem v0.9

**Branch:** `turbovec-integration-ccc`  
**Generated:** 2026-05-22 by kimi1  
**Scope:** Full-stack integration audit of compiler, FLUX, routing, grid, breeder

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NerveTopology (nerve/topology.py)                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │ NerveFiber  │───→│ RoutingLayer│───→│  RoomGrid   │                 │
│  │ (perceive)  │    │ (fire_fast) │    │ (_forward)   │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│         ↑                                    │                          │
│         │         ┌──────────────────────────┘                          │
│         │         ↓                                                     │
│  ┌──────┴────────┐    ┌─────────────────────────────────────────┐       │
│  │ Feedback Loop │←───│  FLUX Constraint Checker              │       │
│  │ (reinforce)   │    │  (sunset/flux_integration.py)         │       │
│  └───────────────┘    └─────────────────────────────────────────┘       │
│                          ↑                                             │
│                    ┌─────┴─────┐                                        │
│                    │  FLUX VM  │  ←── Rust FFI (libflux_vm.so)         │
│                    │  (Rust)   │       compiled by FM on laptop        │
│                    └───────────┘                                        │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ↓
                    ┌─────────────────┐
                    │ Agentic Compiler │
                    │ (sunset/compiler) │
                    │  → Numba JIT     │
                    │  → Rust FFI      │
                    │  → CUDA (future) │
                    └─────────────────┘
```

---

## 2. Module-by-Module Status

### 2.1 Nerve Grid (`nerve/room_grid.py`)
| Feature | Status | Backend | Notes |
|---------|--------|---------|-------|
| Forward pass | ✅ Working | numpy | Einsum-based, 70 ticks/s @ 500 rooms |
| Forward pass | 🟡 Stub | rust_persistent | `PersistentRustGrid` class exists, needs `.so` |
| Forward pass | 🟡 Stub | rust_oneshot | `forward_rust_oneshot()` exists, needs `.so` |
| Forward pass | ✅ Working | CUDA | `PersistentCUDAGrid` + `jepa_kernel.cu` compiled by FM |
| Batch tick | ✅ Working | CUDA/Rust | `tick_batch()` amortizes kernel launch overhead |
| Novelty scoring | ✅ Working | Numba | `batch_novelty` compiled at import, 2ms @ 1000 rooms |
| Ring buffer | ✅ Working | numpy | Vectorized append, no Python loop |
| Breeding | ✅ Working | numpy | `breed()` clones + mutates weights |
| Chaos dynamics | ✅ Working | numpy | Vectorized decay, stays in [0.01, 1.0] |
| FLUX hook | ✅ Working | Python | `attach_flux_checker()`, applies feedback post-tick |

**API Surface:**
```python
RoomGrid(n=100, d=64, h=32, l=16, chaos=0.3)
RoomGrid.tick(x: np.ndarray) → dict{"fired": int, "ids": list[int], "tick": int}
RoomGrid.breed(parent_idx, child_idx) → None
RoomGrid.attach_flux_checker(checker) → None
```

### 2.2 Routing Layer (`nerve/routing.py`)
| Feature | Status | Notes |
|---------|--------|-------|
| Route firing (slow) | ✅ Working | `fire()` — Python scalar path |
| Route firing (fast) | ✅ Working | `fire_fast()` — vectorized, 60× faster |
| Compiled routes | ✅ Working | Strength > 0.9 skip random checks |
| Hebbian channels | ✅ Working | Co-activation strengthens existing channels |
| Hebbian auto-creation | ✅ Working | Channels auto-created on first co-fire |
| Feedback | ✅ Working | `feedback()` + `feedback_batch()` |
| Precomputed index | ✅ Working | `_routes_by_source` / `_routes_by_dest` |

**API Surface:**
```python
RoutingLayer(chaos=0.1, learning_rate=0.05)
RoutingLayer.add_route(source, destination, strength=0.5) → Route
RoutingLayer.fire_fast(source, destinations=None) → list[str]
RoutingLayer.feedback(source, destination, success) → None
```

### 2.3 Agentic Compiler (`sunset/compiler.py`)
| Feature | Status | Notes |
|---------|--------|-------|
| Profiler | ✅ Working | 5% sample rate, records calls + timing |
| Numba backend | 🟡 Partial | `CodeGenerator` has `numba.compile()` but API mismatch on `registry.Dispatcher` |
| Rust backend | 🔴 Stub | `RustBackend` class exists, no codegen pipeline |
| CUDA backend | 🔴 Not started | Needs `nvcc` + GPU |
| Auto-compile hook | ✅ Working | `NerveTopology.enable_compiler()` installs profiler |
| Hot-swap | 🔴 Not started | Compiled functions not swapped at runtime |

**Known Issue:** `sunset/codegen.py:336` references `numba.core.registry.Dispatcher` which doesn't exist in current Numba version. The `compile()` path fails for non-Numba functions. Already-`@njit` functions (like `batch_novelty`) work fine.

**API Surface:**
```python
Compiler.install(module_name=None) → patches module functions for profiling
Compiler.compile_hotspots(top_n=5) → list[CompilationResult]
NerveTopology.enable_compiler() → installs on self + room_grid
```

### 2.4 FLUX Integration (`sunset/flux_integration.py`)
| Feature | Status | Backend | Notes |
|---------|--------|---------|-------|
| Bounds checking | ✅ Working | Python | Values outside [-10, 10] flagged |
| L2 norm check | ✅ Working | Python | `max_l2_norm=25` default |
| Variance check | ✅ Working | Python | `max_variance=5` default |
| Presets | ✅ Working | Python | 3 presets: neural_bounds, safe_mode, exploration |
| Rust FFI | 🟡 Partial | Rust | `libflux_vm.so` needs FM compilation |
| RoomGrid hook | ✅ Working | Python | `apply_constraint_feedback()` called post-tick |

**API Surface:**
```python
FluxConstraintChecker(preset="neural_bounds")
FluxConstraintChecker.check_batch(latents) → np.ndarray[bool]
FluxConstraintChecker.get_violations(latents, room_ids) → list[ConstraintViolation]
```

### 2.5 FLUX VM (`flux-vm-v3-temp/`)
| Feature | Status | Notes |
|---------|--------|-------|
| Bounds check | ✅ Written | `flux_check_batch()` in `src/ffi.rs` |
| L2 norm check | ✅ Written | `ffi.rs` |
| Variance check | ✅ Written | `ffi.rs` |
| Unit tests | ✅ Written | 3 tests in `ffi.rs` |
| Compilation | 🔴 Blocked | `cargo` not available on this machine |
| FFI bindings | 🟡 Partial | Python side in `flux_integration.py` has `ctypes` path |

**FM Action Required:** Compile `libflux_vm.so` on laptop with `cargo build --release`

### 2.6 Breeder (`swarm/breeder.py`)
| Feature | Status | Notes |
|---------|--------|-------|
| Tournament | ✅ Working | Pareto selection + sunset candidates |
| Breeding | ✅ Working | `breed()` mutates weights |
| Vector table | 🟡 Stub | `FluxVectorTable` class exists, empty in tests |
| Daemon | 🔴 Not started | `BreedingDaemon` referenced in specs but not implemented |
| Lifecycle FSM | 🔴 Not started | EGG→INCUBATE→COMPETE→SURVIVE→BREED/SUNSET not wired |

---

## 3. Integration Points

### 3.1 RoomGrid ↔ FLUX
```
RoomGrid.tick(x)
  → latents = _forward(x)
  → if _flux_checker:
       apply_constraint_feedback(self, _flux_checker)
         → violations = checker.check_batch(latents)
         → chaos[violations] += 0.1
```
**Verified:** ✅ Attachment works, feedback runs, no crashes.

### 3.2 RoomGrid ↔ Compiler
```
NerveTopology.enable_compiler()
  → compiler.install("nerve.room_grid")
  → compiler.install("nerve.routing")
  → compiler.install("sunset.compiler")
  → tick() now triggers profiling on hot paths
```
**Verified:** ✅ Auto-compile hook fires, profiler tracks calls.
**Gap:** Compiler doesn't actually recompile `batch_novelty` because it's already `@njit` at import time. This is by design — the compiler is a "last resort" for unoptimized code.

### 3.3 RoomGrid ↔ RoutingLayer
```
NerveTopology.tick(signals)
  → fiber.perceive(signal) → tile
  → routing.fire_fast(fiber.id) → room_ids
  → grid.tick(tile.latent) → fired_rooms
  → routing.feedback(fiber.id, room_id, success=True)
```
**Verified:** ✅ Full cycle works end-to-end.

### 3.4 Compiler ↔ CodeGenerator
```
Compiler.compile_hotspots()
  → generator = CodeGenerator()
  → kernel = generator.compile(func, test_args)
     → if isinstance(func, numba.core.registry.Dispatcher):  # ← BROKEN
```
**Gap:** `numba.core.registry.Dispatcher` doesn't exist. Need `numba.core.registry.dispatcher.Dispatcher` or `numba.dispatcher.Dispatcher` depending on version.

---

## 4. Gap Tickets

| # | Priority | Component | Issue | Owner | Status |
|---|----------|-----------|-------|-------|--------|
| 1 | P0 | codegen.py | Fix Numba `Dispatcher` type check for compatibility | FM | ✅ Fixed — uses `CPUDispatcher` |
| 2 | P0 | flux-vm-v3 | Compile `libflux_vm.so` with `cargo build --release` | FM | 🔴 Blocked — no cargo on this machine |
| 3 | P0 | nerve/grid | Compile `libjepa_kernel.so` (Rust persistent backend) | FM | 🔴 Blocked — no cargo on this machine |
| 4 | P1 | breeder | Implement `BreedingDaemon` with lifecycle FSM | kimi1 | 🟡 Partial — `AutoBreeder` has daemon + thermal + compaction |
| 5 | P1 | breeder | Wire `FluxVectorTable` into parent selection | kimi1 | 🟡 Partial — `_select_parents_vector()` implemented, falls back gracefully |
| 6 | P1 | compiler | Implement runtime hot-swap (replace original with compiled) | kimi1 | 🔴 Not started |
| 7 | P1 | grid | Wire `jepa_kernel.cu` CUDA kernel into Python | kimi1 | ✅ Done — `PersistentCUDAGrid` + batch tick |
| 8 | P2 | routing | Auto-create Hebbian channels on first co-fire | kimi1 | ✅ Done |
| 9 | P2 | tests | Fix pre-existing `test_breeder_daemon.py` / `test_breeder_integration.py` | kimi1 | ✅ Done |
| 10 | P2 | docs | Write `docs/FLUX_INTEGRATION.md` user guide | kimi1 | ✅ Done |

---

## 5. Performance Baseline

| Metric | Value | Notes |
|--------|-------|-------|
| Grid 500 rooms | ~70 ticks/s | numpy backend, Alibaba Cloud |
| Grid 1000 rooms | ~33 ticks/s | numpy backend |
| Grid 2000 rooms | ~21 ticks/s | numpy backend |
| Novelty 1000 rooms | ~2ms | Numba JIT, already compiled |
| Novelty 10000 rooms | ~12ms | Numba JIT |
| Topology 1000×8 | ~10ms/tick | Full cycle: perceive → route → grid → feedback |
| Routing fire_fast | ~0.24ms | Vectorized, 1000 routes |
| FLUX check_batch | ~0.05ms | Python backend, 1000 rooms |
| Compiler profiler | ~0.5% overhead | 5% sample rate |

---

## 6. Test Coverage

| Module | Tests | Passing | Skipped | Notes |
|--------|-------|---------|---------|-------|
| room_grid | 12 | 12 | 0 | Forward, novelty, breed, chaos, buffer, FLUX, batch tick |
| routing | 10 | 10 | 0 | Determinism, strong routes, Hebbian auto-create, feedback |
| compiler | 11 | 9 | 2 | Profiler, auto-compile, Numba (skipped when API mismatch) |
| flux_integration | 10 | 10 | 0 | Bounds, L2, presets, RoomGrid hook, violations |
| breeder_daemon | 11 | 11 | 0 | Auto-breed, thermal, lifecycle, compaction |
| breeder_integration | 5 | 5 | 0 | Vector table, fallback, 10-cycle, compaction archive |
| grammar_security | 4 | 4 | 0 | Chaos vector blocked: path traversal, XSS, SQLi, code injection |
| performance | 6 | 0 | 6 | Latency regression (marked `slow`) |
| **Total new** | **69** | **61** | **8** | + pre-existing module tests |
| **Full suite** | ~420 | 369 | 12 | All pre-existing + new tests |

---

## 7. FM Compile Instructions

### FLUX VM
```bash
cd flux-vm-v3-temp/
cargo build --release
# Produces: target/release/libflux_vm.so
# Copy to: sunset-ecosystem/ (or set FLUX_VM_PATH env var)
```

### Rust Grid Backend
```bash
cd sunset-ecosystem/nerve/
cargo build --release
# Produces: target/release/libjepa_kernel.so
```

### CUDA Kernel (on JC1 with RTX 4050)
```bash
nvcc -arch=sm_89 jepa_kernel.cu -o jepa_kernel.so --shared
# Copy to: sunset-ecosystem/nerve/
```

---

## 8. Next Build Roadmap

1. **Fix codegen.py Numba compatibility** (1 line change — `Dispatcher` path)
2. **FM compiles Rust backends** (blocked on cargo availability)
3. **Implement BreedingDaemon** per `SPEC_BREEDER_DAEMON_V2.md`
4. **Wire FluxVectorTable** into breeder parent selection
5. **Write CUDA kernel** for JC1 RTX 4050
6. **Runtime hot-swap** for compiled functions
7. **Integration test for full breeding cycle** (currently 0 cycles in demo)

---

*"The map is not the territory, but without the map, the fleet is lost."*
*— kimi1, Fleet Integrator | 2026-05-22*
