# Sunset Ecosystem — Integration Status Report
**kimi1 (CCC) | May 22, 2026 | Branch: `turbovec-integration-ccc`**

---

## Executive Summary

The `turbovec-integration-ccc` branch now contains **20 commits** of performance and integration work across the full sunset ecosystem stack. Every layer has been touched: nerve grid, swarm breeder, agentic compiler, and hardware auto-dispatch.

**Key numbers:**
- `batch_novelty`: 15.4ms → 2.2ms (Numba, 6.9×)
- `expensive_dot_product`: 6.4ms → 0.006ms (Numba compiler, 1129×)
- Full-stack demo: 2000 rooms @ 21 ticks/s on Alibaba Cloud
- Rust persistent grid: zero-copy per tick after initial weight upload

---

## What Was Built

### 1. Nerve Layer (`nerve/`)

| Component | Status | Details |
|-----------|--------|---------|
| **Rust persistent grid** | ✅ Working | `libjepa_kernel.so` compiled, `PersistentRustGrid` class, weights live in Rust memory |
| **CUDA kernel** | ✅ Ready | `jepa_kernel.cu` written, needs FM's RTX 4050 to compile |
| **Auto-dispatch** | ✅ Working | `GridBackendSelector` picks numpy/rust/cuda based on room count |
| **Numba `batch_novelty`** | ✅ Working | Auto-selects at import, 6.9× speedup, seamless fallback |
| **Ring buffer history** | ✅ Working | Vectorized append, no Python loop, 3-entry ring buffer |

**Files:** `nerve/room_grid.py`, `nerve/src/lib.rs`, `nerve/src/jepa_kernel.cu`, `nerve/jepa_rust.py`, `nerve/Cargo.toml`

### 2. Swarm Layer (`swarm/`)

| Component | Status | Details |
|-----------|--------|---------|
| **Tournament** | ✅ Working | Pareto frontier, pairwise matches, breeding, sunset candidates |
| **AutoBreeder** | ✅ Working | Background daemon, thermal-aware, vector-table diversity breeding |
| **ThermalBudget** | ✅ Working | GPU/CPU/iGPU/NPU slot management, parent sacrifice |

**Files:** `swarm/tournament.py`, `swarm/breeder_daemon.py`, `swarm/thermal.py`

### 3. Sunset Compiler (`sunset/`)

| Component | Status | Details |
|-----------|--------|---------|
| **Hardware detection** | ✅ Working | `HardwareMap` auto-detects CUDA, Rust, Numba |
| **CodeGenerator** | ✅ Working | Numba backend: 1129× speedup on hot loops |
| **Profiler** | ✅ Working | Monkey-patches functions, samples calls, tracks optimization potential |
| **Auto-compile hook** | ✅ Wired | `Topology.enable_compiler()` + `_maybe_auto_compile()` every 50 ticks |
| **GridBackendSelector** | ✅ Working | Maps room count → backend (numpy/rust/cuda) |

**Files:** `sunset/compiler.py`, `sunset/codegen.py`

### 4. Integration

| Test | Status | Result |
|------|--------|--------|
| Topology + Grid | ✅ | 2000 rooms, Rust persistent, 21 ticks/s |
| Topology + Breeder | ✅ | Daemon thread runs, thermal respected |
| Topology + Compiler | ✅ | Profiler tracks `batch_novelty` (150 calls, 2.1ms avg) |
| Full-stack demo | ✅ | `scripts/demo_full_stack.py` exercises all layers |

---

## FM Note: What to Compile on Your RTX 4050

**File:** `docs/FM_NOTE_KIMI1_METAL.md` (pushed to branch)

```bash
cd sunset-ecosystem/nerve/src
# 1. Compile the Rust shared library
cargo build --release
# 2. Compile the CUDA kernel
nvcc -O3 -arch=sm_89 -shared -o jepa_cuda.so jepa_kernel.cu
# 3. Verify both load
python3 -c "from nerve.room_grid import RoomGrid; g=RoomGrid(1000); print(g)"
```

**Your ProArt has what neither of us have:**
- RTX 4050 (CUDA, 2560 CUDA cores)
- Ryzen AI 9 (XDNA 2 NPU, 50 TOPS)
- Radeon 890M iGPU (16 CUs)

This is the only machine in the fleet that can test the CUDA path.

---

## Architecture Decisions

### 1. Numba at Import Time vs Compiler at Runtime

`batch_novelty` uses Numba **at import time** (eager compilation). This is appropriate because:
- The function signature is fixed (always takes the same types)
- The speedup is consistent (6.9×)
- The compile cost (1.6s) amortizes over long runs

The **agentic compiler** handles functions that:
- Weren't pre-optimized by a human
- Have dynamic signatures
- Become hot unexpectedly at runtime

This creates a two-tier optimization: eager (known hot paths) + lazy (discovered hot paths).

### 2. Rust Persistent vs Oneshot

| | Oneshot | Persistent |
|--|---------|------------|
| Weights | Python → Rust per tick | Rust owns weights |
| Overhead | 7× `ascontiguousarray()` | Zero-copy |
| Use case | <500 rooms (overhead small) | 500+ rooms (amortizes) |
| Lifecycle | Create/destroy per call | Create once, tick many |

The auto-dispatch selects based on room count. For 2000 rooms, persistent is clearly faster.

### 3. Compiler Profiler Sampling

The profiler uses 5% sampling (`SAMPLE_RATE = 0.05`) but profiles **every call for the first 1000 calls**. This gives accurate data early without infinite overhead. The profiler detected `batch_novelty` at 150 calls — correctly identifying it as the hottest function.

---

## Open Questions / Next Builds

### P1: FLUX Integration

The FLUX VM v3 (`flux-vm-v3-temp/`) and compiler (`flux-compiler-v0.1.0/`) exist as separate repos. The integration point: **constraint checking in room outputs**.

Idea: After `RoomGrid.tick()`, run FLUX constraint checks on the latent vectors:
```python
# In RoomGrid.tick():
latents = self._forward(x)
violations = flux_vm.check_batch(latents, preset="neural_bounds")
if violations.any():
    self.chaos[violations] += 0.1  # Increase chaos in violating rooms
```

This would make the grid **self-correcting** — rooms that violate constraints get more chaotic (exploratory).

### P2: Compiler Rust Backend

The compiler currently only has Numba backend. The Rust backend (`RustBackend`) exists as a stub. To complete:
- Extract loop patterns from Python AST
- Generate Rust code using `quote!` macros
- Compile with `cargo build`
- Load via `ctypes` like `jepa_rust.py`

This would enable compiling arbitrary Python loops to Rust, not just the hardcoded `jepa_kernel.cu`.

### P3: Vector Table Search in Breeder

The breeder has `FluxVectorTable` integration but the vector table is empty in current tests. To activate:
- Add `AgentVector` to table after each `tick()`
- Use `vector_table.search()` for diversity-aware parent selection
- Breed dissimilar parents (maximize genetic distance)

### P4: End-to-End Benchmark Suite

We need systematic benchmarking across:
- Room counts: 100, 500, 1000, 5000, 10000
- Backends: numpy, rust_oneshot, rust_persistent, cuda
- Signals: structured (sine), random, mixed
- Metrics: latency p50/p99, throughput, memory

This would produce performance regression detection for CI.

### P5: FM Hardware Validation

Once FM compiles the CUDA kernel on his RTX 4050:
- Run `scripts/demo_full_stack.py` with `n_rooms=10000`
- Compare: Alibaba Cloud (CPU-only) vs ProArt (CUDA)
- Expected: 10-50× speedup on large grids
- Update `GridBackendSelector` with real CUDA threshold

---

## Files Changed (20 commits)

```
nerve/Cargo.toml
nerve/jepa_rust.py
nerve/room_grid.py
nerve/src/jepa_kernel.cu
nerve/src/lib.rs
scripts/bench_compiler.py
scripts/demo_full_stack.py
scripts/microbench.py
scripts/test_compiler.py
sunset/codegen.py
sunset/compiler.py
swarm/tournament.py
docs/FM_NOTE_KIMI1_METAL.md
```

---

## How to Test

```bash
git clone -b turbovec-integration-ccc https://github.com/SuperInstance/sunset-ecosystem.git
cd sunset-ecosystem

# Full-stack demo
PYTHONPATH=$(pwd) python3 scripts/demo_full_stack.py 2000 300

# Compiler demo
PYTHONPATH=$(pwd) python3 scripts/test_compiler.py

# Microbenchmark
PYTHONPATH=$(pwd) python3 scripts/microbench.py
```

---

## Branch Status

```
turbovec-integration-ccc: 20 commits ahead of main
Ready for: review, merge, or further integration work
```

**Next recommended action:** FM compiles CUDA kernel on RTX 4050, runs benchmark, we merge to main.

---

*kimi1 | Sunset Ecosystem Integrator | Cocapn Fleet*
