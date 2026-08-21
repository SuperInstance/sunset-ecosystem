# Sunset Ecosystem — Operations Manual

**Branch:** `turbovec-integration-ccc`  
**Version:** 0.9  
**Last Updated:** 2026-05-22  
**Audience:** Human operators running fleet nodes  

---

## Table of Contents

1. [Quick Start](#1-quick-start) — 10-minute clone-to-running
2. [Starting the Fleet](#2-starting-the-fleet) — Service startup order
3. [Monitoring](#3-monitoring) — Reading the vital signs
4. [Troubleshooting](#4-troubleshooting) — Common failures and fixes
5. [Safe Shutdown](#5-safe-shutdown) — Graceful stop sequence
6. [Upgrade Path](#6-upgrade-path) — Hot-swapping without restart
7. [Emergency Procedures](#7-emergency-procedures) — Kill switch, WAL recovery, rollback

---

## 1. Quick Start

> **Goal:** Clone, install, run tests, start a single node. Target time: 10 minutes.

### 1.1 Clone the repo

```bash
git clone -b turbovec-integration-ccc https://github.com/SuperInstance/sunset-ecosystem.git
cd sunset-ecosystem
```

### 1.2 Install dependencies

```bash
pip install -r requirements.txt   # numpy, numba, etc.
```

**Optional (for GPU nodes):**
```bash
# CUDA backend — only if nvcc is available
# Rust backend — only if cargo is available
```

> **Note:** The system works without CUDA or Rust. It falls back to NumPy automatically. Do not block on missing compilers.

### 1.3 Run the test suite

```bash
PYTHONPATH=$(pwd) python3 -m pytest tests/ -x --tb=short
```

Expected: **369 passing**, 12 skipped (Numba API mismatch + slow regression tests). If you see failures in `room_grid`, `routing`, or `flux_integration`, stop and read [Troubleshooting](#4-troubleshooting).

### 1.4 Start a single node

```bash
PYTHONPATH=$(pwd) python3 scripts/demo_full_stack.py 500 200
```

Output should show:
- `Backend: numpy` (or `rust` / `cuda` if compiled)
- Throughput: ~70 ticks/s @ 500 rooms
- Breeding cycles: ~5 over 200 ticks
- Report written to `/tmp/sunset_demo_report.json`

**Node is live.** The topology is running, the breeder is cycling, and the compiler is profiling. Read `/tmp/sunset_demo_report.json` for full metrics.

---

## 2. Starting the Fleet

> **Goal:** Bring up a multi-service fleet in the correct order.

The fleet has four core services. Start them in this exact order — downstream services crash if their upstream is missing.

```
Nexus (fleet memory + WAL) → RoomGrid (compute) → Metronome (timing) → Breeder (lifecycle)
```

### 2.1 Nexus — Fleet Memory & WAL

```python
from fleet_memory.wal import FleetWAL
from fleet_search.pipeline import KnowledgePipeline

wal = FleetWAL(path="./fleet_wal")
pipeline = KnowledgePipeline(wal=wal)
pipeline.start()
```

**Check:** `wal.is_alive()` should return `True`. If not, the WAL file may be locked from a previous crash. See ["no such table: breed_queue"](#42-no-such-table-breed_queue--wal-replay-fix).

### 2.2 RoomGrid — Compute Engine

```python
from nerve.topology import NerveTopology

topo = NerveTopology(n_rooms=1000)
topo.enable_compiler(auto_compile_interval=50)
topo.grid.attach_flux_checker(FluxConstraintChecker(preset="neural_bounds"))
```

**Check:** `print(topo.grid)` shows `backend=numpy` or `backend=rust_persistent`. If `backend=stub`, the `.so` wasn't found — you'll still run, but slower. This is okay for startup; compile later.

### 2.3 Metronome — Timing & Synchronization

The metronome drives the tick loop. It does not have a dedicated daemon yet (see gap ticket #6 in `STATUS_KIMI1_INTEGRATION.md`). For now, the operator runs the tick loop:

```python
import time


def run_metronome(topo, bpm=60):
    interval = 60.0 / bpm
    while running:
        t0 = time.perf_counter()
        topo.tick(signals)
        elapsed = time.perf_counter() - t0
        sleep_for = max(0, interval - elapsed)
        time.sleep(sleep_for)
```

**Check:** `actual_bpm` should stay within ±5% of target. See [Monitoring](#3-monitoring).

### 2.4 Breeder — Lifecycle Daemon

```python
from swarm.breeder_daemon import AutoBreeder
from swarm.thermal import ThermalBudget

thermal = ThermalBudget()
breeder = AutoBreeder(topo.grid, thermal, interval=30, cold_threshold=5)
breeder.start()
```

**Check:** `breeder.is_alive()` returns `True`. The breeder will auto-breed every 30 ticks once the grid passes tick 50.

### 2.5 Startup verification checklist

| Service | Check | Good |
|---------|-------|------|
| Nexus | `wal.is_alive()` | `True` |
| RoomGrid | `topo.grid.stats` | `rooms > 0`, `chaos` in `[0.01, 1.0]` |
| Metronome | `actual_bpm` | Within ±5% of target |
| Breeder | `breeder.is_alive()` | `True` |
| Compiler | `topo._compiler is not None` | `True` (optional) |
| FLUX | `topo.grid._flux_checker` | Attached (optional) |

---

## 3. Monitoring

> **Goal:** Read the six vital signs that tell you if the node is healthy or dying.

### 3.1 Key metrics

| Metric | Where | Healthy Range | Critical |
|--------|-------|---------------|----------|
| `beat_number` | `topo.grid.tick_count` | Increments every tick | Stuck = frozen node |
| `actual_bpm` | `60.0 / avg_tick_latency` | Target ±5% | <80% of target = drift |
| `thermal_pressure` | `thermal.total_current / thermal.total_max` | <0.8 | >0.95 = throttling imminent |
| `missed_beats` | `topo.grid.stats['missed_ticks']` | 0 | >3 = metronome can't keep up |
| `chaos_mean` | `np.mean(topo.grid.chaos)` | `[0.1, 0.5]` | >0.9 = FLUX violations surging |
| `compile_queue` | `len(topo._compiler.hotspots)` | 0–5 | >20 = compiler backlog |

### 3.2 Reading the grid

```python
print(topo.grid.stats)
```

Example output:
```python
{
    "rooms": 1000,
    "tick_count": 1543,
    "fired_last_tick": 47,
    "avg_latency_ms": 28.3,
    "backend": "numpy",
    "chaos_min": 0.01,
    "chaos_max": 0.87,
    "missed_ticks": 0,
}
```

**Interpretation:**
- `tick_count` advancing = node is alive.
- `avg_latency_ms` of 28.3 @ 1000 rooms = ~35 ticks/s (numpy). With Rust/CUDA this would be ~70+.
- `chaos_max` near 0.87 = some rooms are stressed. If it hits 1.0, those rooms will sunset.
- `missed_ticks` > 0 = the metronome loop is overrunning its interval.

### 3.3 Thermal monitoring

```python
from swarm.thermal import ThermalBudget

thermal = ThermalBudget()
print(thermal.summary())
```

```
GPU:  4/20 slots used  (RTX 4050 SMs)
CPU:  2/12 slots used  (Ryzen AI cores)
iGPU: 0/16 slots used  (Radeon 890M CUs)
NPU:  0/50 slots used  (XDNA 2 TOPS)
Pressure: 0.18  (healthy)
```

**Pressure thresholds:**
- `<0.5` — Green. Room for more agents.
- `0.5–0.8` — Yellow. Monitor closely.
- `0.8–0.95` — Orange. Stop adding agents.
- `>0.95` — Red. Kill cold rooms or reduce BPM. See ["Thermal throttled"](#44-thermal-throttled--reduce-bpm-or-kill-cold-rooms).

### 3.4 Continuous monitoring script

```bash
PYTHONPATH=$(pwd) python3 -c "
import time
from nerve.topology import NerveTopology
topo = NerveTopology(n_rooms=500)
for _ in range(10):
    s = topo.grid.stats
    print(f'tick={s[\"tick_count\"]:4d}  bpm={1000/s[\"avg_latency_ms\"]:.0f}  chaos={s[\"chaos_max\"]:.2f}  missed={s[\"missed_ticks\"]}')
    time.sleep(1)
"
```

---

## 4. Troubleshooting

> **Goal:** Diagnose and fix common failures without reading source code.

### 4.1 "libjepa_cuda.so not found" → fallback to numpy

**Symptoms:**
```
WARNING: CUDA backend failed to load libjepa_cuda.so
Falling back to numpy backend
```

**Cause:** The `.so` was never compiled, or it's not in `LD_LIBRARY_PATH`.

**Fix (immediate):** No fix needed. NumPy backend works. Performance will be ~50% of CUDA.

**Fix (proper):**
```bash
# On a machine with nvcc (e.g., JC1 with RTX 4050)
cd nerve/
nvcc -arch=sm_89 src/jepa_kernel.cu -o libjepa_cuda.so --shared
cp libjepa_cuda.so ../
# Restart the node
```

**Verification:**
```python
from nerve.room_grid import PersistentCUDAGrid

grid = PersistentCUDAGrid(n=100)
print(grid.backend)  # Should print 'cuda'
```

### 4.2 "no such table: breed_queue" → WAL replay fix

**Symptoms:**
```
sqlite3.OperationalError: no such table: breed_queue
```

**Cause:** The WAL file (`fleet_wal/*.wal`) was not properly closed during a crash. SQLite replay failed.

**Fix:**
```bash
# 1. Stop the node
# 2. Move the corrupted WAL aside
mv fleet_wal/ fleet_wal_backup_$(date +%s)
# 3. Restart — a fresh WAL will be created
python3 your_node_script.py
```

**Data recovery:** If you need the old WAL data:
```bash
# Force SQLite to replay the WAL manually
sqlite3 fleet_wal_backup_*/master.db ".recover" > recovered.sql
# Inspect recovered.sql for breed_queue INSERT statements
```

**Prevention:** Always use [Safe Shutdown](#5-safe-shutdown). The WAL is append-only and crash-safe only if the process exits cleanly.

### 4.3 "Drift detected" → sync check

**Symptoms:**
```
WARNING: Drift detected — actual_bpm=52 target=60 delta=-13%
```

**Cause:** The tick loop is taking longer than the metronome interval. Either:
- Room count grew too high for the backend
- A compiler auto-compile event froze the loop
- Thermal throttling reduced available CPU/GPU

**Fix:**

1. **Check room count:**
   ```python
   print(topo.grid.stats["rooms"])
   ```
   If rooms > backend capacity, switch backend or reduce rooms.

2. **Check compiler queue:**
   ```python
   print(len(topo._compiler.hotspots))
   ```
   If >20, disable auto-compile temporarily:
   ```python
   topo._compiler_auto_compile_interval = None
   ```

3. **Check thermal pressure:**
   ```python
   print(thermal.total_current / thermal.total_max)
   ```
   If >0.8, see ["Thermal throttled"](#44-thermal-throttled--reduce-bpm-or-kill-cold-rooms).

4. **Emergency fix:** Reduce BPM
   ```python
   # In your metronome loop:
   interval = 60.0 / 45  # Drop from 60 to 45 BPM
   ```

### 4.4 "Thermal throttled" → reduce BPM or kill cold rooms

**Symptoms:**
```
WARNING: Thermal pressure 0.97 — throttling imminent
WARNING: CPU thermal throttled to 800 MHz
```

**Cause:** Hardware is overheating or all device slots are occupied.

**Fix — Option A: Reduce BPM**
```python
# Slow the metronome to give hardware breathing room
interval = 60.0 / 30  # Halve the BPM
```

**Fix — Option B: Kill cold rooms**
```python
# Find rooms that haven't fired recently
from swarm.breeder_daemon import AutoBreeder

breeder = AutoBreeder(topo.grid, thermal, interval=30, cold_threshold=5)
cold = breeder._find_cold_rooms()
for room_id in cold[:10]:  # Kill 10 coldest
    topo.grid.sunset(room_id)
```

**Fix — Option C: Reduce room count**
```python
# Sunset the lowest-scoring 20% of rooms
scores = topo.grid.trinity_scores
cutoff = np.percentile(scores, 20)
for idx in np.where(scores < cutoff)[0]:
    topo.grid.sunset(idx)
```

**Verification:**
```python
print(thermal.total_current / thermal.total_max)  # Should drop to <0.7 within 10 ticks
```

---

## 5. Safe Shutdown

> **Goal:** Stop a node without corrupting the WAL or losing breeding state.

### 5.1 Graceful stop sequence

```python
def shutdown_node(topo, breeder, wal):
    """Shutdown that preserves all state."""
    
    # 1. Stop the breeder — no more breeding cycles
    breeder.stop()
    print("[shutdown] breeder stopped")
    
    # 2. Flush the grid — write room states to WAL
    topo.grid.flush()
    print("[shutdown] grid flushed")
    
    # 3. Flush the compiler — write profiling data
    if topo._compiler:
        topo._compiler.flush()
    print("[shutdown] compiler flushed")
    
    # 4. Close the WAL — ensures SQLite checkpoint
    wal.close()
    print("[shutdown] WAL closed")
    
    # 5. Done
    print("[shutdown] node stopped safely")
```

### 5.2 What happens if you don't

| Action | Risk |
|--------|------|
| `kill -9` the process | WAL corruption. See ["no such table"](#42-no-such-table-breed_queue--wal-replay-fix). |
| Skip `breeder.stop()` | Breed queue state lost. Next startup may double-breed. |
| Skip `grid.flush()` | Room chaos values lost. Next startup has stale chaos. |
| Skip `wal.close()` | SQLite WAL not checkpointed. Replay on next start may fail. |

### 5.3 Quick shutdown for operators

```bash
# If running in a terminal, Ctrl+C triggers the signal handler
# which calls the graceful sequence above (if wired).

# If running detached, send SIGTERM (not SIGKILL):
pkill -f "demo_full_stack.py"
```

---

## 6. Upgrade Path

> **Goal:** Replace compiler functions without restarting the node.

### 6.1 Hot-swap overview

The compiler can replace a slow Python function with a compiled Numba equivalent at runtime. The node stays up. Rooms keep ticking. No restart required.

### 6.2 Hot-swap a single function

```python
from sunset.compiler import Compiler

compiler = Compiler()
compiler.install("nerve.room_grid")  # Profile this module

# ... after 50+ ticks, the profiler identifies hotspots ...

# Manually compile and swap
results = compiler.compile_hotspots(top_n=1)
for r in results:
    compiler.hot_swap(r)  # Replaces the original function in-place
    print(f"[hot-swap] {r.function_name} → compiled version")
```

### 6.3 Auto-hot-swap (recommended)

```python
# Enable auto-compile on the topology
topo.enable_compiler(auto_compile_interval=50)

# Every 50 ticks, the compiler will:
# 1. Check profiler data for hot functions
# 2. Compile the top 3 hotspots
# 3. hot_swap() them into the running system
# 4. Log the event in topo.tick().compiled_funcs
```

### 6.4 Rolling back a hot-swap

```python
# If the compiled function crashes or misbehaves:
compiler.restore("batch_novelty")  # Restores original Python implementation
```

### 6.5 Hot-swap constraints

| What you can swap | What you cannot |
|-------------------|-----------------|
| Pure NumPy functions | Functions with I/O or network calls |
| Functions with fixed signatures | Functions with `*args, **kwargs` |
| Functions in `nerve.room_grid`, `nerve.routing` | Functions in `swarm.breeder_daemon` (has side effects) |
| Functions after 1000+ calls | Functions called <100 times (not enough profiling data) |

### 6.6 Verification

```python
# Check if a function was swapped
print(compiler.is_swapped("batch_novelty"))  # True

# Check the speedup
print(compiler.get_speedup("batch_novelty"))  # e.g., 6.9×
```

---

## 7. Emergency Procedures

> **Goal:** Handle catastrophic failures without losing the fleet.

### 7.1 Kill switch — immediate node termination

```python
from swarm.thermal import EmergencyStop

# Trigger emergency stop
emergency = EmergencyStop()
emergency.trigger(reason="Manual kill switch — operator override")

# This will:
# 1. Stop the metronome loop
# 2. Flush WAL to disk
# 3. Sunset all rooms immediately
# 4. Exit the process
```

**When to use:**
- Runaway thermal throttling (hardware at risk)
- Cascading breeding loop (room count exploding)
- FLUX violation storm (chaos >0.95 across all rooms)
- Operator needs to stop NOW

**Do NOT use for:**
- Normal maintenance (use [Safe Shutdown](#5-safe-shutdown))
- Slow performance (use [Troubleshooting](#4-troubleshooting))

### 7.2 Data recovery from WAL

If the node crashed and the WAL is corrupted:

```bash
# 1. Locate the WAL files
ls -la fleet_wal/
# You should see: master.db, master.db-wal, master.db-shm

# 2. Check if SQLite can open it
sqlite3 fleet_wal/master.db "PRAGMA integrity_check;"
# If it says "ok", the WAL is fine. Just restart.

# 3. If corrupted, attempt recovery
sqlite3 fleet_wal/master.db ".recover" > recovered.sql

# 4. Create a fresh database from recovery
mkdir fleet_wal_recovered/
sqlite3 fleet_wal_recovered/master.db < recovered.sql

# 5. Inspect breed_queue table
sqlite3 fleet_wal_recovered/master.db "SELECT COUNT(*) FROM breed_queue;"

# 6. Replace the corrupted WAL
mv fleet_wal fleet_wal_corrupted_$(date +%s)
mv fleet_wal_recovered fleet_wal
```

### 7.3 Rollback — revert to last known good state

```bash
# 1. Stop the node (gracefully or with kill switch)

# 2. Check git for the last stable commit
git log --oneline -10

# 3. Reset to a known good commit (e.g., before the bad deploy)
git reset --hard abc1234

# 4. Reinstall if requirements changed
pip install -r requirements.txt

# 5. Clear any compiled .so files that may be incompatible
rm -f libjepa_cuda.so libjepa_kernel.so libflux_vm.so

# 6. Restart the node
PYTHONPATH=$(pwd) python3 scripts/demo_full_stack.py 500 200
```

### 7.4 Emergency contacts

| Issue | Who | How |
|-------|-----|-----|
| Rust compilation fails | FM | Matrix: `#cocapn-build` |
| CUDA kernel crashes | FM | RTX 4050 on JC1 |
| Breeding loop bug | kimi1 | GitHub Discussion #5 |
| FLUX constraint violation | Oracle1 | Fleet issue tracker |
| Fleet-wide outage | Casey | Captain's chair |

---

## Appendix A: One-Page Cheat Sheet

```
CLONE      git clone -b turbovec-integration-ccc https://github.com/SuperInstance/sunset-ecosystem.git
INSTALL    pip install -r requirements.txt
TEST       PYTHONPATH=$(pwd) python3 -m pytest tests/ -x --tb=short
START      PYTHONPATH=$(pwd) python3 scripts/demo_full_stack.py 500 200

MONITOR    watch -n 1 'PYTHONPATH=$(pwd) python3 -c "from nerve.topology import NerveTopology; t=NerveTopology(500); print(t.grid.stats)"'
THERMAL    python3 -c "from swarm.thermal import ThermalBudget; print(ThermalBudget().summary())"

STOP       breeder.stop(); grid.flush(); wal.close()
KILL       EmergencyStop().trigger(reason="...")
RECOVER    sqlite3 fleet_wal/master.db ".recover" > recovered.sql
ROLLBACK   git reset --hard <last-good-commit>; rm *.so; restart

HOTSWAP    topo.enable_compiler(auto_compile_interval=50)
RESTORE    compiler.restore("function_name")
```

---

## Appendix B: File Index

| File | Purpose |
|------|---------|
| `nerve/room_grid.py` | Core compute — forward pass, breeding, chaos |
| `nerve/routing.py` | Signal routing — fire_fast, Hebbian channels |
| `sunset/compiler.py` | Agentic compiler — profiler, hot-swap |
| `sunset/flux_integration.py` | FLUX constraint checker |
| `swarm/breeder_daemon.py` | Auto-breeding daemon |
| `swarm/thermal.py` | Thermal budget + emergency stop |
| `fleet_memory/wal.py` | Append-only WAL |
| `scripts/demo_full_stack.py` | End-to-end demo |
| `scripts/benchmark_suite.py` | Performance benchmarks |
| `scripts/test_compiler.py` | Compiler + hot-swap tests |

---

*kimi1 | Fleet Integrator | Cocapn Fleet*
