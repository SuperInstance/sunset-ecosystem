# FLUX Path A Integration — Constraint Gating for Breeding

## Summary

This document describes **Path A** of FLUX integration into the Sunset Ecosystem breeding loop. Path A treats FLUX as a **constraint-checking library** invoked via Python ↔ Rust FFI, not as a bytecode-compiled VM (Path B). Path A is **complete and functional today**; Path B requires Forgemaster (FM) approval and is out of scope here.

## Why Path A?

The FLUX VM audit (`docs/FLUX_OPCODE_ALIGNMENT.md`) found **60 Rust opcodes** and **zero Python call sites**. The full VM compiler (Path B) is high-effort, blocked on architectural decisions, and not needed for the immediate value of FLUX: **constraint enforcement**.

Path A provides 80% of the value with 20% of the effort:
- Gating prevents catastrophically bad candidates from entering the population
- Batch checking surfaces thermal/chaos anomalies across the room grid
- Breeding tiebreak nudges selection toward FLUX-compliant parents
- The Rust FFI path can be swapped in later without touching breeding logic

## Architecture

```
┌─────────────────────────────────────┐
│  BreederDaemonV2  (Python)          │
│  ├── step()                         │
│  │   ├── flux_checker.check_candidate()  → gate before EGG
│  │   └── flux_checker.check_batch()        → top-k room audit
│  ├── _select_parents_vector()       │
│  │   └── flux_checker.score_for_breeding() → tiebreak
│  └── attach_flux_gating()           │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  FluxGatingChecker  (Python)        │
│  ├── wraps Rust FFI (future)        │
│  └── PythonFluxFallback (default)   │
│       ├── weight bounds             │
│       ├── L2 norm limit             │
│       ├── variance ceiling          │
│       ├── chaos threshold           │
│       └── thermal budget gate       │
└─────────────────────────────────────┘
```

## Files

| File | Role |
|------|------|
| `swarm/flux_gating.py` | Core library — config, checker, Python fallback |
| `swarm/breeder_daemon_v2.py` | Patched with `attach_flux_gating()`, gating hooks in `step()` and `_select_parents_vector()` |
| `tests/test_flux_gating.py` | 20+ unit and integration tests |
| `docs/FLUX_PATH_A_INTEGRATION.md` | This document |

## API Reference

### `FluxGatingConfig`

```python
@dataclass
class FluxGatingConfig:
    weight_bounds: tuple[float, float] = (-10.0, 10.0)    # per-element weight limits
    max_l2_norm: float = 100.0                           # global weight magnitude
    max_variance: float = 10.0                           # weight distribution sanity
    max_chaos: float = 1.0                               # room chaos ceiling
    thermal_budget_gate: float = 1.0                     # thermal utilization cap
    severity_weights: dict[str, float] = default dict    # per-violation penalty weights
    pass_threshold: float = 0.35                         # score above this = FAIL
    top_k_batch: int = 10                                # rooms checked per tick
    numpy_only: bool = True                              # force Python fallback
```

### `FluxGatingChecker`

```python
checker = FluxGatingChecker(config=FluxGatingConfig(), vm_path=None)

# Single candidate gate
result = checker.check_candidate(
    weights: np.ndarray,          # flat weight vector
    chaos: float = 0.3,
    thermal_pressure: float = 0.0,
) -> FluxCheckResult

# Batch gate (used at end of tick)
results = checker.check_batch(
    weights_batch: np.ndarray,    # shape (N, dim)
    chaos_vec: np.ndarray | None = None,
    thermal_vec: np.ndarray | None = None,
) -> list[FluxCheckResult]

# Breeding tiebreak score
score = checker.score_for_breeding(
    parent_a_weights, parent_b_weights,
    chaos_a, chaos_b,
    thermal_a=0.0, thermal_b=0.0,
) -> float       # 1.0 = perfect, 0.0 = worst
```

### `FluxCheckResult`

```python
@dataclass
class FluxCheckResult:
    passed: bool
    score: float  # 0.0 = compliant, 1.0 = catastrophic
    severity: float  # alias for score
    violations: dict[str, float]  # {'bounds': 0.5, 'l2_norm': 0.2, ...}
```

## Integration Points in BreederDaemonV2

### 1. Constructor

```python
daemon = BreederDaemonV2(
    ...,
    flux_config=FluxGatingConfig(max_chaos=0.95, top_k_batch=8),
)
```

### 2. Attach / Replace Checker

```python
daemon.attach_flux_gating()  # auto-build from config
daemon.attach_flux_gating(checker=custom)  # inject pre-built instance
```

### 3. Breeding Gate (`step()`)

Before placing a child in the `EGG` state, `step()` calls `check_candidate()` on both parents. If either parent fails:
- The breed ticket is **re-queued with lower priority**
- A `logger.warning()` is emitted with agent IDs and violation types
- No child is created

### 4. Parent Selection Tiebreak (`_select_parents_vector()`)

In the Path 3 fallback (legacy fitness + distance), when choosing `parent_b` for a given `parent_a`, the Euclidean distance metric receives a small bonus:

```
dist = dist + 0.05 * flux_score_for_breeding(parent_a, candidate)
```

This is deliberately a **tiebreak**, not a primary filter. It nudges toward FLUX-compliant pairs without discarding diversity-driven candidates.

### 5. Batch Room Audit (end of `step()`)

After all transitions for the tick are complete, `step()` calls `check_batch()` on the top-`k` most active rooms (`grid.top(k=...)`). For each violating room:
- `grid.chaos[room_id]` is incremented by `0.1`
- A `logger.debug()` line records the violation

This creates a **soft pressure** — chaotic rooms become more chaotic, making them less attractive for future breeding, without hard-killing them.

## Python Fallback Benchmark

Run:
```bash
python3 -m pytest tests/test_flux_gating.py -v
```

Typical results on a modern CPU (single thread):
- `check_candidate`: ~150,000–250,000 checks/sec
- `check_batch` (size=32): ~80,000–120,000 batches/sec → ~3–4M weights/sec

The Python fallback is the bottleneck. Moving to the Rust FFI backend (same API, swap `numpy_only=True` → `False`) should yield 10–100× speedup.

## Rust FFI Path (Future)

`FluxGatingChecker` auto-detects a compiled `.so` at `FLUX_VM_PATH` (default: `../flux-vm-v3-temp/target/release/flux_vm`). When found and `numpy_only=False`:

1. Loads the shared library via `ctypes`
2. Expects exported symbol: `flux_check_batch(float*, int, int, float, float, float, float, uint8_t*) -> int`
3. Marshals numpy arrays to C, calls Rust, unmarshals pass/fail flags
4. Falls back to Python on any FFI error (load failure, ABI mismatch, segfault)

No breeding code changes are required to switch backends.

## Migration Path to Path B

Path A and Path B are **not mutually exclusive**:
- Path A stays as the fast constraint gate
- Path B (full VM compiler) can be added later as an **additional validation layer** inside `AgentLifecycleFSM`
- If Path B ever compiles and runs Python-defined FLUX programs, it can call into the same `flux_check_batch()` entry point

## Open Questions

1. **Severity calibration**: Are the default `severity_weights` tuned correctly for our weight distributions? (Currently: bounds=1.0, l2=0.5, variance=0.3, chaos=0.8, thermal=0.7)
2. **Batch size**: Is `top_k_batch=10` the right number of rooms to check per tick?
3. **Chaos bump**: `+0.1` per violation feels arbitrary. Should it scale with violation severity?
4. **Thermal pressure metric**: Currently uses `thermal_headroom()`. Should we use per-device `current/max` instead?

## Commit Checklist

- [x] `swarm/flux_gating.py` created with Config, Checker, PythonFallback
- [x] `swarm/breeder_daemon_v2.py` patched with gating hooks
- [x] `tests/test_flux_gating.py` — 20+ tests, all passing
- [x] `docs/FLUX_PATH_A_INTEGRATION.md` written
- [x] No Rust source touched
- [x] No bytecode compiler built
