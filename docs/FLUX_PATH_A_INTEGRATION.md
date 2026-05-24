# FLUX Path A Integration — BreederDaemonV2

## What is Path A?

**Path A** treats FLUX as a **constraint library**, not a full VM.

The FLUX VM audit (`docs/FLUX_OPCODE_ALIGNMENT.md`) found 60 Rust opcodes
and **zero** used from Python.  Path B (a full VM compiler) is high effort
and blocked on a Fleet Manager decision.  Path A sidesteps the compiler
entirely by calling `flux_check_batch()` via the existing FFI (or a pure-Python
fallback), getting back pass/fail/severity, and using those results to gate
breeding decisions.

| Path | Scope | Effort | Status |
|------|-------|--------|--------|
| A | Library call (`check_candidate`, `check_batch`) | Low | **Implemented** |
| B | Full VM compiler (Rust opcodes → Python bytecode) | High | Blocked on FM |

## Interface Contract

The contract between the breeder and FLUX is defined by `FluxGatingChecker`:

```python
class FluxGatingChecker:
    def check_candidate(parent_idx, mutation_plan) -> GatingResult
    def check_batch(candidates) -> list[GatingResult]
    def score_for_breeding(room_vector, room_metadata) -> float  # 0.0–1.0
    def record_violation(room_id, constraint_id, severity, context) -> None
```

* `GatingResult.passed` — `True` = candidate is acceptable.
* `GatingResult.score` — 1.0 = perfectly clean, 0.0 = blocked.
* `record_violation` writes to a WAL with `event_type="flux_violation"`.

## Swapping in the Real FFI

When the Rust FFI is ready, swap is a **one-line change**:

```python
# Before (Python fallback)
checker = PythonFluxFallback(config)
breeder.attach_flux_gating(checker)

# After (Rust FFI — same API)
checker = RustFluxChecker(config)  # wraps flux_check_batch() FFI call
breeder.attach_flux_gating(checker)
```

No changes to `BreederDaemonV2`, `tournament_select`, or `breed_cycle` are
required because they only call methods on the abstract `FluxGatingChecker`
interface.

## Where Gating is Wired

### 1. `breed_cycle()` — per-candidate gate

Before creating a child from a tournament winner, the daemon calls:

```python
result = flux_checker.check_candidate(room_idx, mutation_plan)
if not result.passed:
    logger.warning("FLUX blocked winner %s", winner.agent_id)
    continue
```

This prevents constraint-violating mutations from entering the fleet.

### 2. `tournament_select()` — FLUX score tiebreak

When FLUX is active, winners are re-ranked by `score_for_breeding()`:

```python
flux_score = flux_checker.score_for_breeding(room_vector, room_meta)
```

This breaks ties when two agents have the same tournament win count.

### 3. `run_tick()` — batch check on top-k rooms

After every grid tick, the daemon runs a batch check on the top-k active
rooms:

```python
flux_batch = flux_checker.check_batch(candidates)
```

This provides fleet-wide constraint surveillance without blocking the hot
path.

## Files Added / Modified

| File | Role |
|------|------|
| `swarm/flux_gating.py` | New. Core Path A module: `FluxGatingChecker`, `PythonFluxFallback`, `FluxWAL`. |
| `swarm/breeder_daemon_v2.py` | New. `BreederDaemonV2` with FLUX hooks in lifecycle. |
| `tests/test_flux_gating.py` | New. 8 tests covering fallback, batch, scoring, WAL, wiring. |
| `nerve/room_grid.py` | New (stub). `JEPAGrid` with `attach_flux_checker()` hook. |
| `swarm/tournament.py` | Copied from `flux-compat-work`. Tournament logic. |
| `swarm/thermal.py` | Copied from `flux-compat-work`. Thermal budget. |

## Integration Touchpoints

### `BreederDaemonV2` → `AutoBreeder`

`BreederDaemonV2` is the canonical v2 orchestrator.  It passes the
`FluxGatingChecker` to any downstream `AutoBreeder` instance via
`attach_flux_gating()`.  The checker is stored as `breeder.flux_checker`
and consulted in every breeding decision.

### `MeshVectorGossip` → FLUX scores in vector table metadata

`score_for_breeding()` returns a float that can be stored as a metadata
field in the vector table (e.g. `flux_score`).  This lets gossip
protocols propagate constraint health alongside fitness scores:

```python
vector_table[room_id]["metadata"]["flux_score"] = flux_checker.score_for_breeding(vec, meta)
```

### `WALQuery` → `event_type="flux_violation"`

`FluxWAL` writes records with `event_type="flux_violation"`.  A future
`WALQuery` system can index on this field for post-hoc fleet audits:

```python
violations = wal.query_by_event_type("flux_violation")
```

## Performance

* **Python fallback**: ~1 ms per single-room check (numpy L2 norm + chaos check).
* **Batch amortization**: `check_batch()` runs the same checks in a list
  comprehension; overhead is negligible because numpy operations dominate.
* **Rust FFI target**: Expected ~10–50 µs per check once the Rust kernel is
  wired (batchable via `flux_check_batch`).

## Safe Defaults

`FluxGatingConfig` defaults are conservative:

* `max_violations_per_cycle = 5` — allows up to 5 minor violations before blocking.
* `severity_weights = {"critical": 100, "warning": 10, "info": 1}` — critical
  violations heavily penalize the score.
* `weight_bounds = (0.0, 10.0)` — generous L2 norm window.
* `chaos_limit = 1.0` — effectively disabled by default (all chaos values pass).
* `thermal_budget_limit = 0.95` — breeding throttles only when fleet is 95 % full.
* `enable_wal = True` — violations are always logged.

These defaults ensure the fallback does **not** block normal breeding
unless something is genuinely wrong.

## Branch

`feature/flux-path-a-breeder`
