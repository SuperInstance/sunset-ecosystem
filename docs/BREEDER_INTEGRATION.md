# BreederDaemonV2 Fleet Integration Guide

## Overview

This guide documents how the 5 new fleet modules are wired into `BreederDaemonV2` to make the breeding loop fleet-aware.

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐
│ MetronomeBridge │────▶│ BreederDaemonV2      │
│   (tick)        │     │   .auto_breed()      │
└─────────────────┘     │                      │
┌─────────────────┐     │  ┌────────────────┐  │
│FleetVectorIndex │────▶│  │ select_parents │  │
│ get_breedable() │     │  │ (cross-node)   │  │
└─────────────────┘     │  └────────────────┘  │
┌─────────────────┐     │                      │
│FluxPresetLibrary│────▶│  apply_preset()      │
│ suggest_preset  │     │  (FLUX gating)       │
└─────────────────┘     │                      │
┌─────────────────┐     │  ┌────────────────┐  │
│  AgentIdentity   │────▶│  │ sign_task()    │  │
│   (sign WAL)    │     │  │ (breed record) │  │
└─────────────────┘     │  └────────────────┘  │
┌─────────────────┐     │                      │
│  TrapRegistry    │────▶│  run_all()           │
│  (health check)  │     │  (post-breed)        │
└─────────────────┘     └──────────────────────┘
```

## Constructor Parameters

```python
BreederDaemonV2(
    grid=grid,
    thermal=thermal,
    # ... existing params ...
    metronome_bridge=Optional[MetronomeBridge],
    fleet_vector_index=Optional[FleetVectorIndex],
    trap_registry=Optional[TrapRegistry],
    flux_preset_library=Optional[FluxPresetLibrary],
    agent_identity=Optional[AgentIdentity],
)
```

All 5 params are **optional** — the daemon works fine without any of them.

## Wiring Details

### 1. Metronome Sync

**When**: At the start of every `auto_breed()` / `cycle()` call.

**What**: Calls `metronome_bridge.tick()` to synchronize with the fleet-wide beat.

**Fail-safe**: If `tick()` raises, the exception is logged but breeding continues.

### 2. Cross-Node Parent Selection

**When**: Inside `select_parents()`.

**What**: If `fleet_vector_index` is attached, calls `get_breedable_pool()` and merges the returned `VectorTableEntry` objects with local candidates. Deduplication is by numeric `agent_id`.

**ID format**: Fleet entries use `"node2::agent_42"` format. The numeric suffix is extracted and parsed as `int`.

**Fail-safe**: If `get_breedable_pool()` raises, falls back to local candidates only.

### 3. FLUX Preset Gating

**When**: At the start of every `auto_breed()` / `cycle()` call, right after metronome tick.

**What**: 
1. `flux_preset_library.suggest_preset_for_task("breeding")` → preset name
2. `flux_preset_library.apply_preset(preset_name, context)` → applies constraints

**Replaces**: The old `FluxGatingChecker.attach_flux_gating()` / `breed_cycle()` API.

**Fail-safe**: If either call raises, the exception is logged and breeding continues without preset constraints.

### 4. Agent Identity Signing

**When**: After each breed transition that reaches `COMPETE` state.

**What**: Calls `agent_identity.sign_task(payload)` where payload includes:
- `task: "breed"`
- `agent_id`
- `parent_a`, `parent_b`
- `generation`
- `preset` (name of applied preset, if any)

**Storage**: Signature is stored in `daemon._breed_signatures[agent_id]`.

**Fail-safe**: If `sign_task()` raises, the exception is logged and the transition is not aborted.

### 5. Operational Traps

**When**: At the end of every `auto_breed()` / `cycle()` call, after all breed transitions.

**What**: Calls `trap_registry.run_all()` to run health checks (diversity collapse, stale agents, etc.).

**Fail-safe**: If `run_all()` raises, the exception is logged.

### 6. Fleet Status

**Method**: `daemon.get_fleet_status()` → `dict[str, Any]`

Returns a unified status dict including:
- `running`, `tick_count`, `agent_count`, `diversity_score`
- Boolean flags for each attached module
- Module-specific metrics (beat count, pool size, trap result count, suggested preset, agent name)

## Migration from Old API

| Old API (V1) | New API (V2) |
|---|---|
| `attach_flux_gating(checker)` | Pass `flux_preset_library` to constructor |
| `breed_cycle(n_winners=2)` | `auto_breed(n_winners=2)` or `cycle(...)` |
| `tournament_select(population)` | `select_parents(n_children)` (diversity-aware) |
| `cold_threshold` kwarg | Ignored ( absorbed by `**kwargs` for backward compat) |
| `flux_config` kwarg | Ignored (use `flux_preset_library` instead) |

## Test Coverage

See `tests/test_breeder_integration.py` (23 tests):

- **Initialization** — all / none / partial modules attached
- **Metronome** — ticked during cycle, exception logged, absent = no crash
- **Fleet Index** — queried for parents, merged with local, exception fallback, absent fallback
- **Flux Preset** — applied during cycle, exception logged, absent = no crash
- **Agent Identity** — signs WAL entries, stores signatures, exception logged
- **Trap Registry** — runs after cycle, exception logged, absent = no crash
- **Fleet Status** — includes all modules, handles missing, pools fleet size
- **Thread Safety** — 5 concurrent cycles, no errors, correct call counts

## Running Tests

```bash
cd sunset-ecosystem
python3 -m pytest tests/test_breeder_integration.py -v
```

Full suite:
```bash
python3 -m pytest tests/test_flux_gating.py tests/test_breeder_integration.py -v
```

Expected: `14 passed, 4 xfailed` (flux gating legacy tests) + `23 passed` (integration tests).
