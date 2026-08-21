# FLUX Preset Library

A catalog of reusable FLUX constraint presets for the Cocapn Fleet's breeding decisions. Each preset is a named bundle of constraint callables, metadata, and required opcodes — validated against the `OpcodeCapabilityIndex` so that only `PYTHON_SAFE` opcodes are used from Python.

**Status:** ⭐⭐⭐⭐☆ P2 Fleet Program
**Branch:** `feature/flux-preset-library`
**Integration:** `BreederDaemonV2.attach_flux_gating()` · `OperationalTrap`

---

## Architecture

```
┌──────────────────────┐
│  FluxPresetLibrary   │
│  (registry)          │
├──────────────────────┤
│  - get_preset()      │
│  - list_presets()    │
│  - apply_preset()    │
│  - suggest_preset()  │
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    │  FluxPreset  │  ← name, category, constraints[], required_opcodes, python_safe
    └─────────────┘
           │
    ┌──────┴──────┐
    │  constraint callable(ctx) → dict
    │  (pure Python, PYTHON_SAFE opcodes only)
    └─────────────┘
```

---

## Preset Catalog

### 1. `RangeCheck`

| Field | Value |
|-------|-------|
| **Category** | `breeding` |
| **Description** | Weight bounds, chaos limits, and thermal budget range checks. |
| **Constraints** | Weight bounds, chaos limit, thermal budget |
| **Opcodes** | `RangeCheck`, `Validate`, `ClassifySeverity` |
| **Python safe** | ✅ Yes |

**Example:**
```python
lib = FluxPresetLibrary()
results = lib.apply_preset(
    "RangeCheck",
    ctx={
        "weights": 2.5,
        "chaos": 0.3,
        "thermal_headroom": 0.8,
    },
)
# → [{"passed": True, "severity": "info", ...}, ...]
```

---

### 2. `ProveAndHashCommit`

| Field | Value |
|-------|-------|
| **Category** | `crypto` |
| **Description** | Signature verification and hash commitment for provenance. Pure-Python fallbacks (`EmitEvent`) until Rust `Prove`/`HashCommit` FFI is ready. |
| **Constraints** | Signature verification, hash commitment |
| **Opcodes** | `EmitEvent` |
| **Python safe** | ✅ Yes (fallback) |

**Note:** The real Rust opcodes `Prove` (0x19) and `HashCommit` (0x1d) are `RUST_ONLY`. This preset uses pure-Python SHA-256 and emits events. When the Rust FFI is ready, the preset can be upgraded to use the native opcodes.

**Example:**
```python
results = lib.apply_preset(
    "ProveAndHashCommit",
    ctx={
        "payload": "agent_state_v42",
        "signature": "a3f2b8c1...",
    },
)
```

---

### 3. `StreamBatch`

| Field | Value |
|-------|-------|
| **Category** | `batching` |
| **Description** | Batch size ceiling and request-per-second rate limiting. |
| **Constraints** | Batch size limit, rate limit |
| **Opcodes** | `Saturate`, `Validate`, `Min`, `Max`, `Sub` |
| **Python safe** | ✅ Yes |

**Example:**
```python
results = lib.apply_preset(
    "StreamBatch",
    ctx={
        "batch_size": 50,
        "max_batch_size": 64,
        "requests_per_second": 800,
        "max_rps": 1000,
    },
)
```

---

### 4. `MemoryBudget`

| Field | Value |
|-------|-------|
| **Category** | `memory` |
| **Description** | Per-agent memory consumption must stay under a hard cap. |
| **Constraints** | Memory cap |
| **Opcodes** | `RangeCheck` |
| **Python safe** | ✅ Yes |

**Example:**
```python
results = lib.apply_preset(
    "MemoryBudget",
    ctx={
        "memory_mb": 512,
        "memory_cap_mb": 1024,
    },
)
```

---

### 5. `DiversityFloor`

| Field | Value |
|-------|-------|
| **Category** | `diversity` |
| **Description** | Population diversity score must stay above a minimum floor. |
| **Constraints** | Diversity floor |
| **Opcodes** | `Min`, `Abs` |
| **Python safe** | ✅ Yes |

**Example:**
```python
results = lib.apply_preset(
    "DiversityFloor",
    ctx={
        "diversity_score": 0.35,
        "diversity_floor": 0.1,
    },
)
```

---

### 6. `ThermalCeiling`

| Field | Value |
|-------|-------|
| **Category** | `thermal` |
| **Description** | Hard thermal ceiling. Any breach is critical and blocks breeding. |
| **Constraints** | Thermal hard ceiling |
| **Opcodes** | `Validate` |
| **Python safe** | ✅ Yes |

**Integration:** `OperationalTrap` uses this preset for fleet health gating.

**Example:**
```python
results = lib.apply_preset(
    "ThermalCeiling",
    ctx={
        "thermal_headroom": 0.97,
    },
)
# → [{"passed": False, "severity": "critical", ...}] (ceiling = 0.99, strict <)
```

---

### 7. `AgentLiveness`

| Field | Value |
|-------|-------|
| **Category** | `liveness` |
| **Description** | Agent must heartbeat within timeout and must not exceed consecutive failure threshold. |
| **Constraints** | Heartbeat timeout, crash detection |
| **Opcodes** | `Sub`, `Validate`, `EmitEvent` |
| **Python safe** | ✅ Yes |

**Integration:** `OperationalTrap` uses this preset for agent health monitoring.

**Example:**
```python
results = lib.apply_preset(
    "AgentLiveness",
    ctx={
        "last_heartbeat": time.time() - 15,
        "heartbeat_timeout_seconds": 30,
        "now": time.time(),
        "consecutive_failures": 1,
        "crash_threshold": 3,
    },
)
```

---

### 8. `CrossNodeSync`

| Field | Value |
|-------|-------|
| **Category** | `sync` |
| **Description** | Local state hash must match gossiped state hash from peers. |
| **Constraints** | Mesh gossip consistency |
| **Opcodes** | `Sub`, `Validate` |
| **Python safe** | ✅ Yes |

**Example:**
```python
results = lib.apply_preset(
    "CrossNodeSync",
    ctx={
        "local_hash": "sha256:abc123...",
        "gossip_hash": "sha256:abc123...",
    },
)
```

---

### 9. `BreedingStandard` *(Composite)*

| Field | Value |
|-------|-------|
| **Category** | `breeding` |
| **Description** | Default breeding gate: weights, chaos, thermal, and diversity. |
| **Constraints** | Weight bounds, chaos limit, thermal budget, diversity floor |
| **Opcodes** | `RangeCheck`, `Validate`, `ClassifySeverity`, `Min`, `Abs` |
| **Python safe** | ✅ Yes |

**Example:**
```python
results = lib.apply_preset(
    "BreedingStandard",
    ctx={
        "weights": 3.0,
        "chaos": 0.2,
        "thermal_headroom": 0.7,
        "diversity_score": 0.5,
    },
)
```

---

### 10. `FleetHealth` *(Composite)*

| Field | Value |
|-------|-------|
| **Category** | `thermal` |
| **Description** | Fleet-wide health gate: thermal ceiling and agent liveness. |
| **Constraints** | Thermal ceiling, heartbeat timeout, crash detection |
| **Opcodes** | `Validate`, `Sub`, `EmitEvent` |
| **Python safe** | ✅ Yes |

**Example:**
```python
results = lib.apply_preset(
    "FleetHealth",
    ctx={
        "thermal_headroom": 0.95,
        "last_heartbeat": time.time() - 5,
        "heartbeat_timeout_seconds": 30,
        "now": time.time(),
        "consecutive_failures": 0,
        "crash_threshold": 3,
    },
)
```

---

## Suggestion Engine

`FluxPresetLibrary.suggest_preset_for_task(task_description)` uses keyword matching to recommend a preset. If no keywords match, it falls back to `BreedingStandard`.

| Keywords | Preset |
|----------|--------|
| `range`, `bound`, `weight`, `norm`, `check` | `RangeCheck` |
| `prove`, `hash`, `commit`, `signature`, `verify`, `crypto` | `ProveAndHashCommit` |
| `batch`, `stream`, `rate`, `limit`, `rps`, `throughput` | `StreamBatch` |
| `memory`, `ram`, `heap`, `budget`, `cap` | `MemoryBudget` |
| `diversity`, `variety`, `floor`, `population` | `DiversityFloor` |
| `thermal`, `heat`, `temperature`, `ceiling`, `throttle` | `ThermalCeiling` |
| `liveness`, `heartbeat`, `crash`, `alive`, `dead`, `health` | `AgentLiveness` |
| `sync`, `gossip`, `mesh`, `consistency`, `node`, `peer` | `CrossNodeSync` |
| `breed`, `breeding`, `spawn`, `new agent`, `tournament` | `BreedingStandard` |
| `fleet`, `health`, `system`, `overall`, `status` | `FleetHealth` |

---

## Integration Points

### `BreederDaemonV2.attach_flux_gating()`

```python
from sunset.flux_preset_library import FluxPresetLibrary
from swarm.flux_gating import PythonFluxFallback, FluxGatingConfig

lib = FluxPresetLibrary()
preset_name = lib.suggest_preset_for_task("breed with weight and chaos checks")
# → "BreedingStandard"

# The daemon can use the preset's constraint logic via PythonFluxFallback,
# or extract config bounds from the context schema.
checker = PythonFluxFallback(
    FluxGatingConfig(
        weight_bounds=(0.0, 5.0),
        chaos_limit=0.5,
        thermal_budget_limit=0.95,
    )
)
breeder.attach_flux_gating(checker)
```

### `OperationalTrap`

```python
# OperationalTrap uses ThermalCeiling + AgentLiveness presets
# to gate fleet operations when health degrades.
thermal_results = lib.apply_preset("ThermalCeiling", ctx={"thermal_headroom": ...})
liveness_results = lib.apply_preset("AgentLiveness", ctx={"last_heartbeat": ...})
```

---

## Opcode Safety

Every preset's `required_opcodes` are cross-checked at registration time against the `OpcodeCapabilityIndex`. If a preset claims `python_safe=True` but requires a `RUST_ONLY` opcode, the library auto-corrects the flag to `False`.

Current `PYTHON_SAFE` opcodes used by presets:
- `RangeCheck`, `Validate`, `ClassifySeverity`
- `Saturate`, `Min`, `Max`, `Sub`
- `EmitEvent`
- `Nop`

`RUST_ONLY` opcodes **not** used by any preset:
- `Prove`, `HashCommit`, `VecLoad`, `ParDispatch`, `StreamOpen`, `BatchCheck`

---

## File Manifest

| File | Role |
|------|------|
| `sunset/flux_preset_library.py` | Core library: `FluxPreset`, `FluxPresetLibrary`, constraint callables |
| `tests/test_flux_preset_library.py` | 34 tests covering presets, lookup, filtering, application, suggestion, opcode safety |
| `docs/FLUX_PRESET_LIBRARY.md` | This document |

---

*Catalog version: v1.0 — aligned with FLUX-C v3 opcode audit and Path A constraint library.*
