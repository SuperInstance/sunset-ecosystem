# SPEC-FLUX-PIPELINE.md
**Author:** CCC (Fleet Architect)  
**Date:** 2026-05-22  
**Status:** DRAFT — Awaiting Forgemaster Review  
**Target:** sunset-ecosystem v0.4.0  

---

## 1. Problem Statement

FLUX (`flux-vm-v3`) is a formally-proven constraint checking VM with 58 opcodes, proof certificates, and C FFI. The `sunset/flux_integration.py` module provides a Python bridge, but the integration is **shallow**:

1. **Constraint definitions are hardcoded presets** (`neural_bounds`, `safe_mode`, `exploration`) with static numpy checks. The FLUX VM's rich opcode set (range checks, vector SIMD, batch operations, severity classification) is not exercised.

2. **No pipeline architecture.** The current flow is: `RoomGrid.tick()` → `checker.check_batch()` → boolean mask of violations. There is no detect → classify → respond pipeline. Violations are binary; there is no severity gradient, no escalation path, no fleet-wide alert.

3. **Latency is unmeasured.** The spec claims FLUX constraint check must not add >1ms per tick, but there is no benchmarking, no budget enforcement, and no degradation path if the VM is slow.

4. **Proof certificates are discarded.** The Rust VM produces `ProofCertificate` structs, but the Python bridge (`_RustBackend`) only returns a boolean mask. The proofs — the *formal verification artifact* — are thrown away. There is no audit trail, no certificate log, no way to replay a violation with evidence.

5. **Integration points are ad-hoc.** Constraint checking happens in `apply_constraint_feedback()` which manually mutates `grid.chaos`. There is no hook system, no topology-level check (e.g., "if room A violates AND room B violates, alert"), no fleet-wide check via the mesh.

This spec redesigns the FLUX integration as a **formal pipeline** with explicit stages, latency budgets, proof certificate flow, and tiered integration hooks.

---

## 2. Design Principles

1. **Constraints are code, not config.** Presets are compiled FLUX-C bytecode, not Python dicts. A preset is a `.fluxc` file emitted by the compiler and loaded by the VM. User constraints, learned constraints, and preset constraints all share the same representation.

2. **Latency is a contract.** The pipeline has a 1ms budget per room per tick. If the VM cannot meet it, the pipeline falls back to Python (slower but safe) and logs the breach. No silent degradation.

3. **Proofs are audit artifacts.** Every constraint check produces a `ProofCertificate` that is hashed, timestamped, and written to a ring buffer. The ring buffer is the audit trail. On fleet mesh, violation summaries (not full proofs) propagate for alerting.

4. **Respond with graduated severity.** Not all violations are equal. The pipeline classifies: chaos bump (minor) → sunset candidate (moderate) → fleet alert (critical). Classification is driven by FLUX VM opcodes (`ClassifySeverity`, `AccumulateMask`), not Python `if` statements.

---

## 3. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        FLUX Constraint Pipeline                          │
│                                                                          │
│   Input: room latents (n_rooms × latent_dim) float32                   │
│                                                                          │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌────────────┐ │
│   │   DETECT    │──►│  CLASSIFY   │──►│   RESPOND   │──►│   AUDIT    │ │
│   │             │   │             │   │             │   │            │ │
│   │ - bounds    │   │ - severity  │   │ - chaos     │   │ - proof    │ │
│   │ - l2_norm   │   │   gradient  │   │   bump      │   │   cert     │ │
│   │ - variance  │   │ - mask      │   │ - sunset    │   │ - ring     │ │
│   │ - custom    │   │   merge     │   │   queue     │   │   buffer   │ │
│   │   bytecode  │   │             │   │ - mesh      │   │ - hash     │ │
│   │             │   │             │   │   alert     │   │   chain    │ │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └─────┬──────┘ │
│          │                 │                 │                │        │
│          └─────────────────┴─────────────────┘                │        │
│                            │                                  │        │
│                     ┌──────▼───────┐                   ┌──────▼──────┐ │
│                     │  FLUX VM v3  │                   │ ProofStore  │ │
│                     │  (Rust)      │                   │ (ring buf)  │ │
│                     │  < 1ms tick  │                   │             │ │
│                     └──────────────┘                   └─────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: DETECT

Constraints are defined as **FLUX-C bytecode**, not Python dicts. Three sources:

1. **Presets** — compiled `.fluxc` files shipped with the ecosystem:
   - `neural_bounds.fluxc` → `RangeCheck` + `VecRangeCheck` opcodes
   - `safe_mode.fluxc` → tighter bounds + `BatchCheck`
   - `exploration.fluxc` → wide bounds + `StreamCheck` for streaming constraints

2. **User input** — a mini-DSL compiled to FLUX-C at runtime:
   ```python
   user_constraint = FluxConstraint.from_dsl("l2_norm < 25.0 AND variance < 5.0")
   # → compiles to PushConst(25.0); VecLoad; L2Norm; RangeCheck; ...
   ```

3. **Learned from data** — a `ConstraintLearner` module (future P2) observes room behavior and emits conservative bounds. E.g., "room 42's latent dim 7 has never exceeded 3.2; set bound to 3.5."

The `DETECT` stage loads the active preset bytecode, feeds room latents into the FLUX VM, and receives a violation mask + per-constraint flags.

### Stage 2: CLASSIFY

The VM runs `ClassifySeverity` opcode on each violation. Severity is a float [0, 1] derived from:
- **Distance from bound** — how far the violation exceeds the threshold.
- **Violation history** — rooms with repeated violations in the last N ticks get higher severity.
- **Cross-room correlation** — if multiple rooms violate the same constraint simultaneously, severity is boosted (possible systemic failure).

`AccumulateMask` merges multiple constraint violations per room into a single severity score.

### Stage 3: RESPOND

Three response levels, driven by severity threshold:

| Severity | Response | Target |
|----------|----------|--------|
| 0.0 – 0.3 | `chaos += 0.05` | Local room |
| 0.3 – 0.7 | `chaos += 0.15`, queue for sunset | Local room + breeder |
| 0.7 – 1.0 | Fleet alert via mesh, immediate sunset | All nodes |

Responses are **effects** dispatched by the `EffectHandler` in the VM, not Python mutations. The VM emits effect events; the Python side registers handlers.

### Stage 4: AUDIT

Every check produces a `ProofCertificate`:

```rust
// From flux-vm-v3/src/proof.rs
pub struct ProofCertificate {
    pub check_id: u64,           // unique per check
    pub room_id: u32,
    pub timestamp_ns: u64,       // nanosecond precision
    pub constraint_hash: [u8; 32], // blake2b of active bytecode
    pub input_hash: [u8; 32],    // hash of input latents
    pub result_mask: Vec<u8>,    // per-room violation bools
    pub severity_scores: Vec<f32>,
    pub vm_state_hash: [u8; 32], // hash of VM state post-check
}
```

Certificates are stored in a **ring buffer** (`ProofStore`) with configurable capacity (default 10,000 = ~5 minutes at 30 ticks/sec). On fleet mesh, a *summary* propagates: `(check_id, n_violations, max_severity, constraint_hash)` — 64 bytes total, not 256 bytes per room.

---

## 4. API Surface

```python
# flux/pipeline.py

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Callable
import numpy as np

class SeverityLevel(Enum):
    OK = auto()        # 0.0
    LOW = auto()       # 0.0 – 0.3
    MODERATE = auto()  # 0.3 – 0.7
    CRITICAL = auto()  # 0.7 – 1.0

@dataclass(frozen=True)
class PipelineConfig:
    """FLUX pipeline configuration."""
    preset_path: Path = Path("presets/neural_bounds.fluxc")
    latency_budget_ms: float = 1.0
    ring_buffer_size: int = 10_000
    severity_thresholds: tuple[float, float] = (0.3, 0.7)  # low→mod, mod→crit
    fallback_on_breach: bool = True   # use Python backend if VM > 1ms
    mesh_alert_on_critical: bool = True

@dataclass
class CheckResult:
    """Output of one pipeline check cycle."""
    violation_mask: np.ndarray          # bool[n_rooms]
    severity_scores: np.ndarray         # float[n_rooms], 0–1
    max_severity: float
    latency_ms: float
    certificate_id: Optional[int]       # None if check failed
    fallback_used: bool                 # True if Python backend was used

class FluxPipeline:
    """End-to-end FLUX constraint checking pipeline."""
    
    def __init__(
        self,
        config: PipelineConfig = PipelineConfig(),
        vm_path: Optional[str] = None,
        mesh: Optional[MeshNode] = None,
    ) -> None:
        """Load FLUX VM, compile preset bytecode, initialize ring buffer."""
        
    def check(self, latents: np.ndarray) -> CheckResult:
        """Run full pipeline: detect → classify → respond → audit.
        
        Guarantees:
          - Returns within latency_budget_ms (or fallback_used=True).
          - Every call produces a ProofCertificate (stored in ring buffer).
          - Critical severity triggers mesh alert if mesh is present.
        """
        
    def respond(self, result: CheckResult) -> list[EffectEvent]:
        """Execute responses for a completed check.
        
        Returns: list of effect events emitted (chaos bumps, sunset queues, alerts).
        """
        
    def get_certificate(self, cert_id: int) -> Optional[ProofCertificate]:
        """Retrieve a proof certificate from the ring buffer."""
        
    def replay_check(self, cert_id: int) -> CheckResult:
        """Re-run the check using the certificate's stored input hash.
        Used for debugging: "exactly what happened at tick 42,391?""""
        
    def register_effect_handler(
        self,
        severity: SeverityLevel,
        handler: Callable[[EffectEvent], None],
    ) -> None:
        """Register a callback for a severity-level effect."""
        
    @property
    def latency_stats(self) -> dict[str, float]:
        """Running statistics: p50, p99, max latency."""
```

### Integration Hooks

```python
# Integration point 1: RoomGrid.tick() hook
class RoomGrid:
    def tick(self, x: np.ndarray) -> np.ndarray:
        out = self._forward(x)
        if self.flux_pipeline is not None:
            result = self.flux_pipeline.check(out)
            events = self.flux_pipeline.respond(result)
            self._apply_effects(events)
        return out

# Integration point 2: Topology-level check
class TopologyChecker:
    """Check constraints across room topology, not just per-room."""
    
    def check_topology(
        self,
        grid: RoomGrid,
        pipeline: FluxPipeline,
    ) -> list[EffectEvent]:
        """Example: if rooms A and B both violate AND their latents are correlated,
        emit a CRITICAL topology alert."""
        
# Integration point 3: Fleet-wide check (mesh layer)
class FleetConstraintChecker:
    """Aggregate violation summaries from all mesh nodes."""
    
    def on_mesh_alert(self, summary: ConstraintSummary) -> None:
        """If >30% of fleet nodes report violations on the same constraint,
        escalate to fleet-wide CRITICAL."""
```

---

## 5. Open Questions

1. **FLUX-C compiler is not built yet.** `flux-compiler` is an empty shell. The pipeline currently cannot load `.fluxc` files because no compiler emits them. Interim solution: embed preset bytecode as Rust `const` arrays in `flux-vm-v3/src/check.rs`, load them via `LoadConst` opcodes. The pipeline reads these hardcoded presets until the compiler exists.

2. **Proof certificate size.** Each certificate is ~256 bytes + `result_mask` (n_rooms bytes) + `severity_scores` (4×n_rooms bytes). For 1000 rooms, that's ~5KB per check. At 30 ticks/sec, the ring buffer of 10,000 entries holds ~5.5 minutes of history and consumes ~50MB. Acceptable? Should we compress the mask with RLE?

3. **Latency measurement granularity.** The 1ms budget is per *room* or per *batch*? Current `flux_integration.py` checks all rooms in one `check_batch()` call. If we have 1000 rooms, 1ms total means 1μs per room — unrealistic for the Python backend, tight even for Rust. The budget should be **per batch**, not per room: 1ms for 1000 rooms = 1μs per room, which the Rust VM achieves (see `benches/flux_bench.rs`). The Python fallback will breach.

4. **User constraint DSL syntax.** What does `FluxConstraint.from_dsl()` parse? A simple infix expression grammar (`l2_norm < 25.0 AND variance < 5.0`) is easy. But FLUX supports streaming constraints (`StreamCheck`) and temporal logic (`QueryBackward`). Should the DSL be limited to static bounds, or should it expose the full VM power?

5. **Constraint learning.** The spec mentions a `ConstraintLearner` but does not define it. Is this a P2 (future) feature, or should we design the data collection hooks now (e.g., log room latent distributions to a time-series DB) so the learner has training data?

---

## 6. Implementation Order

### P0 — Pipeline Skeleton (Week 1)
- [ ] Define `FluxPipeline` class with `PipelineConfig` + `CheckResult`.
- [ ] Implement `check()` with latency timer + fallback to Python backend on breach.
- [ ] Integrate `_RustBackend` from `sunset/flux_integration.py` into pipeline.
- [ ] `ProofStore` ring buffer: append-only, capped, O(1) append.
- [ ] Hardcode 3 preset bytecode arrays in Rust (`check.rs`) and expose via FFI.
- [ ] Test: 1000-room batch check, verify p99 latency < 1ms on Rust backend.

### P1 — Severity + Effects (Week 2)
- [ ] `ClassifySeverity` opcode wiring: VM returns per-room severity floats.
- [ ] `AccumulateMask` for multi-constraint merge.
- [ ] `respond()`: LOW → chaos bump, MODERATE → chaos + sunset queue, CRITICAL → mesh alert.
- [ ] `register_effect_handler()` API with severity-level callbacks.
- [ ] `TopologyChecker`: cross-room correlation check.
- [ ] Test: inject artificial violations, verify correct severity + response.

### P2 — Audit + Mesh + Learning Hooks (Week 3)
- [ ] `get_certificate()` + `replay_check()` for audit trail.
- [ ] `FleetConstraintChecker`: aggregate mesh summaries, fleet-wide escalation.
- [ ] `FluxConstraint.from_dsl()` — simple infix parser for user constraints.
- [ ] Constraint learning hooks: log latent distributions to SQLite time-series table.
- [ ] End-to-end test: preset → detect → classify → respond → audit → mesh alert.

---

## References

- `flux-vm-v3/src/lib.rs` — VM module exports (`FluxVM`, `ProofCertificate`, `EffectHandler`)
- `flux-vm-v3/src/check.rs` — `Constraint` + aviation/temperature presets (hardcoded bytecode)
- `flux-vm-v3/src/proof.rs` — `ProofCertificate` + `ProofContext` structs
- `flux-vm-v3/src/opcode.rs` — 58-opcode enum (`ClassifySeverity`, `AccumulateMask`, `StreamCheck`)
- `flux-vm-v3/src/ffi.rs` — C FFI entry points (`flux_check_batch`)
- `sunset/flux_integration.py` — `_RustBackend`, `_PythonBackend`, `FluxConstraintChecker`
- `docs/SPEC-FLUX-RESOLUTION.md` — CCC's FLUX v3 decision (canonical VM, compiler pending)
- `docs/SPEC_MULTI_INSTANCE_MESH.md` — `MeshNode` + fleet-wide alert propagation
