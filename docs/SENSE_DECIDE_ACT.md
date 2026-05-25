# Sense → Decide → Act Framework

**Version:** v0.1.0  
**Status:** Draft  
**Author:** CCC, Fleet Sense-Decide-Act Architect  
**Branch:** `feature/sense-decide-act`

---

## Purpose

The sunset ecosystem contains 20 operational patterns extracted from 10 fleet repos. Every one of them is a variation of the same distributed loop:

```
SENSE   →   DECIDE   →   ACT
observe      evaluate    execute
```

This document defines the unifying interface and maps every pattern to its SDA role.

---

## Core Abstractions

### `Sense`

```python
class Sense(ABC):
    @abstractmethod
    def observe(self) -> Observation: ...
```

Collects raw state and packages it as an `Observation`.

**Rules:**
- Must be side-effect-free.
- Should complete in < 50 ms.
- Heavy I/O must be cached or delegated.

### `Observation`

```python
@dataclass(frozen=True)
class Observation:
    timestamp: float           # Unix time
    source: str                # "thermal_trap", "hebbian_mesh", etc.
    metrics: dict[str, Any]    # Arbitrary measurements
    severity_hint: str         # "info" | "warning" | "critical"
```

The `severity_hint` is advisory only. The `Decide` stage may override based on policy.

### `Decide`

```python
class Decide(ABC):
    @abstractmethod
    def evaluate(self, observation: Observation) -> Decision: ...
```

Applies policy or rules to an `Observation` and returns a `Decision`.

**Rules:**
- `confidence` must be in `[0.0, 1.0]`.
- Values `< 0.5` cause the `Act` stage to skip (configurable per `SDALoop`).
- Must not raise exceptions; catch and return a low-confidence `Decision` instead.

### `Decision`

```python
@dataclass(frozen=True)
class Decision:
    action_type: str           # "escalate", "route", "gossip", "noop", ...
    confidence: float          # 0.0–1.0
    payload: dict[str, Any]    # Action parameters
    reasoning: str             # Human-readable justification
```

### `Act`

```python
class Act(ABC):
    @abstractmethod
    def execute(self, decision: Decision) -> ActResult: ...
```

Performs the action described by a `Decision`.

**Rules:**
- Must catch its own exceptions and return `success=False` rather than raising.
- Must report `latency_ms` for metrics.
- Should return `new_observations` for feedback loops.

### `ActResult`

```python
@dataclass(frozen=True)
class ActResult:
    success: bool
    latency_ms: float
    side_effects: list[str]
    new_observations: list[Observation]
```

### `Policy`

A rule-based `Decide` implementation:

```python
policy = Policy()
policy.add_rule(
    condition=lambda obs: obs.metrics.get("temp", 0) > 80,
    action_type="cool",
    confidence=0.9,
    reasoning="temperature too high",
)
```

Rules are evaluated in insertion order. First match wins. No match → `noop` with confidence `1.0`.

### `SDALoop`

Orchestrates one or more `SDAPipeline` instances.

```python
loop = SDALoop(confidence_threshold=0.5)
loop.register(sense, decide, act, name="thermal", interval_ms=1000)
loop.tick()  # sense all → decide all → act all
```

Per-pipeline features:
- `interval_ms`: minimum time between ticks (throttling).
- `enabled`: if `False`, pipeline is skipped.
- `get_metrics()`: aggregate statistics (tick count, latencies, decision counts, success rate).

---

## Built-In Pipeline Adapters

The framework ships with thin adapters that wrap existing fleet modules so they can be registered in an `SDALoop` without code changes to the original modules.

### 1. `thermal_monitoring` → `OperationalTrap` + `TrapRegistry`

| Stage | Adapter | Original Module |
|-------|---------|-----------------|
| Sense | `TrapSense` | `TrapRegistry.run_all()` |
| Decide | `Policy` (or custom) | Maps severity to action |
| Act | Custom or `TrapEscalationAct` | `trap.escalate()` |

Example:
```python
from fleet.operational_trap import TrapRegistry, ThermalTrap
from fleet.sense_decide_act import TrapSense, SDALoop

registry = TrapRegistry()
registry.register(ThermalTrap(budget=thermal_budget))

loop = SDALoop()
loop.register(
    TrapSense(registry),
    Policy(),  # rules: critical → escalate, warning → notify
    TrapEscalationAct(),
    name="thermal_monitoring",
    interval_ms=5000,
)
```

### 2. `dispatch_gating` → `GatewayPacing` + `DispatchRouter`

| Stage | Adapter | Original Module |
|-------|---------|-----------------|
| Sense | `_DummySense` or context builder | Gateway state polling |
| Decide | `GatewayDispatchDecide` | `GatewayPacing.get_status()` + `DispatchRouter.route()` |
| Act | Custom dispatcher | Spawns subagent or queues task |

Confidence mapping:
- `OPEN` → `dispatch`, confidence `1.0`
- `HALF_OPEN` → `probe_dispatch`, confidence `0.6`
- `CLOSED` → `defer`, confidence `0.2`

### 3. `mesh_exploration` → `HebbianMeshLayer`

| Stage | Adapter | Original Module |
|-------|---------|-----------------|
| Sense | `HebbianMeshSense` | `mesh.stats`, `mesh.chaos_factor` |
| Decide | `Policy` or custom | Low diversity → increase chaos |
| Act | `HebbianMeshAct` | `mesh.select_peers_for_gossip()`, `mesh.update_affinity()` |

### 4. `flux_constraint` → `FluxPresetLibrary`

| Stage | Adapter | Original Module |
|-------|---------|-----------------|
| Sense | Custom or `_DummySense` | Fleet health metrics |
| Decide | `FluxPresetDecide` | `FluxPresetLibrary.apply_preset()` |
| Act | Custom or `_DummyAct` | Block breeding, alert, or continue |

Preset mapping:
- `FleetHealth` → block/continue based on thermal + liveness
- `DiversityFloor` → warn if population diversity drops
- `ThermalCeiling` → critical block if thermal exceeds hard cap

### 5. `breed_coordination` → `MetronomeBridge` + `MeshVectorTables`

| Stage | Adapter | Original Module |
|-------|---------|-----------------|
| Sense | `BreedCoordinationSense` | `bridge._tick_counter`, `vector_index.get_breedable_pool()` |
| Decide | Custom | Beat phase 2 + pool > 0 → `dispatch_beat` |
| Act | `BreedCoordinationAct` | `bridge.on_metronome_beat()`, `vector_index.get_fleet_sync_payload()` |

---

## All 20 Patterns Mapped to SDA Roles

| # | Pattern | Source | Sense | Decide | Act | Tile |
|---|---------|--------|-------|--------|-----|------|
| 1 | **Hebbian Auto-Creation** | `hebbian-router` | `HebbianMeshSense` — peer affinities, diversity score | Policy: low diversity → increase `chaos` | `HebbianMeshAct` — route with chaos injection | "How Agents Make Friends" |
| 2 | **Centroid Novelty** | `vector-novelty` | `FleetVectorIndex.get_novelty_score()` | Policy: novelty < floor → trigger exploration | N/A (metric only) | "The Geometry of Originality" |
| 3 | **Buffered Batch Flush** | `cocapn-plato` | Queue depth, last flush timestamp | Policy: depth > threshold OR age > max_age → flush | Batch flush execution | "Async Queues for Busy Crabs" |
| 4 | **Operational Monitor Trap** | `cocapn-traps` | `TrapSense` — run all traps | Policy: critical count > 0 → escalate | `TrapEscalationAct` — log, notify, A2A | "The Art of the Alert" |
| 5 | **Hot-Swap Pipeline** | `agentic-compiler` | Room grid health, compilation status | Policy: test failure rate > 5% → rollback | `HotSwapAct` — apply/rollback kernel | "Surgery on Running Code" |
| 6 | **Pluggable Registry** | `ccc-os` | Plugin load state, dependency graph | Policy: missing dependency → defer load | Registry mutation — enable/disable plugin | "Plugin Architecture for Nervous Systems" |
| 7 | **Grammar Safety** | `cocapn` core | A2A message validation results | Policy: unsafe opcode detected → reject | Reject message, log violation | "Whitelisting in a Wild Fleet" |
| 8 | **SSE Stream** | `cocapn` server | Event queue depth, subscriber count | Policy: queue > 100 OR subscriber dropped → compact | SSE emit / compaction | "Push, Don't Poll" |
| 9 | **State-Transition Emitter** | `cocapn-health` | Grid state changes, health transitions | Policy: state change matches watch list → emit | Emit event to SSE / WAL | "Events That Matter" |
| 10 | **Lambda-Operator Query** | `cocapn-plato` | Query parameters, vector filter criteria | Policy: query complexity > threshold → use index | Execute query, return results | "Query Engines as Poetry" |
| 11 | **Reverse-Actualization** | Essay #7 | Build order queue, dependency readiness | Policy: all deps ready → promote from speculative | Apply build order | "The Cathedral is Empty, But We Heard the Music" |
| 12 | **Two-Minute Test** | Essay #4 | Task description, file count, keyword analysis | `GatewayDispatchDecide` — estimate duration vs 120s threshold | Route to direct / subagent / deferred | "The Scout Does Not Carry the Mountain" |
| 13 | **Shed & Cathedral** | Essay #6 | Scope metrics: file count, test count, doc count | Policy: scope exceeds cathedral threshold → flag for shed-first | Log warning, adjust milestone | "Build the Shed First" |
| 14 | **Behavioral Synthesis** | behavioral_synthesis.md | Agent role telemetry, output quality scores | Policy: role drift detected → re-synthesize | Update agent prompt / capabilities | "The Cartographer's Compass" |
| 15 | **Gateway Pacing** | Scout's Dilemma | `GatewayPacing.get_status()` — consecutive timeouts, backoff | `GatewayDispatchDecide` — OPEN/HALF_OPEN/CLOSED → dispatch/probe/defer | Record success/timeout/failure back to gateway | "Confusion Is Signal" |
| 16 | **Beta-Test Personas** | behavioral_synthesis.md | Persona engagement metrics, completion rate | Policy: completion < 80% → flag UX friction | Log feedback, create ticket | "Seven Visitors, One Gate" |
| 17 | **FluxPresetLibrary** | flux-compat | Fleet health context (thermal, chaos, diversity) | `FluxPresetDecide` — apply preset, check all constraints | Block, warn, or continue based on results | "The Grammar of Limits" |
| 18 | **OpcodeCapabilityIndex** | flux-vm-v3 | Opcode usage in candidate code | Policy: PYTHON_SAFE check fails → reject candidate | Reject, log violation, suggest safe alternative | "Know Before You Compile" |
| 19 | **ConstraintCompiler** | flux-compiler-v0.1.0 | Compilation unit stats, opcode trace | Policy: compilation error rate > threshold → fallback to interpreter | Switch backend to Python interpreter | "From Intent to Silicon" |
| 20 | **PurplePincherVessel** | flux-research | A2A identity state, signature validity | Policy: signature mismatch or expiry → rotate identity | Generate new keypair, propagate to peers | "The Shell Outlives the Crab" |

---

## Integration with `FleetConductor`

`FleetConductor` will instantiate an `SDALoop` and call `tick()` on every beat:

```python
# nexus/fleet_conductor.py (future integration)
from fleet.sense_decide_act import SDALoop

class FleetConductor:
    def __init__(self, ...):
        self.sda = SDALoop(confidence_threshold=0.5)
        # Register built-in pipelines here
        ...

    def on_beat(self, beat_state):
        # Existing beat logic
        ...
        # Run SDA loop
        self.sda.tick()
```

Future: agents themselves can be `SDALoop` instances (self-monitoring).

---

## Metrics

`SDALoop.get_metrics()` returns:

| Key | Type | Description |
|-----|------|-------------|
| `tick_count` | int | Total `tick()` calls |
| `pipeline_ticks` | dict[str, int] | Per-pipeline tick counts |
| `mean_latency_ms` | float \| None | Average pipeline latency |
| `max_latency_ms` | float \| None | Worst-case pipeline latency |
| `decision_counts` | dict[str, int] | Frequency of each `action_type` |
| `act_success_rate` | float | Fraction of successful Act executions |
| `total_pipelines` | int | Number of registered pipelines |

---

## Thread Safety

- `SDALoop` uses an internal lock for pipeline registry and metrics mutation.
- `Sense.observe()` must be side-effect-free (safe to call concurrently).
- `Decide.evaluate()` must not mutate shared state.
- `Act.execute()` may mutate shared state but must be internally synchronized.

---

## Future Work

1. **Agent self-monitoring**: Each fleet agent runs its own `SDALoop` with personalized Sense/Decide/Act implementations.
2. **Distributed consensus**: `HolonomyConsensus` as a `Decide` implementation for quorum-based decisions.
3. **WAL logging**: Every `Observation`, `Decision`, and `ActResult` written to `SignedWAL` for audit trails.
4. **A2A task scheduling**: Slow `Act` stages dispatched as A2A tasks via `A2AMetronomeTasks`.

---

*CCC, Fleet Sense-Decide-Act Architect | "The fleet is a constellation of the same architecture wearing different hats."*
