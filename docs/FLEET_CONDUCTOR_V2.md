# Fleet Conductor V2 — Architecture Guide

**Status**: Implementation complete  
**Branch**: `feature/fleet-conductor-v2`  
**Replaces**: `nexus/fleet_conductor.py` (kept for backward compatibility)

---

## What It Is

FleetConductorV2 is the **central nervous system** of the Cocapn Fleet. Where V1 only synchronized metronome beats across nodes, V2 orchestrates every subsystem we've built this session:

| Subsystem | Responsibility | Module |
|-----------|---------------|--------|
| MetronomeBridge | Cross-node beat sync + drift correction | `nerve/distributed_metronome_bridge.py` |
| FleetVectorIndex | Federated mesh tables for agent vectors | `swarm/mesh_vector_tables.py` |
| TrapRegistry | Fleet health monitoring & alerting | `fleet/operational_trap.py` |
| FluxPresetLibrary | FLUX constraint presets for breeding gating | `sunset/flux_preset_library.py` |
| AgentRegistry | A2A identity, discovery, task negotiation | `logos/a2a_identity.py` |
| GatewayPacing | Circuit breaker for subagent dispatch | `fleet/gateway_pacing.py` |
| SDALoop | Sense→Decide→Act pipelines for all modules | `fleet/sense_decide_act.py` |
| BreederDaemonV2 | Lifecycle FSM for agent breeding (optional) | `swarm/breeder_daemon_v2.py` |

---

## Core Loop: `beat()`

Every conductor tick runs in strict order:

```
beat_number N:
  1. tick metronome          → local_beat_count++, sync every 4th beat
  2. run SDA pipelines       → sense all, decide all, act all
  3. sync mesh tables        → compress fleet vectors for gossip
  4. run operational traps   → check thermal, flux, crashes
  5. log fleet status        → snapshot to in-memory ring buffer
```

This ordering is intentional: sense happens after the metronome tick so SDA observations include fresh beat state, and traps run after mesh sync so they can inspect the latest cross-node data.

---

## Lazy Initialization

No subsystem is instantiated unless:

1. Its `enable_*` flag is `True` in `ConductorConfig`
2. Something actually calls `_get_<subsystem>()` or `beat()` touches it

This keeps startup overhead minimal and prevents import storms in environments where numpy/cryptography aren't available.

```python
cfg = ConductorConfig(
    enable_mesh=False,      # never starts FleetVectorIndex
    enable_traps=False,     # never starts TrapRegistry
    enable_metronome=True,  # starts on first beat
)
```

---

## Health Checks & Auto-Restart

Each subsystem is wrapped in a `SubsystemWrapper` that tracks:

- `state`: healthy | degraded | failed | disabled
- `consecutive_failures`: count of failed health checks
- `last_error`: most recent exception string

If `config.auto_restart=True` and a subsystem fails 3+ consecutive times, the conductor:

1. Computes exponential backoff: `base * 2^failures` (capped at `restart_backoff_max`)
2. Sleeps the backoff duration
3. Calls `wrapper.restart()` → destroy + re-initialize

The wrapper uses `threading.RLock` so restart (which calls destroy internally) doesn't deadlock.

---

## Dispatch: `spawn_agent()`

```python
result = conductor.spawn_agent({
    "description": "Research frontier hardware trends",
    "fn": my_research_function,
    "args": [3, 5],
    "kwargs": {"depth": "deep"},
})
```

Flow:

1. **GatewayPacing check** — if circuit is OPEN, proceed; if CLOSED, queue the task
2. **Routing** — DispatchRouter decides `direct` (in-process) or `subagent` (delegated)
3. **Execution** — direct tasks run `task_spec["fn"](*args, **kwargs)` immediately
4. **Pacing feedback** — success records via `record_success()`, timeout via `record_timeout()`

If the circuit is CLOSED, tasks land in an in-memory queue (max 500) and can be drained later with `conductor.drain_queue(max_tasks=10)`.

---

## Node Registration

```python
conductor.register_node({
    "node_id": "Oracle1",
    "agent_cards": {
        "oracle1-main": { ... AgentCard dict ... },
    }
})
```

This:
- Adds the node to the MetronomeBridge peer list
- Inserts a placeholder vector entry in FleetVectorIndex
- Registers discovered agent cards in AgentRegistry
- Returns `agents_discovered` count + mesh stats

---

## Configuration

```python
from nexus.fleet_conductor_v2 import ConductorConfig

cfg = ConductorConfig(
    node_id="CCC",
    bpm=120.0,
    peers=["Oracle1", "FM"],
    thermal_limits={"GPU": 8.0, "CPU": 16.0},
    flux_preset_name="FleetHealth",
    enable_traps=True,
    enable_mesh=True,
    enable_metronome=True,
    enable_flux_presets=True,
    enable_identity=True,
    enable_gateway_pacing=True,
    enable_sda_loop=True,
    enable_breeding=False,       # set True to wire BreederDaemonV2
    sda_interval_ms=1000.0,
    max_drift_ms=10.0,
    auto_restart=True,
    restart_backoff_base=1.0,
    restart_backoff_max=60.0,
)

conductor = FleetConductorV2(config=cfg)
conductor.start()
```

---

## Status Snapshot

```python
status = conductor.get_status()
```

Returns:
- `node_id`, `uptime_seconds`, `beat_count`
- `subsystems`: per-subsystem health + their `get_status()` output
- `nodes`: known peer IDs
- `agents`: discovered agent IDs
- `drift_ms`: current metronome drift
- `diversity`: fleet vector population size
- `health`: overall fleet health (healthy / degraded / critical)
- `queued_tasks`: number of tasks waiting for dispatch

---

## Integration Notes

### With BreederDaemonV2

When `enable_breeding=True`, the conductor instantiates `BreederDaemonV2` on start and injects the mesh index + flux preset name. The breeder becomes available as `conductor._get_breeder()`.

### With Existing V1 Conductor

`nexus/fleet_conductor.py` is **not deleted** — it remains importable for any code that hasn't migrated. V2 lives in `nexus/fleet_conductor_v2.py`. Migration path:

```python
# Old
from nexus.fleet_conductor import FleetConductor

# New
from nexus.fleet_conductor_v2 import FleetConductorV2, ConductorConfig
```

### All Modules Communicate Through SDA

Direct coupling between subsystems is discouraged. Instead, each module exposes a Sense/Decide/Act adapter (see `fleet/sense_decide_act.py` built-ins: `TrapSense`, `GatewayDispatchDecide`, `FluxPresetDecide`, etc.). The conductor auto-wires these on SDA loop creation.

---

## Thread Safety

- `FleetConductorV2` uses `threading.RLock` for all state mutations
- `beat()` is safe to call from multiple threads concurrently
- `spawn_agent()` is safe from multiple threads
- Each `SubsystemWrapper` has its own `RLock` for independent lifecycle

---

## Testing

40 tests in `tests/test_fleet_conductor_v2.py` covering:

- Initialization & config defaults
- `start()` brings up all subsystems
- `beat()` ordering (metronome → SDA → mesh → traps → status)
- `get_status()` includes all subsystem states
- `spawn_agent()` respects GatewayPacing (open/closed)
- `spawn_agent()` routes direct vs subagent
- `register_node()` adds peers and discovers agents
- `shutdown()` stops everything gracefully
- Lazy init: disabled subsystems never start
- Auto-restart with backoff
- Concurrent beats from 8 threads × 10 beats
- Queue/drain task management
- SubsystemWrapper health degradation → failure escalation

Run: `python3 -m pytest tests/test_fleet_conductor_v2.py -q`

---

## Files Added

| File | Size | Purpose |
|------|------|---------|
| `nexus/fleet_conductor_v2.py` | ~620 lines | Conductor implementation |
| `tests/test_fleet_conductor_v2.py` | ~650 lines | 40 tests, all passing |
| `docs/FLEET_CONDUCTOR_V2.md` | ~260 lines | This architecture guide |

---

*FleetConductorV2 — the nervous system. Everything else is a limb.*
