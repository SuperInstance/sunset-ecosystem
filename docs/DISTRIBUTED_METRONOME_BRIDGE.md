# Distributed Metronome Bridge

> Cross-node metronome synchronisation for the Cocapn Fleet.

## Problem

The fleet runs overnight symphonies across multiple nodes. Each node has its own local metronome scheduler. Without synchronisation, the beats drift — rooms tick at different rates, breeding rounds desynchronise, and the symphony turns into noise.

## Solution

`nerve/distributed_metronome_bridge.py` provides a **self-healing, signed, PID-corrected** bridge that keeps every node's metronome aligned.

---

## Architecture

```
┌──────────────┐      ┌──────────────────────┐      ┌──────────────┐
│  Node Alpha  │◄────►│   MeshVectorGossip   │◄────►│  Node Beta   │
│  Scheduler   │      │  "metronome_sync"    │      │  Scheduler   │
└──────┬───────┘      └──────────────────────┘      └──────┬───────┘
       │                                                   │
       ▼                                                   ▼
┌─────────────────┐                              ┌─────────────────┐
│ MetronomeBridge │                              │ MetronomeBridge │
│  • tick()       │                              │  • tick()       │
│  • sync_with_peers()                              │  • sync_with_peers()
│  • receive_sync()│                              │  • receive_sync()│
│  • compute_drift()                               │  • compute_drift()
│  • adjust_bpm() │                              │  • adjust_bpm() │
└─────────────────┘                              └─────────────────┘
       │                                                   │
       ▼                                                   ▼
┌─────────────────┐                              ┌─────────────────┐
│ DriftCorrection │                              │ DriftCorrection │
│  PID (P+I+D)    │                              │  PID (P+I+D)    │
└─────────────────┘                              └─────────────────┘
```

---

## Components

### 1. `UnifiedBeat` (dataclass)

A snapshot of fleet-wide alignment:

| Field | Meaning |
|-------|---------|
| `global_beat_count` | Consensus beat (max of all known peers + local) |
| `local_beat_count` | This node's own counter |
| `drift_ms` | Maximum absolute drift observed |
| `peer_count` | How many peers have sent sync messages |
| `timestamp` | Wall-clock nanoseconds when snapshot was taken |

### 2. `SyncMessage` (dataclass)

Plain JSON-serialisable payload that travels over the wire:

```json
{
  "node_id": "alpha",
  "beat_count": 420,
  "timestamp_ns": 1700000000000000000,
  "bpm": 120.0,
  "signature": "base64..."
}
```

The `payload_for_signing()` method returns the dict *without* the signature so `AgentIdentity.sign_task()` can canonicalise deterministically.

### 3. `DriftCorrection` (PID controller)

| Term | Role |
|------|------|
| **P** (proportional) | `kp * drift_ms` — immediate reaction |
| **I** (integral) | `ki * ∫ drift dt` — corrects persistent bias |
| **D** (derivative) | `kd * d(drift)/dt` — dampens oscillation |

Safety clamps:
- Correction factor always in **0.95 – 1.05**
- Integral wind-up capped at **±500 ms**
- BPM change per step capped at **±5%** (via `MetronomeBridge.adjust_bpm`)

### 4. `MetronomeBridge` (orchestrator)

#### Lifecycle

1. **Initialise** — `MetronomeBridge(local_bpm, node_id, peers, identity)`
2. **Tick** — `tick()` increments local beat counter with nanosecond precision
3. **Sync** — `sync_with_peers()` builds a signed `SyncMessage` per peer and dispatches it
4. **Receive** — `receive_sync(node_id, timestamp, signature)` validates signature, records peer state
5. **Compute** — `compute_drift()` returns max absolute drift across all peers (ms)
6. **Correct** — `maybe_correct_drift()` → if drift > threshold, computes PID factor, nudges BPM

#### Thread Safety

All public methods acquire an internal `threading.Lock()`. Concurrent ticks, syncs, and receives are safe.

#### Integration Hooks

| Hook | Usage |
|------|-------|
| `send_fn` callback | Passed to `__init__`; called as `send_fn(peer_id, payload_dict)` during `sync_with_peers()` |
| `identity` | `AgentIdentity`-like object with `sign_task()` / `verify_task()` |
| `public_key_pem` | Optional per-peer key override for `receive_sync()` |

---

## Fleet Wiring

### FleetConductor

```python
from nerve.distributed_metronome_bridge import MetronomeBridge

class FleetConductor:
    def __init__(self, node_id, ...):
        self.bridge = MetronomeBridge(
            local_bpm=120.0,
            node_id=node_id,
            peers=self.discovered_peers(),
            identity=self.agent_identity,
            send_fn=self._gossip_send,
        )
```

The conductor calls `bridge.tick()` on every scheduler beat and `bridge.sync_with_peers()` at the configured sync interval.

### MeshVectorGossip

Sync messages are carried as gossip payload type `"metronome_sync"`:

```python
gossip.inject({
    "type": "metronome_sync",
    "payload": sync_message.to_dict(),
})
```

On receipt, the gossip layer extracts the payload and calls:

```python
bridge.receive_sync(
    node_id=payload["node_id"],
    timestamp=payload["timestamp_ns"],
    signature=payload["signature"],
    beat_count=payload["beat_count"],
    bpm=payload["bpm"],
)
```

### AgentIdentity

`AgentIdentity.sign_task(payload_dict)` canonicalises the payload (sorted keys, no whitespace), signs with Ed25519, and returns base64.

`AgentIdentity.verify_task(payload_dict, signature, public_key_pem)` decodes base64 and validates against the provided (or local) public key.

### OperationalTrap

Monitors `bridge.get_status()["max_drift_ms"]` every beat. If drift > **50 ms**, raises an alert:

```python
if status["max_drift_ms"] > 50.0:
    operational_trap.alert(
        level="warning",
        source="metronome_bridge",
        message=f"Drift exceeded 50 ms on {node_id}",
    )
```

---

## Test Coverage

`tests/test_distributed_metronome_bridge.py` contains **17 tests** covering:

| Area | Tests |
|------|-------|
| Tick increment | microsecond precision, monotonic count |
| Sync dispatch | message per peer, send_fn callback, no-identity fallback |
| Signature validation | valid accepted, invalid rejected, no-crypto mode |
| Drift computation | zero with no peers, single peer, multiple peers |
| BPM adjustment | triggered above threshold, skipped below threshold, clamped to ±5% |
| PID behaviour | integral accumulation, wind-up clamp, derivative damping |
| Safe range | correction factor always 0.95–1.05 |
| UnifiedBeat | dataclass fields, dict round-trip |
| Status | accurate peer count, global beat, history |
| Thread safety | 16 concurrent syncs, 20 concurrent receives |

---

## File List

| File | Description |
|------|-------------|
| `nerve/distributed_metronome_bridge.py` | Core module (3 classes + 2 exceptions) |
| `tests/test_distributed_metronome_bridge.py` | 17 pytest cases |
| `docs/DISTRIBUTED_METRONOME_BRIDGE.md` | This architecture guide |

---

## Branch

`feature/distributed-metronome-bridge`

## Next Steps

1. Wire `FleetConductor` to instantiate `MetronomeBridge` on startup.
2. Implement `MeshVectorGossip` payload filter for `"metronome_sync"`.
3. Add `OperationalTrap` drift monitor (> 50 ms alert).
4. Deploy to a 2-node test fleet and measure overnight drift.
