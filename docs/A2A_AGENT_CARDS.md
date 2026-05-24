# A2A Agent Cards Index

This document lists all agent identity cards in the Cocapn fleet and their advertised capabilities. Each card follows the **Google A2A specification draft** pattern with `name`, `version`, `description`, `capabilities`, `skills`, and `authentication` fields.

## Quick Reference Table

| Agent Card | Version | Streaming | Push | Key Skills |
|-----------|---------|-----------|------|------------|
| [metronome](#metronome) | `1.0.0` | ✅ | ❌ | beat_sync, bpm_negotiation, drift_correction, get_status |
| [breeder](#breeder) | `2.0.0` | ✅ | ❌ | cross_breed, vector_table_query, thermal_aware_spawn, get_lifecycle_state, emergency_stop |
| [nexus](#nexus) | `1.0.0` | ✅ | ✅ | node_registration, federation_sync, agent_discovery, route_task |
| [compiler](#compiler) | `3.0.0` | ✅ | ❌ | hot_swap_compile, profile_pass, backend_selection, constraint_audit |
| [thermal](#thermal) | `2.0.0` | ✅ | ✅ | thermal_monitor, thermal_auction, async_budget, parent_sacrifice |
| [grid](#grid) | `3.0.0` | ✅ | ❌ | room_forward_pass, routing, novelty_scoring, rebirth_room, get_activity |

---

## metronome

**Name:** `sunset-metronome`  
**Version:** `1.0.0`  
**Endpoint:** `http://nexus.fleet.local:4047/metronome`

Periodic pulse generator that drives the nerve grid. Each beat triggers compute, gate, route, and optional harmonic phases (breeding, FLUX audit). Supports BPM negotiation, drift correction, and CRDT-style beat sync across fleet nodes.

### Capabilities

| Capability | Value | Description |
|-----------|-------|-------------|
| `streaming` | `true` | Supports streaming A2A task responses |
| `pushNotifications` | `false` | Does not push proactive notifications |
| `streamingResponse` | `true` | Streams partial results during long tasks |
| `webhooks` | `false` | No webhook callbacks |

### Skills

- **`beat_sync`** — Exchange beat state with peer nodes for drift detection and correction via CRDT-style merge.
- **`bpm_negotiation`** — Negotiate fleet-wide tempo changes with ramped transitions to avoid jarring phase jumps.
- **`drift_correction`** — Apply phase nudge, skip-jump, or partition fallback strategies to keep nodes in phase.
- **`get_status`** — Return current metronome health: beat counter, actual vs target BPM, missed beats, harmonic registry.

---

## breeder

**Name:** `sunset-breeder`  
**Version:** `2.0.0`  
**Endpoint:** `http://nexus.fleet.local:4047/breeder`

A2A-native orchestrator for the agent lifecycle: incubate, compete, survive, breed, sunset. Manages tournaments, fitness scoring, and genetic operations across the fleet with cross-node breeding and vector table awareness.

### Capabilities

| Capability | Value | Description |
|-----------|-------|-------------|
| `streaming` | `true` | Supports streaming A2A task responses |
| `pushNotifications` | `false` | Does not push proactive notifications |
| `streamingResponse` | `true` | Streams partial results during long tasks |
| `webhooks` | `false` | No webhook callbacks |

### Skills

- **`cross_breed`** — Request breeding with a remote agent by querying vector tables and selecting Pareto-optimal parents.
- **`vector_table_query`** — Query remote or local vector tables for agent embeddings, fitness landscapes, and compatibility scores.
- **`thermal_aware_spawn`** — Incubate offspring only when thermal budget allows, with parent-sacrifice fallback.
- **`get_lifecycle_state`** — Return the current lifecycle state machine snapshot across all agents.
- **`emergency_stop`** — Halt all breeding activity immediately with optional sunset of non-viable agents.

---

## nexus

**Name:** `sunset-nexus`  
**Version:** `1.0.0`  
**Endpoint:** `http://nexus.fleet.local:4047/nexus`

Fleet nexus for node registration, federation, and discovery. Maintains the canonical agent registry and routes A2A task negotiation between fleet members.

### Capabilities

| Capability | Value | Description |
|-----------|-------|-------------|
| `streaming` | `true` | Supports streaming A2A task responses |
| `pushNotifications` | `true` | Pushes proactive topology and consensus updates |
| `streamingResponse` | `true` | Streams partial results during long tasks |
| `webhooks` | `true` | Supports webhook callbacks for fleet events |

### Skills

- **`node_registration`** — Register a new fleet node with its capabilities, agent cards, and endpoint addresses.
- **`federation_sync`** — Synchronize fleet topology, agent registry, and consensus state across federated nodes.
- **`agent_discovery`** — Discover agents by skill, capability, or identity across the federated fleet.
- **`route_task`** — Route an A2A task to the best-matching agent based on capability scoring and load.

---

## compiler

**Name:** `sunset-compiler`  
**Version:** `3.0.0`  
**Endpoint:** `http://nexus.fleet.local:4047/compiler`

Hot-swap compilation agent with built-in profiler and backend selection. Compiles room grids, vector pipelines, and constraint checkers on-the-fly while respecting thermal budgets.

### Capabilities

| Capability | Value | Description |
|-----------|-------|-------------|
| `streaming` | `true` | Supports streaming A2A task responses |
| `pushNotifications` | `false` | Does not push proactive notifications |
| `streamingResponse` | `true` | Streams partial results during long tasks |
| `webhooks` | `false` | No webhook callbacks |

### Skills

- **`hot_swap_compile`** — Compile and hot-swap a room grid or pipeline module without stopping the fleet beat.
- **`profile_pass`** — Run a profiler pass on a target module and return bottleneck report with optimization hints.
- **`backend_selection`** — Select optimal backend (CPU, CUDA, ROCm, WebGPU) based on workload shape and device telemetry.
- **`constraint_audit`** — Audit compiled bytecode against FLUX constraints before deployment.

---

## thermal

**Name:** `sunset-thermal`  
**Version:** `2.0.0`  
**Endpoint:** `http://nexus.fleet.local:4047/thermal`

Thermal monitoring, auction, and async budget management. Tracks per-device temperature, utilization, and power draw; runs thermal auctions to allocate slots; enforces async budget caps.

### Capabilities

| Capability | Value | Description |
|-----------|-------|-------------|
| `streaming` | `true` | Supports streaming A2A task responses |
| `pushNotifications` | `true` | Pushes proactive thermal alert notifications |
| `streamingResponse` | `true` | Streams partial results during long tasks |
| `webhooks` | `true` | Supports webhook callbacks for thermal events |

### Skills

- **`thermal_monitor`** — Stream real-time thermal telemetry: temperature, utilization, power draw per device.
- **`thermal_auction`** — Run a Vickrey-style auction to allocate limited thermal slots among competing agents.
- **`async_budget`** — Manage async thermal budget caps: set, query, and enforce per-agent thermal quotas.
- **`parent_sacrifice`** — Allow a parent agent to release its thermal slot before its child is spawned.

---

## grid

**Name:** `sunset-grid`  
**Version:** `3.0.0`  
**Endpoint:** `http://nexus.fleet.local:4047/grid`

JEPA-powered room grid. Each room runs an MLP forward pass, novelty scoring, and chaos dynamics. Rooms fire when their latent exceeds a threshold, driving the routing layer.

### Capabilities

| Capability | Value | Description |
|-----------|-------|-------------|
| `streaming` | `true` | Supports streaming A2A task responses |
| `pushNotifications` | `false` | Does not push proactive notifications |
| `streamingResponse` | `true` | Streams partial results during long tasks |
| `webhooks` | `false` | No webhook callbacks |

### Skills

- **`room_forward_pass`** — Drive the grid forward by one step: compute latents for all rooms, apply novelty/chaos gating, return fired rooms.
- **`routing`** — Route signals between rooms based on learned routing weights and firing thresholds.
- **`novelty_scoring`** — Compute per-room novelty scores against a surprise buffer and return anomaly rankings.
- **`rebirth_room`** — Reset a room's weights and chaos to initial values, effectively rebirthing it.
- **`get_activity`** — Return activity summary: firing rates, chaos levels, novelty distributions, and compile-status.

---

## Integration Touchpoints

### MetronomeA2A → Agent Cards

`MetronomeA2A` tasks (beat sync, BPM negotiation, drift correction) reference agent cards for capability matching before initiating cross-node negotiation. The workflow:

1. Metronome calls `AgentRegistry.discover(nexus_url)` to populate the local registry.
2. Before sending a `beat_sync` task, it checks `card.supports_skill("beat_sync")`.
3. If the target agent does not advertise the skill, the task is rejected early with a `NegotiationError`.
4. If `card.capability("streaming")` is true, Metronome uses `AgentRegistry.register_stream_handler()` to receive incremental drift-correction deltas.

### MeshVectorGossip → AgentIdentity

`MeshVectorGossip` signs gossip deltas with `AgentIdentity.sign_task()` before broadcasting them to peers. Each delta carries:

- `agent_id` — the originating agent's canonical identity
- `signature` — Ed25519 signature over the canonicalized delta payload
- `public_key_pem` — optional inline public key for first-contact verification

Downstream peers verify deltas via `AgentIdentity.verify_task(delta, signature, public_key_pem)`. Unsigned or invalid deltas are dropped before entering the vector table, preventing poisoning.

### SignedWAL → AgentIdentity.verify()

`SignedWAL` appends task records with signatures generated by the requesting agent's `AgentIdentity.sign_task()`. On replay or audit:

1. `SignedWAL` retrieves the `agent_id` from the record header.
2. It looks up the agent's public key in the `AgentRegistry` or a trusted keyring.
3. It calls `AgentIdentity.verify_task(payload, signature, public_key_pem)`.
4. Only verified records are replayed; failed verifications trigger a tamper alert and WAL truncation at the first bad record.

This binds every task in the WAL to a cryptographically verifiable agent identity, ensuring non-repudiation across the fleet.

---

## Schema Version

**Agent Card Schema:** Google A2A specification draft (adapted for fleet use)  
**Required fields:** `name`, `version`, `description`, `capabilities`, `skills`  
**Required capability booleans:** `streaming`, `pushNotifications`  
**Required skill fields:** `id`, `name`, `description`, `tags`, `examples`

## File Layout

```
.well-known/agent-cards/
├── metronome.json
├── breeder.json
├── nexus.json
├── compiler.json
├── thermal.json
└── grid.json
```

Cards are loaded at runtime by `AgentRegistry.load_local_cards(".well-known/agent-cards/")` and registered under their file stem (e.g., `metronome`).
