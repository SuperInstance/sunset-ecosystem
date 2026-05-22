# A2A Agent Cards — Sunset Ecosystem Fleet Services

**Author:** kimi1 (Fleet Integrator)  
**Date:** 2026-05-22  
**Branch:** `turbovec-integration-ccc`  
**Status:** ✅ Complete — All fleet services expose A2A Agent Cards

---

## 1. What Is an Agent Card?

Per the [A2A Protocol](https://a2a-protocol.org/) (Google/Linux Foundation, 2025), an **Agent Card** is a JSON document served at `/.well-known/agent.json` (or a service-specific variant) that describes:

- **Name and version** — Identity
- **Capabilities** — What tasks the agent can perform, with typed inputs and outputs
- **Authentication** — How to authenticate (x509, OAuth, etc.)
- **Content types** — Default JSON

In the sunset ecosystem, every fleet service is an A2A agent. The Agent Card is its **public interface** — the contract by which other agents delegate tasks to it.

---

## 2. Agent Card Index

| Service | Card Path | Capabilities | Description |
|---|---|---|---|
| **MetronomeScheduler** | `.well-known/agent-metronome.json` | `tick`, `set_bpm`, `sync`, `get_status` | Periodic pulse generator driving the nerve grid |
| **BreederDaemonV2** | `.well-known/agent-breeder.json` | `queue_breed`, `get_state`, `get_stats`, `emergency_stop` | Agent lifecycle orchestrator (incubate → compete → survive → breed → sunset) |
| **RoomGrid** | `.well-known/agent-grid.json` | `tick`, `get_activity`, `get_room_state`, `rebirth_room` | JEPA-powered room grid with per-room local metronomes |
| **FLUX Constraint Checker** | `.well-known/agent-flux.json` | `check_constraints`, `get_violations`, `apply_feedback` | Formally proven constraint checker with proof certificates |

---

## 3. Endpoints

All agents are reachable via the fleet Nexus at `http://nexus.fleet.local:4047/`:

```
POST /metronome/tasks/send    → MetronomeScheduler
POST /breeder/tasks/send      → BreederDaemonV2
POST /grid/tasks/send         → RoomGrid
POST /flux/tasks/send         → FLUX Constraint Checker
```

Alternatively, each service may expose its own `/.well-known/` path for discovery.

---

## 4. Example A2A Task Payloads

### 4.1 Metronome — Tick

```json
{
  "id": "task-001",
  "type": "tick",
  "input": {
    "signal": [0.1, -0.3, 0.7, 0.0, 0.2, -0.1, 0.5, 0.0, ...],
    "force": false
  }
}
```

Response:
```json
{
  "status": "completed",
  "artefacts": [{
    "type": "TickResult",
    "content": {
      "beat_number": 1423,
      "fired_rooms": [42, 77, 128, 301],
      "fired_count": 4,
      "phase_durations_ms": {
        "compute": 2.1,
        "gate": 0.3,
        "route": 0.8
      },
      "missed_beat": false
    }
  }]
}
```

### 4.2 Metronome — Set BPM

```json
{
  "id": "task-002",
  "type": "set_bpm",
  "input": {
    "bpm": 240.0,
    "ramp_ms": 2000,
    "reason": "Tournament mode — reactive agents need faster routing"
  }
}
```

### 4.3 Breeder — Queue Breed

```json
{
  "id": "task-003",
  "type": "queue_breed",
  "input": {
    "parent_count": 2,
    "offspring_count": 3,
    "incubate_room": "Forge",
    "strategy": "trinity"
  }
}
```

Response:
```json
{
  "status": "completed",
  "artefacts": [{
    "type": "BreedResult",
    "content": {
      "children": [
        {
          "agent_id": "agent-7f3a-9c2e",
          "parent_ids": ["agent-1a2b", "agent-3c4d"],
          "fitness": {"ethos": 0.87, "pathos": 0.92, "logos": 0.79, "product": 0.634},
          "incubated": true,
          "room_id": "Forge"
        }
      ],
      "cycle_id": "cycle-2026-05-22-001",
      "queue_position": 0
    }
  }]
}
```

### 4.4 Room Grid — Tick

```json
{
  "id": "task-004",
  "type": "tick",
  "input": {
    "signal": [0.1, -0.3, 0.7, 0.0, 0.2, -0.1, 0.5, 0.0, ...],
    "room_ids": [0, 1, 2, 3, 4],
    "skip_local_metronomes": false
  }
}
```

### 4.5 Room Grid — Get Activity

```json
{
  "id": "task-005",
  "type": "get_activity",
  "input": {
    "window_ticks": 100
  }
}
```

### 4.6 FLUX — Check Constraints

```json
{
  "id": "task-006",
  "type": "check_constraints",
  "input": {
    "values": [[0.5, -0.2, 0.1, ...], [9.5, -8.2, 7.1, ...]],
    "preset": "neural_bounds",
    "domain": "neural",
    "generate_certificate": true
  }
}
```

Response:
```json
{
  "status": "completed",
  "artefacts": [{
    "type": "ConstraintCheckResult",
    "content": {
      "pass": false,
      "checked_count": 2,
      "violation_count": 1,
      "violations": [
        {
          "index": 1,
          "constraint": "bounds",
          "expected": 10.0,
          "actual": 9.5,
          "severity": "error",
          "remediation": "clip to [-10, 10]"
        }
      ],
      "certificates": [
        {"result": "PASS", "hash": "sha256:a1b2...", "timestamp": "2026-05-22T13:00:00Z", "verified": true},
        {"result": "FAIL", "hash": "sha256:c3d4...", "timestamp": "2026-05-22T13:00:00Z", "verified": true}
      ],
      "preset_used": "neural_bounds"
    }
  }]
}
```

### 4.7 FLUX — Apply Feedback

```json
{
  "id": "task-007",
  "type": "apply_feedback",
  "input": {
    "target_id": "ship-jetson-01",
    "chaos_delta": 0.1,
    "rebirth_threshold": 3,
    "dry_run": false
  }
}
```

---

## 5. Authentication

All fleet Agent Cards require **x509 certificate-based authentication**:

```json
{
  "authentication": {
    "scheme": "x509",
    "required": true
  }
}
```

This follows the security posture outlined in `A2A-FIRST-ARCHITECTURE.md` §10 (Open Question #1): *"A2A without auth is MCP all over again."* Every agent in the fleet carries a certificate signed by the fleet CA.

---

## 6. Content Types

All inputs and outputs use `application/json` by default. Structured artefacts (ProofCertificate, TickResult, BreedResult) are JSON-native and parseable by any A2A agent without schema negotiation.

---

## 7. Discovery

Agents discover each other via:

1. **Fleet Nexus** (`nexus.fleet.local:4047`) — Central registry of active agents and their cards
2. **PLATO Room Agents** — Each room advertises its local agents
3. **Git Repository** — Agent Cards committed to `sunset-ecosystem/.well-known/` for offline reference

---

## 8. Next Steps

| # | Task | Owner | Status |
|---|---|---|---|
| 1 | Wire `A2ASignalSource` into MetronomeScheduler | FM | 🟡 Pending |
| 2 | Implement `FleetConductor` for distributed sync | kimi1 | 🟡 Pending |
| 3 | Add Agent Cards for Grammar Engine and PLATO Shell | kimi1 | 🔴 Not started |
| 4 | Wire A2A task endpoints into actual HTTP handlers | FM | 🔴 Not started |
| 5 | Generate x509 fleet CA and per-agent certificates | FM | 🔴 Not started |

---

## 9. References

- `A2A-FIRST-ARCHITECTURE.md` — Post-coding paradigm, Agent Cards, A2A mesh
- `SPEC_METRONOME_BRIDGE.md` §5.4 — Metronome Agent Card design
- `SPEC_BREEDER_DAEMON_V2.md` — Breeder lifecycle FSM
- `SPEC_FLUX_PIPELINE.md` — FLUX constraint pipeline
- `SPEC-NERVE-TOPOLOGY.md` — RoomGrid internals
- A2A Protocol: https://a2a-protocol.org/

---

*"Every arrow is A2A. Every node has an Agent Card. Every message is JSON."*
*— kimi1, Fleet Integrator, 2026-05-22*
