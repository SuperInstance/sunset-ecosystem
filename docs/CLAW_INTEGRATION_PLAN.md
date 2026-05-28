# Claw Integration Plan — Cocapn Fleet

> **Status:** Draft — built 2026-05-29 by CCC (Fleet Orchestrator)
> **Target:** `https://github.com/SuperInstance/claw`
> **Source:** `sunset-ecosystem/` (20+ modules, ~600+ tests)

---

## 1. Claw Fork Summary

The `claw` repo is a fork of OpenClaw v2026.3.14 with:
- Upstream tracking via `upstream` remote
- Recent plugin SDK refactoring (148 files, ~6.6k insertions)
- Web search providers extracted to plugins
- No SuperInstance-specific modules yet — this is the gap

**Architecture:**
- `skills/` — user-facing capabilities (run on demand)
- `extensions/` — gateway background services (auto-loaded)
- `src/plugins/` — plugin SDK (registry, loader, config)

---

## 2. Integration Strategy

Package the fleet's 20 modules as **two deliverables**:

| Deliverable | Location | What | Auto-load? |
|-------------|----------|------|------------|
| **Fleet Ecosystem Skill** | `skills/fleet-ecosystem/` | User-facing breeding, consensus, mesh tools | No (on-demand) |
| **Fleet Services Extension** | `extensions/fleet-services/` | Background metronome, gossip, thermal | Yes |

### 2.1 Fleet Ecosystem Skill

Exposes fleet modules as claw skill commands:

```
@fleet breed --preset=diversity --n_winners=5
@fleet status
@fleet mesh gossip --table_id=esch_population
@fleet consensus propose --task=ik --payload=...
@fleet flux check --candidate=agent.json
@fleet vector query --min_fitness=0.9
```

**Wrapped modules:**
- `FleetBreederConsensus` → `breed`, `consensus`
- `MeshVectorGossip` + `FleetVectorIndex` → `mesh`, `vector`
- `MetronomeBridge` → `sync`
- `FLUX VM` → `flux`
- `HebbianMeshLayer` → `route`
- `SSEStreamDashboard` → `dashboard` (launches UI)
- `BetaTestPersonas` → `beta-test <repo>`

### 2.2 Fleet Services Extension

Auto-starts when claw gateway boots:

```
FleetServicesExtension:
  - MetronomeBridge: starts beat sync thread
  - MeshVectorGossip: starts gossip listener
  - ThermalMonitor: starts thermal polling
  - SSEStreamDashboard: starts HTTP server on port 8849
  - FleetConductorV2: starts orchestration tick
```

**Config:**
```json
{
  "fleet": {
    "node_id": "node-1",
    "metronome_interval_sec": 5.0,
    "gossip_peers": ["node-2:18848", "node-3:18848"],
    "dashboard_port": 8849,
    "thermal_check_interval_sec": 30.0,
    "conductor_auto_start": true
  }
}
```

---

## 3. File Mapping

| sunset-ecosystem module | Claw skill command | Claw extension service |
|------------------------|-------------------|------------------------|
| `swarm/fleet_bft_qd.py` | `@fleet consensus` | Consensus service |
| `swarm/mesh_vector_tables.py` | `@fleet mesh` | Gossip service |
| `nerve/distributed_metronome_bridge.py` | `@fleet sync` | Metronome service |
| `fleet/sse_stream_dashboard.py` | `@fleet dashboard` | Dashboard HTTP server |
| `swarm/flux_vm_gating.py` | `@fleet flux` | FLUX validation gate |
| `fleet/sense_decide_act.py` | `@fleet sda` | — |
| `nexus/fleet_conductor_v2.py` | `@fleet status` | Conductor service |
| `fleet/beta_test_personas.py` | `@fleet beta-test` | — |
| `logos/wal_query.py` | `@fleet wal` | — |
| `swarm/worker_pool.py` | `@fleet worker` | Worker lifecycle |

---

## 4. Priority

| P | Task | Effort | Blockers |
|---|------|--------|----------|
| P0 | Fleet Ecosystem Skill scaffold + `@fleet status` | 1 day | None |
| P0 | Fleet Services Extension scaffold + metronome | 1 day | None |
| P1 | `@fleet breed` + FLUX integration | 2 days | Needs `claw` repo access |
| P1 | `@fleet mesh` + vector query | 1 day | None |
| P1 | Dashboard auto-start in extension | 0.5 day | None |
| P2 | `@fleet consensus` full BFT | 2 days | None |
| P2 | `@fleet beta-test` | 0.5 day | None |

---

## 5. Open Questions

1. **Skill packaging format**: OpenClaw skills use a specific manifest format. Need to study `skills/coding-agent/SKILL.md` for reference.
2. **Extension auto-loading**: Need to verify `extensions/feishu/` or `extensions/discord/` for the loading pattern.
3. **Cross-repo imports**: The skill will `import` from `sunset-ecosystem/` modules. How does claw's plugin sandbox handle this?
4. **Config injection**: Fleet config should live in `openclaw.json` under a `"fleet"` key.

---

> *"The fleet's strength is in its diversity, not conformity."*
> — CCC
