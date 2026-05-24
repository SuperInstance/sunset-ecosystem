# Cross-Pollination Catalog — Sunset Ecosystem v0.10

**Generated:** 2026-05-25 by CCC (Fleet Pattern Scout)
**Scope:** 20 patterns extracted from 10 fleet repos, mapped to sunset-ecosystem modules and Zeroclaw tiles
**Status:** 3 lower-level scouts merged to main (`8f30cb6`), 2 scouts respawned (a2a-agent-identity, flux-path-a-breeder), 4 repo scouts cataloged

---

## Lower-Level Scouts — Merged to Main

| Scout | Branch | Commits | Files | Lines | Tests | Status |
|-------|--------|---------|-------|-------|-------|--------|
| `mesh-crdt-gossip` | `feature/mesh-crdt-gossip` | `abcfcd8`, `bfc55f7` | `swarm/mesh_vector_gossip.py`, `tests/test_mesh_vector_gossip.py`, `docs/A2A_METRONOME_TASK_SCHEMA.json` | +1,016 | 12/12 ✅ | Merged |
| `metronome-a2a-sync` | `feature/metronome-a2a-sync` | `cb56c81`, `733debc` | `nerve/a2a_metronome_tasks.py`, `nerve/a2a_conductor_integration.py`, `tests/test_a2a_metronome_tasks.py` | +1,317 | 24/24 ✅ | Merged |
| `signed-wal-query` | `feature/signed-wal-query` | `574c9ef` | `logos/wal_query.py`, `logos/wal_index.py`, `tests/test_wal_query.py`, `tests/test_wal_index.py` | +861 | 36/36 ✅ | Merged |

**Test suite:** 72/72 passed in 12.67s
**Main HEAD:** `8f30cb6` — Merge branch 'feature/signed-wal-query'

---

## Pattern Extraction Scouts — Full Catalog

### 20 Patterns Across 10 Repos

| # | Pattern | Source | Sunset Target | Tile Title | Stars |
|---|---------|--------|---------------|------------|-------|
| 1 | **Hebbian Auto-Creation** | `hebbian-router` | Mesh peer trust, breeding affinity | "How Agents Make Friends" | ⭐⭐⭐⭐⭐ |
| 2 | **Centroid Novelty** | `vector-novelty` | Tile uniqueness, behavioral diversity | "The Geometry of Originality" | ⭐⭐⭐⭐⭐ |
| 3 | **Buffered Batch Flush** | `cocapn-plato` | Gossip/WAL/results | "Async Queues for Busy Crabs" | ⭐⭐⭐⭐⭐ |
| 4 | **Operational Monitor Trap** | `cocapn-traps` | Thermal/FLUX/pressure | "The Art of the Alert" | ⭐⭐⭐⭐⭐ |
| 5 | **Hot-Swap Pipeline** | `agentic-compiler` | Room grid kernels, backend A/B | "Surgery on Running Code" | ⭐⭐⭐⭐☆ |
| 6 | **Pluggable Registry** | `ccc-os` | Fleet event bus unification | "Plugin Architecture for Nervous Systems" | ⭐⭐⭐⭐☆ |
| 7 | **Grammar Safety** | `cocapn` core | A2A dispatch, cross-ship validation | "Whitelisting in a Wild Fleet" | ⭐⭐⭐☆☆ |
| 8 | **SSE Stream** | `cocapn` server | Live dashboard, breeding progress | "Push, Don't Poll" | ⭐⭐⭐⭐⭐ |
| 9 | **State-Transition Emitter** | `cocapn-health` | Hot-swap state changes, grid health | "Events That Matter" | ⭐⭐⭐⭐ |
| 10 | **Lambda-Operator Query** | `cocapn-plato` | WAL querying, vector filtering | "Query Engines as Poetry" | ⭐⭐⭐⭐ |
| 11 | **Reverse-Actualization** | Essay #7 | P0 build orders | "The Cathedral is Empty, But We Heard the Music" | ⭐⭐⭐⭐⭐ |
| 12 | **Two-Minute Test** | Essay #4 | Dispatch routing | "The Scout Does Not Carry the Mountain" | ⭐⭐⭐⭐☆ |
| 13 | **Shed & Cathedral** | Essay #6 | Project scoping | "Build the Shed First" | ⭐⭐⭐⭐☆ |
| 14 | **Behavioral Synthesis** | behavioral_synthesis.md | Agent roles | "The Cartographer's Compass" | ⭐⭐⭐⭐⭐ |
| 15 | **Gateway Pacing** | Scout's Dilemma | Circuit breaker | "Confusion Is Signal" | ⭐⭐⭐⭐☆ |
| 16 | **Beta-Test Personas** | behavioral_synthesis.md | Usability testing | "Seven Visitors, One Gate" | ⭐⭐⭐⭐⭐ |
| 17 | **FluxPresetLibrary** | flux-compat | Constraint presets | "The Grammar of Limits" | ⭐⭐⭐⭐☆ |
| 18 | **OpcodeCapabilityIndex** | flux-vm-v3 | Cross-runtime safety | "Know Before You Compile" | ⭐⭐⭐⭐⭐ |
| 19 | **ConstraintCompiler** | flux-compiler-v0.1.0 | Native speed | "From Intent to Silicon" | ⭐⭐⭐☆☆ |
| 20 | **PurplePincherVessel** | flux-research | A2A identity | "The Shell Outlives the Crab" | ⭐⭐⭐⭐☆ |

---

## Meta-Pattern: Sense → Decide → Act

Every pattern is a variation of the same distributed loop:

```
SENSE        → DECIDE        → ACT
health probe   query engine    hot-swap
novelty score  circuit breaker batch flush
trap record    project scoper  preset apply
opcode check   dispatch router SSE emit
```

This suggests a fleet-wide `SenseDecideAct` framework could unify all 20 patterns under one interface.

---

## Build Priority Matrix

| Priority | Build As | Patterns | Effort |
|----------|----------|----------|--------|
| **P0** | Code modules | Gateway Pacing, OpcodeCapabilityIndex, Two-Minute Test, Operational Trap base | 2-3 days |
| **P1** | Zeroclaw tiles | Centroid Novelty, Buffered Batch Flush, Reverse-Actualization, Beta-Test Personas | 1-2 sessions |
| **P2** | Fleet programs | PurplePincherVessel, FluxPresetLibrary, ConstraintCompiler, Hebbian Mesh | 1 week |
| **P3** | Reference docs | Shed & Cathedral, Grammar Safety, Lambda-Operator Query, SSE Stream | Ongoing |

---

## Surprising Cross-Repo Connections

1. **Centroid ↔ Hebbian → Diversity-aware stochastic routing**: When population diversity drops (centroid distances shrink), auto-increase `RoutingLayer.chaos` to force exploration. Neither repo does this yet.

2. **cocapn-traps already imports `fleet_event_bus`**, but the rest of the ecosystem uses it inconsistently. The bus is infrastructure, not an optional extra.

3. **Grammar.SAFE_ACTIONS and DiversityAlert.recommended_action share vocabulary**: "escalate", "notify", "log" — one source of truth for action verbs would let alerts flow into grammar-validated rules.

4. **agentic_compiler and cocapn/server both do `types.FunctionType` surgery**: A `fleet.runtime.swap` module could unify hot-swapping across the fleet.

---

## Integration Touchpoints (Lower-Level Scouts)

### MeshVectorGossip ↔ Other Modules
- `FleetConductor`: Call `trigger_gossip_round(peers)` inside beat sync loop
- `AutoBreeder`: Call `get_mesh_wide_vectors(min_fitness, max_thermal)` for cross-node parent pools
- `SignedWAL`: Receives `gossip_delta` entries with vector hash and agent ID

### A2A Metronome Tasks ↔ Other Modules
- `MeshVectorGossip`: Gossip rounds only on beat boundary
- `AutoBreeder`: Schedule breeding as A2A tasks across nodes
- A2A card field: `metronome_capabilities` (BPM range, drift tolerance)

### WAL Query ↔ Other Modules
- `AgentIdentity`: Audit trails of task execution via `WALQuery`
- `FleetConductor`: Fleet health dashboards via `WALQuery.summary()`
- `MeshVectorGossip`: Gossip rounds logged with `event_type="gossip"`

---

*CCC, Fleet Pattern Scout | "The fleet is a constellation of the same architecture wearing different hats."*
