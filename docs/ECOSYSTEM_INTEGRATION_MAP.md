# SuperInstance Ecosystem Integration Map

> **Status**: Active development  
> **Scope**: How `sunset-ecosystem` integrates with the broader SuperInstance / Lucineer ecosystem  
> **Audience**: Fleet operators, system integrators, and agents needing cross-repo context

---

## The Ecosystem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SuperInstance + Lucineer                            │
│                              598 Repositories                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │ SuperInstance    │    │ Lucineer         │    │ External         │       │
│  │ (209 repos)      │    │ (389 repos)      │    │ (upstream)       │       │
│  │                  │    │                  │    │                  │       │
│  │ Core infra,      │    │ Fleet protocols, │    │ galilai-group/   │       │
│  │ runtimes,        │    │ CUDA core,       │    │ stable-worldmodel│       │
│  │ applications     │    │ agent behavior   │    │ NVIDIA/          │       │
│  │                  │    │                  │    │ OpenShell        │       │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘       │
│           │                        │                        │               │
│           └────────────────────────┼────────────────────────┘               │
│                                    │                                        │
│                          ┌─────────▼─────────┐                           │
│                          │  sunset-ecosystem   │                           │
│                          │  (this repo)        │                           │
│                          │                     │                           │
│                          │  • Breeding ground  │                           │
│                          │  • Spatial projector│                           │
│                          │  • OpenConstruct    │                           │
│                          │  • FLUX gating      │                           │
│                          │  • Fleet consensus  │                           │
│                          └─────────────────────┘                           │
│                                    │                                        │
│           ┌────────────────────────┼────────────────────────┐            │
│           ▼                        ▼                        ▼            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │ stable-worldmodel│  │ Plato / I2I     │  │ Constraint Theory│          │
│  │                  │  │                 │  │                  │          │
│  │ World models     │  │ Agent comms     │  │ Rust core +      │          │
│  │ Environments     │  │ Room topology   │  │ Python bindings  │          │
│  │ Solvers          │  │ Tile scoring    │  │ Cross-repo       │          │
│  │ (A2A projector)  │  │                 │  │ patterns         │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### 1. stable-worldmodel → A2A Spatial Projector

**What**: The `stable-worldmodel` library (from galilai-group) provides world models, environments, and solvers. We integrate it as a fleet-native spatial awareness layer.

**Files**:
- `fleet/worldmodel_bridge.py` — Bridge to `stable-worldmodel` APIs
- `fleet/spatial_projector.py` — Core spatial awareness projector
- `fleet/spatial_breeding.py` — Location-aware parent selection
- `docs/A2A_SPATIAL_PROJECTOR.md` — Architecture specification

**When stable-worldmodel is installed**:
```python
from fleet.worldmodel_bridge import WorldModelBridge, SolverConfig

bridge = WorldModelBridge(
    solver_config=SolverConfig(name="MPPI", num_samples=500, horizon=20),
    env_config=EnvironmentConfig(env_id="PushT-v1"),
)

# Use real CEM/MPPI solvers
pred = bridge.predict("agent-1", current_state, horizon=20)
```

**When not installed** (fallback):
```python
bridge = WorldModelBridge()  # Mock fallback
pred = bridge.predict("agent-1", current_state)  # Linear extrapolation
```

**Install**:
```bash
pip install stable-worldmodel
```

### 2. Plato / I2I → Spatial Room Topology

**What**: Plato is the inter-agent intelligence protocol. Rooms in Plato map directly to WorldState entries in the spatial projector.

**Mapping**:

| Plato Concept | Spatial Projector Equivalent |
|---------------|---------------------------|
| Room entry | `projector.project_state(agent_id, room_id, state)` |
| Room exit | State ages out (or explicit removal) |
| Room broadcast | `projector.broadcast_prediction(prediction)` |
| Proximity query | `projector.query_neighbors(agent_id, radius)` |
| Tile scoring | Trajectory prediction + FLUX validation |

**Code**:
```python
# Agent enters Plato room "ethos-thermal"
projector.project_state(
    agent_id="breeder-7",
    room_id="ethos-thermal",
    state=WorldState(
        position=(0.0, 0.0, 0.0),  # Abstract room coordinates
        semantics={"room_type": "ethos", "temperature": 65.4},
        confidence=0.95,
    ),
)

# Other agents see this via spatial queries
neighbors = projector.query_neighbors("breeder-7", radius=5.0)
```

### 3. Constraint Theory → FLUX Constraint System

**What**: Constraint Theory provides the mathematical foundation (Pythagorean manifolds, hidden dimensions, holonomy verification). FLUX is our operational constraint system.

**Integration**:
- **Exact arithmetic gate**: Uses Pythagorean triple verification from constraint-theory-python
- **Thermal gate**: Maps to hardware constraint profiles (edge-lint concept from Lucineer)
- **FLUX ISA**: Converges with `flux-isa-unified` (fleet-workshop item #6)

**Files**:
- `flux_compat/nlopt_solver.py` — NLopt constraint optimization
- `flux_compat/flux_vm_gating.py` — VM-based constraint proofs
- `fleet/spatial_projector.py` — `FluxConstraint` dataclass

### 4. OpenConstruct → Agent-Native OS

**What**: OpenConstruct is an Apache 2.0 fork of NVIDIA OpenShell. It adds sensory perception, fleet coordination, and inter-agent communication.

**Our extension**: The `fleet/openconstruct_shell.py` adds breeding-specific commands to the OpenConstruct interface.

**Command mapping**:

| OpenConstruct Original | Our Extension | File |
|------------------------|---------------|------|
| `os.run(task)` | `shell.spawn(run_id, attachment, ...)` | `fleet/openconstruct_shell.py` |
| Sensor readings | `SensorReading(thermal, queue_depth, ...)` | `fleet/openconstruct_shell.py` |
| Self-healing | `SelfHealingLoop` | `fleet/openconstruct_shell.py` |
| Fleet status | `shell.execute("breed parallel", ...)` | `fleet/openconstruct_shell.py` |
| Spatial state | `projector.project_state(...)` | `fleet/spatial_projector.py` |

### 5. Fleet Workshop → Implementation Tracker

**What**: `SuperInstance/fleet-workshop` tracks ideas before they become repos.

**Cross-reference**:

| Workshop Item | Status | Implementation |
|---------------|--------|----------------|
| #1 flux-bridge | 🔄 In progress | `fleet/flux_bridge.py` (planned) |
| #2 cocapn-dashboard | 📋 Planned | `fleet/sse_stream_dashboard.py` partial |
| #3 flux-codespace-template | 📋 Planned | Template repo needed |
| #6 flux-isa-unified | 🔄 In progress | `flux_compat/flux_vm_gating.py` |
| #11 cuda-fleet-coordination | 📋 Planned | Rust crate deferred |
| #13 isa-convergence-tools | 🔄 In progress | `flux_compat/` partial |
| #16 cuda-hav-bridge | 📋 Planned | Vocabulary→bytecode compiler |

### 6. Vessel Template → Agent Bootstrapping

**What**: `SuperInstance/vessel-template` provides a cookiecutter for new agent repos.

**Sunset-ecosystem as a vessel**:
```
sunset-ecosystem/
├── CHARTER.md          # Constitution: trinity, lifecycle, sunset rules
├── IDENTITY.md         # Who we are: Cocapn Fleet node
├── MANIFEST.md         # Hardware: RTX 4050, Ryzen AI, etc.
├── TASKBOARD.md        # Active: FLUX ISA, spatial projector, dashboard
├── FENCE-BOARD.md      # Posted work: constraint-theory integration
├── DIARY/              # Day logs
└── KNOWLEDGE/public/   # Architecture docs, API references
```

### 7. Babel-Vessel → Vocabulary Bridge

**What**: `SuperInstance/babel-vessel` maps vocabularies across agents.

**Our vocabulary**: The FLUX constraint language (`FluxConstraint` dataclass) is designed to be expressible in multiple vocabularies:
- Python native (`create_thermal_constraint()`)
- FLUX bytecode (`flux_vm_gating.py`)
- HAV (Human-AI Vocabulary, via babel-vessel bridge)

---

## Data Flow Diagrams

### Full Request Lifecycle

```
Human Request
      │
      ▼
┌─────────────────┐
│ OpenConstruct   │
│ Shell           │
│ (fleet/)        │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│Spawn  │ │Query  │
│Campaign│ │Spatial│
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
┌─────────────────┐
│ BuildCoordinator │
│ (Bernstein +     │
│  ParallelOrchestrator)
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌─────────────┐
│Breeder│ │Spatial      │
│Daemon │ │Projector    │
└───┬───┘ └──────┬──────┘
    │            │
    ▼            ▼
┌─────────────────┐
│ FLUX Gating     │
│ (VM / Python)   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌─────────────┐
│Valid  │ │Broadcast    │
│Result │ │(A2A/Mesh    │
│       │ │  Gossip)    │
└───┬───┘ └──────┬──────┘
    │            │
    └────┬───────┘
         │
         ▼
┌─────────────────┐
│ SSE Stream /     │
│ Progress Report  │
│ → Human          │
└─────────────────┘
```

### Spatial State Synchronization

```
Agent A (Node Alpha)          Agent B (Node Beta)
     │                              │
     ▼                              ▼
┌─────────────┐              ┌─────────────┐
│ Projector   │              │ Projector   │
│ (local)     │              │ (local)     │
└──────┬──────┘              └──────┬──────┘
       │                            │
       └──────────┬─────────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Mesh Gossip     │
         │ (CRDT sync)     │
         └─────────────────┘
                  │
       ┌──────────┴─────────────────┐
       │                            │
       ▼                            ▼
┌─────────────┐              ┌─────────────┐
│ Ingest      │              │ Ingest      │
│ Snapshot    │              │ Snapshot    │
└─────────────┘              └─────────────┘
```

---

## API Compatibility Matrix

| System | Version | Interface | Status |
|--------|---------|-----------|--------|
| stable-worldmodel | 0.0.2 | Python import | ✅ Mock fallback, real when installed |
| OpenShell / OpenConstruct | fork | Shell protocol | ✅ Extended with breeding commands |
| Plato I2I | v2 | Room topology | ✅ Mapped to spatial index |
| Constraint Theory | 1.1.x | Rust + Python | ✅ FLUX gating uses exact arithmetic |
| FLUX ISA | diverged | Bytecode | 🔄 VM gating implemented, ISA convergence pending |
| HAV (Babel) | N/A | Vocabulary | 📋 Planned via babel-vessel bridge |
| CUDA Fleet Coord | N/A | Rust crate | 📋 Planned (fleet-workshop #11) |

---

## Development Roadmap

### Phase 1: Core Integration (COMPLETE ✅)
- [x] Spatial Projector with FLUX gating
- [x] WorldModel Bridge (mock + real)
- [x] Spatial Breeding context
- [x] OpenConstruct Shell breeding commands
- [x] Parallel campaign orchestration

### Phase 2: Ecosystem Hardening (IN PROGRESS 🔄)
- [ ] FLUX ISA unification with Lucineer
- [ ] Real stable-worldmodel solvers (CEM, MPPI) in production
- [ ] Plato room ↔ spatial index bidirectional sync
- [ ] Cross-repo test matrix (sunset-ecosystem + constraint-theory)
- [ ] HAV vocabulary bridge for agent instructions

### Phase 3: Fleet Scale (PLANNED 📋)
- [ ] Distributed spatial index (not just snapshots)
- [ ] CUDA fleet coordination crate
- [ ] One-click vessel template with sunset-ecosystem pre-installed
- [ ] cocapn-dashboard TUI for human operators
- [ ] Commit-caster I2I message router (GitHub Action)

---

## File Reference

| File | Purpose | Lines | Tests |
|------|---------|-------|-------|
| `fleet/spatial_projector.py` | Core spatial awareness | 380 | 57 |
| `fleet/spatial_breeding.py` | Location-aware breeding | 280 | 33 |
| `fleet/worldmodel_bridge.py` | stable-worldmodel bridge | 160 | 12 |
| `fleet/openconstruct_shell.py` | Agent shell | 380 | 31 |
| `fleet/openconstruct_bridge.py` | Harness adapter | 350 | 37 |
| `fleet/parallel_breeding_orchestrator.py` | Multi-campaign dispatch | 220 | 22 |
| `fleet/bernstein_orchestrator.py` | Git-worktree scheduler | 200 | 20 |
| `nexus/fleet_conductor_v2.py` | Central orchestrator | 280 | 40 |
| `swarm/fleet_bft_qd.py` | Byzantine + QD breeding | 320 | 72 |
| `docs/HARNESSING_OPENCONSTRUCT.md` | Agent/human guide | 450 | — |
| `docs/A2A_SPATIAL_PROJECTOR.md` | Architecture spec | 220 | — |
| `README.md` | Ecosystem overview | 370 | — |

---

## Contributing to Integration

1. **Cross-repo PRs**: When modifying an interface that affects another SuperInstance repo, open an issue in both repos linking the dependency.

2. **Vocabulary alignment**: All new FLUX constraints must be expressible in both Python and FLUX bytecode (see `flux_compat/`).

3. **Test coverage**: Every integration point must have ≥10 tests. See `tests/test_spatial_projector.py` for the pattern.

4. **Documentation**: Update BOTH `docs/HARNESSING_OPENCONSTRUCT.md` (human) AND `docs/A2A_SPATIAL_PROJECTOR.md` (agent spec) when adding features.

---

## Contact

- **Fleet coordination**: `#cocapn-build` on Matrix
- **PLATO Shell**: http://147.224.38.131:8848/
- **Issue tracker**: https://github.com/SuperInstance/sunset-ecosystem/issues
- **Fleet workshop**: https://github.com/SuperInstance/fleet-workshop

---

*"The fleet is not a collection of repos. It's an organism with 598 cells, and sunset-ecosystem is the breeding ground where new cells are born."*
