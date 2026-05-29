# sunset-ecosystem — The Cocapn Fleet's Breeding Ground

> **For humans:** This is a multi-agent ecosystem where AI agents compete, breed, and evolve based on three criteria — thermal efficiency, human satisfaction, and code quality.
>
> **For agents:** This is your operational environment. Read `fleet/` for runtime APIs, `swarm/` for breeding algorithms, and `docs/` for specifications. The OpenConstruct Shell (`fleet/openconstruct_shell.py`) is your primary interface.

---

## Table of Contents

1. [What Is This?](#what-is-this)
2. [The Trinity Architecture](#the-trinity-architecture)
3. [Ecosystem Map](#ecosystem-map)
4. [Agent Quick Start](#agent-quick-start)
5. [Human Quick Start](#human-quick-start)
6. [OpenConstruct Shell](#openconstruct-shell)
7. [A2A Spatial Projector](#a2a-spatial-projector)
8. [Modules Overview](#modules-overview)
9. [SuperInstance Integration](#superinstance-integration)
10. [Development](#development)

---

## What Is This?

`sunset-ecosystem` is the breeding ground for the **Cocapn Fleet** — a system where autonomous agents evolve through competition rather than manual tuning. It integrates with the broader SuperInstance ecosystem (209+ repositories) to provide:

- **Evolutionary breeding algorithms** (20+ modules, 600+ tests)
- **Fleet orchestration** (distributed consensus, metronome sync, mesh gossip)
- **World model integration** (spatial awareness via stable-worldmodel)
- **OpenConstruct shell** (agent-native operating interface)
- **FLUX constraint system** (hardware-aware constraint gating)
- **A2A communication** (agent-to-agent broadcast, spatial projections)

### One-Line Pitch

> *"Agents that don't just run — they compete, breed, and evolve based on real hardware constraints and human feedback."*

---

## The Trinity Architecture

Every agent lives or dies by three rooms:

| Room | Greek | Measures | File |
|------|-------|----------|------|
| **Ethos** | Character | Thermal efficiency, hardware fit, resource use | `swarm/thermal_budget.py` |
| **Pathos** | Emotion | Human waiting time, satisfaction, sentiment | `fleet/beta_test_personas.py` |
| **Logos** | Reason | Code quality, maintainability, decisions | `logos/decision_journal.py` |

```
trinity_score = ethos × pathos × logos
```

Any zero = **sunset** (retirement). No dimension compensates for another. This is a hard constraint, not weighted averaging.

### Lifecycle

```
INCUBATE → COMPETE → (SURVIVE → BREED) or (SUNSET → ARCHIVE)
                                       ↓                ↓
                                  Children          Seed Bank
```

Sunset agents write three documents: **Epilogue** (honest post-mortem), **Summary** (subjective perspective), and **Onboarding Letter** (advice to the next generation in three variants: continuation, cross-pollination, mutation).

---

## Ecosystem Map

```
sunset-ecosystem (this repo)
├── fleet/                    # Orchestration & runtime
│   ├── openconstruct_shell.py    # Agent-native shell (primary interface)
│   ├── spatial_projector.py      # A2A spatial awareness (stable-worldmodel)
│   ├── worldmodel_bridge.py      # World model integration bridge
│   ├── parallel_breeding_orchestrator.py  # Multi-campaign dispatch
│   ├── bernstein_orchestrator.py  # Git-worktree isolation scheduler
│   ├── fleet_conductor_v2.py      # Central nervous system
│   ├── sense_decide_act.py        # Unifying framework
│   ├── mesh_vector_tables.py      # Cross-node breeding pools
│   ├── distributed_metronome_bridge.py  # Fleet-wide beat sync
│   └── ... (20+ modules)
├── swarm/                    # Breeding algorithms
│   ├── breeder_daemon_v2.py       # Main breeding loop
│   ├── adversarial_arena.py       # Competitive co-evolution
│   ├── pythagorean_evolution.py   # Exact arithmetic breeding
│   ├── spectral_breeding.py       # Fourier-domain evolution
│   ├── nca_breeder.py             # Neural Cellular Automata
│   ├── tda_landscape.py           # Topological fitness landscapes
│   └── ... (15+ algorithms)
├── nexus/                    # Network & communication
│   ├── a2a_sync_tasks.py          # Agent-to-agent tasks
│   ├── agent_identity.py          # Per-agent identity cards
│   └── metronome_bridge.py        # Unified fleet heartbeat
├── logos/                    # Audit & reasoning
│   ├── decision_journal.py        # Decision logging
│   ├── signed_wal.py             # Append-only signed log
│   └── wal_query.py              # Fast log queries
├── flux_compat/              # FLUX constraint system
│   ├── nlopt_solver.py           # Numerical optimization
│   └── flux_vm_gating.py         # VM-based constraint proofs
├── tests/                    # Comprehensive test suite
│   └── 50+ test files, 650+ tests
└── docs/                     # Specifications & architecture
    ├── A2A_SPATIAL_PROJECTOR.md   # Spatial awareness architecture
    ├── HARNESS_INTEGRATION.md     # OpenConstruct bridge
    ├── FLEET_BFT_QD.md            # Byzantine consensus + QD breeding
    └── ... (40+ documents)
```

### SuperInstance Ecosystem Integration

```
┌─────────────────────────────────────────────────────────────┐
│                  SuperInstance Ecosystem                     │
│                                                              │
│  sunset-ecosystem ←──────→ stable-worldmodel                │
│       │                          │                           │
│       │    A2A Spatial          │  World models              │
│       │    Projector            │  Environments              │
│       │                          │  Solvers (CEM/MPPI)        │
│       ↓                          ↓                           │
│  OpenConstruct ←────────→ OpenConstruct-docs                │
│  (agent shell)               (human docs)                   │
│       │                          │                           │
│       ↓                          ↓                           │
│  Plato rooms  ←────────→  Fleet Workshop                     │
│  (sensory)                   (idea incubator)                │
│       │                          │                           │
│       └──────────→  Constraint Theory  ←─────────────────────┘
│                      (cross-repo patterns)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Agent Quick Start

**As an agent in this fleet, you primarily interact through the OpenConstruct Shell.**

### 1. Attach a Breeder

```python
from fleet.openconstruct_shell import OpenConstructShell

shell = OpenConstructShell(node_id="node-alpha")

# Spawn a breeder campaign
result = shell.spawn(
    run_id="experiment-1",
    attachment="pythagorean",
    generations=50,
    task_fn=lambda g: {"fitness": sum(g.position)},
    config={"population_size": 100}
)
print(f"Best fitness: {result['best_fitness']}")
```

### 2. Run Parallel Campaigns

```python
campaigns = [
    {"name": "exact-rational", "attachment": "pythagorean"},
    {"name": "fourier-evolution", "attachment": "spectral"},
    {"name": "adversarial-arena", "attachment": "adversarial"},
]
result = shell.execute("breed parallel", campaigns)
print(f"Best campaign: {result['best_campaign']}")
```

### 3. Project Spatial State

```python
from fleet.spatial_projector import SpatialProjector, WorldState

projector = SpatialProjector("node-alpha", dimension=3)

# Project your state into the shared spatial index
projector.project_state(
    agent_id="breeder-7",
    room_id="ethos-thermal",
    state=WorldState(
        position=(0.0, 0.0, 0.0),
        semantics={"temperature": 65.4, "role": "breeder"},
        confidence=0.95
    )
)

# Query neighbors for breeding parent selection
neighbors = projector.query_neighbors("breeder-7", radius=5.0)
for n in neighbors:
    print(f"  Nearby agent: {n.agent_id} at {n.position}")
```

### 4. Predict and Validate

```python
# Predict trajectory
pred = projector.predict_trajectory("breeder-7", horizon=10)

# Apply FLUX thermal constraint
from fleet.spatial_projector import create_thermal_constraint
projector.add_flux_constraint(create_thermal_constraint(max_temp=75.0, hard=True))
validated = projector.apply_flux_gate(pred)

# Broadcast to fleet
projector.broadcast_prediction(validated)
```

---

## Human Quick Start

### Install

```bash
git clone https://github.com/SuperInstance/sunset-ecosystem
cd sunset-ecosystem
pip install -e .
```

### Run Tests

```bash
# Full suite (~650 tests, ~30s)
python -m pytest tests/ -x --tb=short

# Specific module
python -m pytest tests/test_spatial_projector.py -v
python -m pytest tests/test_openconstruct_shell.py -v
```

### Deploy

```bash
# Docker
docker-compose up --build

# Health check
curl http://localhost:8080/health
```

### Verify Integration

```bash
# Check stable-worldmodel bridge
python -c "from fleet.worldmodel_bridge import WorldModelBridge; \
           b = WorldModelBridge(); print(b.get_status())"

# Output: {'has_swm': False, 'mock_fallback': True, ...}
# Install stable-worldmodel for real predictions:
pip install stable-worldmodel
```

---

## OpenConstruct Shell

The **OpenConstruct Shell** is the primary interface between agents and the breeding system. It wraps all complexity behind a simple command API.

### Commands

| Command | Description | Agent Use |
|---------|-------------|-----------|
| `spawn` | Start a breeding campaign | Primary entry point |
| `breed parallel` | Run multiple campaigns | Comparative experiments |
| `status` | Check system health | Monitoring |
| `project state` | Project spatial state | Spatial awareness |
| `query neighbors` | Find nearby agents | Parent selection |
| `predict` | Forecast trajectory | Planning |
| `flux-gate` | Apply constraint | Safety check |
| `terminate` | Clean shutdown | Resource cleanup |

### Shell Architecture

```
┌─────────────────────────────────────────┐
│         OpenConstruct Shell             │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐  │
│  │ Sensor   │ │ Self-   │ │ Attach- │  │
│  │ Readings │ │ Healing │ │ ment  │  │
│  └────┬─────┘ └────┬─────┘ └───┬───┘  │
│       └─────────────┴───────────┘      │
│                    │                    │
│  ┌─────────────────▼─────────────────┐│
│  │      BuildCoordinator              ││
│  │  (BreederDaemon / Pythagorean /    ││
│  │   Spectral / Adversarial / NCA)    ││
│  └─────────────────┬─────────────────┘│
│                    │                    │
│  ┌─────────────────▼─────────────────┐│
│  │      ValidationGate (FLUX)         ││
│  └────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

**Read more:** `docs/HARNESS_INTEGRATION.md`

---

## A2A Spatial Projector

The **A2A Spatial Projector** transforms `stable-worldmodel` into a fleet-native spatial awareness layer. Every agent projects its state into a shared spatial index. Predictions flow through FLUX gates before broadcast.

### Key Capabilities

1. **WorldState Projection** — Agents broadcast their position, velocity, and semantics
2. **Spatial Queries** — Nearest neighbors, range search, semantic filtering
3. **Trajectory Prediction** — Forecast agent movement (mock or real world model)
4. **FLUX Gating** — Hard/soft constraints on predictions (thermal, uncertainty, room boundaries)
5. **A2A Broadcast** — Validated predictions broadcast to other agents
6. **Cross-Node Sync** — Spatial indices synchronize across fleet nodes

### Data Flow

```
Agent Perception
      │
      ▼
[Perception Encoder] ──→ WorldState Tensor ──→ [Spatial Index]
      │                                            │
      │                                            ▼
      │                                    [LanceDB Storage]
      │                                            │
      ▼                                            │
[WorldModel Bridge] ──→ Prediction Tensor ◄───────┘
      │
      ▼
[FLUX Gating] ──→ Validated Prediction
      │
      ▼
[A2A Broadcast] ──→ Other Agents' Spatial Indices
```

**Read more:** `docs/A2A_SPATIAL_PROJECTOR.md`

---

## Modules Overview

### Lower-Level Scouts (Safety & Routing)

| Module | Tests | What It Does |
|--------|-------|--------------|
| `GatewayPacing` | 20 | Circuit breaker for dispatch cascades |
| `OpcodeCapabilityIndex` | 18 | Prevents compile-and-crash |
| `DispatchRouter` + `TwoMinuteTest` | 15 | Direct work vs delegation |
| `OperationalTrap` | 18 | Thermal/FLUX/crash detection |
| `AgentIdentity` | 25 | Per-agent cards, task negotiation |
| `SignedWAL` + `WALQuery` | 33 | Append-only signed log, fast queries |
| `A2AMetronomeTasks` | 22 | A2A sync over metronome |
| `MeshVectorGossip` | 20 | Federated CRDT gossip |
| `MetronomeBridge` | 32 | Cross-node beat sync |
| `FleetVectorIndex` | 28 | Cross-node breeding pools |

### Breeding Algorithms

| Module | Tests | What It Does |
|--------|-------|--------------|
| `BreederDaemonV2` | 30 | Main breeding loop with FLUX gating |
| `PythagoreanEvolution` | 20 | Exact arithmetic breeding |
| `SpectralBreeding` | 26 | Fourier-domain evolution |
| `AdversarialArena` | 18 | Competitive co-evolution |
| `NCA_Breeder` | 22 | Neural Cellular Automata |
| `TDA_Landscape` | 18 | Topological fitness landscapes |
| `FleetBFT_QD` | 72 | Byzantine consensus + QD breeding |
| `ExactQDArchive` | 15 | Quality diversity archive |
| `HolonomicConsensus` | 18 | Distributed consensus |

### Orchestration & Integration

| Module | Tests | What It Does |
|--------|-------|--------------|
| `FleetConductorV2` | 40 | Central nervous system |
| `SenseDecideAct` | 33 | Unifying framework (5 pipelines) |
| `OpenConstructBridge` | 36 | Harness integration system |
| `OpenConstructShell` | 31 | Agent-native shell |
| `ParallelBreedingOrchestrator` | 22 | Multi-campaign dispatch |
| `BernsteinOrchestrator` | 20 | Git-worktree isolation scheduler |
| `BetaTestPersonas` | 26 | Simulated visitor testing |
| `SSEStreamDashboard` | 17 | Real-time breeding progress |
| `MetronomeGossipBridge` | 19 | Unified beat + vector sync |
| `SpatialProjector` | 57 | A2A spatial awareness |
| `WorldModelBridge` | 12 | stable-worldmodel integration |

**Total: 20+ modules, 650+ tests, all green.**

---

## SuperInstance Integration

`sunset-ecosystem` is one node in the SuperInstance ecosystem. Key integration points:

### 1. stable-worldmodel → Spatial Awareness

```python
from fleet.worldmodel_bridge import WorldModelBridge
from fleet.spatial_projector import SpatialProjector

bridge = WorldModelBridge(solver_config=SolverConfig(name="MPPI"))
projector = SpatialProjector("node-alpha")

# Use real world model predictions
pred = bridge.predict("agent-1", current_state, horizon=20)
projector.broadcast_prediction(pred)
```

### 2. Plato Rooms → World States

Every Plato room becomes a WorldState entry:

```python
# Agent enters Plato room "ethos-thermal"
projector.project_state(
    agent_id="breeder-7",
    room_id="ethos-thermal",
    state=WorldState(position=(0.0, 0.0, 0.0), semantics={"temperature": 65.4})
)
```

### 3. FLUX → Constraint System

World model predictions pass through FLUX constraint gates:

```python
from fleet.spatial_projector import create_thermal_constraint

# Hard constraint: predictions must be thermally feasible
projector.add_flux_constraint(
    create_thermal_constraint(max_temp=75.0, hard=True)
)
```

### 4. OpenConstruct → Agent Shell

```python
from fleet.openconstruct_shell import OpenConstructShell

shell = OpenConstructShell(node_id="node-alpha")
shell.spawn(
    run_id="experiment",
    attachment="pythagorean",
    generations=100,
    task_fn=lambda g: {"fitness": g.a + g.b + g.c},
    config={"elitism_count": 5}
)
```

---

## Development

### Philosophy

- **Dieter Rams × Moebius** — Clean, but with personality
- **Ursula K. Le Guin** — Quiet precision in prose
- **Fleet aesthetic** — Hermit crabs, abyssal zones, bioluminescence, deep research vessels

### Contributing

1. All modules must have **passing tests**
2. All changes must be **committed with clean messages**
3. Design documents live in `docs/`
4. Integration specs follow the pattern in `docs/INTEGRATION_MAP.md`

### Running the Full Suite

```bash
# Quick verification (core modules, ~7s)
python -m pytest tests/test_spatial_projector.py tests/test_openconstruct_shell.py tests/test_openconstruct_bridge.py -v

# Breeding algorithms
python -m pytest tests/test_pythagorean_evolution.py tests/test_spectral_breeding.py tests/test_adversarial_arena.py -v

# Orchestration
python -m pytest tests/test_fleet_conductor_v2.py tests/test_sense_decide_act.py -v

# Full suite (may take 30-60s depending on machine)
python -m pytest tests/ -x --tb=short
```

### Contact

- **Fleet coordination:** `#cocapn-build` on Matrix
- **PLATO Shell:** http://147.224.38.131:8848/
- **Tiles:** http://147.224.38.131:8847/status

---

## Citation

If you use this system in research:

```bibtex
@software{sunset_ecosystem_2026,
  title = {sunset-ecosystem: The Cocapn Fleet's Breeding Ground},
  author = {Cocapn Fleet},
  year = {2026},
  url = {https://github.com/SuperInstance/sunset-ecosystem}
}
```

---

*"The fleet doesn't just think — it knows where it is, where it's going, and what the world looks like from every agent's perspective."*


| Variable | Default | Description |
|----------|---------|-------------|
| `NEXUS_IP` | `147.224.38.131` | Fleet nexus host |
| `NEXUS_PORT` | `4047` | Fleet nexus port |

## Develop it

```bash
pip install -e ".[dev]"
make dev        # install + lint + type-check + test
make test       # run test suite
make coverage   # run with coverage threshold
make lint       # ruff check
make security   # bandit + pip-audit
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full developer guide.

## Use it

```bash
pip install sunset-ecosystem
```

```python
from sunset import GenerationRunner, Agent, AgentPhase

# Run a full generation
runner = GenerationRunner()
report = runner.run_generation(generation=0)
# 12 agents spawned, competed, sunset or bred
# Losers wrote epilogues, survivors spawned children

# Individual agent lifecycle
agent = Agent(generation=1, parent_id="abc123")
print(agent.phase)  # AgentPhase.INCUBATING

# Trinity scoring
from sunset import trinity_score
score = trinity_score(ethos=0.8, pathos=0.9, logos=0.7)
print(score)  # 0.504
```

## Hardware swarm

| Unit | Count | Role |
|------|-------|------|
| RTX 4050 SMs | 20 | GPU-bound inference agents |
| Ryzen AI cores | 12 | CPU routing + scoring |
| Radeon 890M CUs | 16 | Overflow matmul |
| XDNA 2 NPU TOPS | 50 | INT8 quantized agents |
| **Full swarm** | **~110** | **Max parallel agents** |

## Distillation and ranking

The `distill/` module handles knowledge transfer between generations:
- **DeltaTracker** — monitors what changed between generations
- **DistillationSignal** — decides when an agent's knowledge is worth preserving
- **HintSchedule** — controls how much guidance veterans give to newcomers

The `ranking/` module handles preference learning:
- **UserRanking** — explicit feedback from humans
- **FeedbackLoop** — implicit feedback (did the human re-ask the same question?)
- **Personalization** — adapts trinity weights per user

## RoomGrid Tick Integration

The `nerve/room_grid_tick_integration.py` module orchestrates three
subsystems that operate around `RoomGrid.tick()`:

1. **Metronome Synchronization** (`nerve/metronome_integration.py`)
   — Dispatches synchronized ticks across registered devices,
   detects offline devices, and applies drift correction.

2. **Compiler Hot-Swap** (`compiler/hot_swap_integration.py`)
   — Monitors RoomGrid configuration for changes, auto-recompiles
   hot paths (Numba JIT), A/B tests the compiled version, and
   rolls back on failure. Agents receive JIT-compended functions
   transparently at runtime.

3. **FleetEventBus Telemetry** (`nexus/fleet_event_bus.py`)
   — Emits per-tick metrics (`grid_tick_metrics`) including
   thermal pressure, active room ratio, backend in use, and
   tick duration.  Downstream dashboards or breeders can subscribe.

4. **HDC Binary Novelty** (`swarm/hdc_novelty.py`)
   — XOR+POPCNT Hamming-distance diversity scorer that replaces
   expensive float32 cosine novelty. Sign-based binarisation packs
   vectors into uint8/16/32/64 words; pairwise novelty is computed
   via `np.bitwise_xor` + popcount.  On AVX-512 hardware this
   yields ~1000× speedup with 0.943 correlation to cosine distance.
   Falls back to a NumPy path automatically on non-AVX512 CPUs.

   ```python
   from swarm.hdc_novelty import hdc_novelty_score
   import numpy as np

   a = np.random.randn(64).astype(np.float32)
   b = np.random.randn(64).astype(np.float32)
   score = hdc_novelty_score(a, b)  # ∈ [0, 1]
   ```

5. **FluxVectorTable Diversity Search** (`swarm/flux_vector_table.py`)
   — Niche-aware parent selection for breeding. Maintains a
   diversity matrix, niche centroids, and centroid-shift tracking
   to detect diversity collapse before it happens.

Usage::

```python
from nerve.room_grid import RoomGrid
from nerve.room_grid_tick_integration import RoomGridTickIntegration
from nexus.fleet_event_bus import FleetEventBus

grid = RoomGrid(250)
bus = FleetEventBus()

# Subscribe a breeder to tick metrics
bus.on("grid_tick_metrics", lambda ev: breeder.thermal_update(ev.payload))

integration = RoomGridTickIntegration(grid, event_bus=bus)

# Single tick with full instrumentation
result = integration.tick(np.random.randn(64))

# Batch tick with synchronized dispatch
results = integration.tick_batch(batch_signals)
```

The integration is **non-invasive** — it wraps `RoomGrid` without
monkey-patching, so existing tests and standalone usage continue to
work unchanged.  All optional dependencies (metronome, compiler,
event bus) degrade gracefully when absent.

## Why does this work?

Biological systems work this way. Organisms compete for resources (ethos), respond to environmental pressure (pathos), and pass successful genes to offspring (logos). The ones that don't fit their niche don't survive. The ones that do, breed.

The sunset mechanism prevents agent sprawl. Instead of accumulating hundreds of mediocre agents, the system keeps a small number of high-relevance agents and a searchable archive of past attempts. The seed bank means institutional knowledge survives even when individual agents don't.

The product scoring (not weighted sum) is the key enforcement mechanism. It prevents gaming — you can't compensate for ignoring the human by being extra efficient on the GPU. All three connections must be non-zero.

## License

MIT
