# sunset-ecosystem 🌅

> *Agents that breed, vote, sunset with dignity, and seed the next generation — governed by a trinity of ethos (metal), pathos (human), and logos (code).*

[![Tests](https://img.shields.io/badge/tests-8729%20passing-brightgreen)](./tests)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](./pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-yellow)](./LICENSE)
[![Modules](https://img.shields.io/badge/modules-29-orange)](./)
[![Files](https://img.shields.io/badge/source_files-1028-blueviolet)](./)

---

## Table of Contents

- [What Is This?](#what-is-this)
- [Architecture](#architecture)
- [30-Second Quickstart](#30-second-quickstart)
- [Module Guide](#module-guide)
  - [ethos/ — Metal Surveyor](#ethos--metal-surveyor)
  - [pathos/ — Human Interface](#pathos--human-interface)
  - [logos/ — Decisions, Audit, Memory](#logos--decisions-audit-memory)
  - [nerve/ — Forward Inference Engine](#nerve--forward-inference-engine)
  - [swarm/ — Breeding, Diversity, Selection](#swarm--breeding-diversity-selection)
  - [sunset/ — Agent Lifecycle & Compilation](#sunset--agent-lifecycle--compilation)
  - [nexus/ — Fleet Coordination](#nexus--fleet-coordination)
  - [fleet/ — Operational Bridges (263 files)](#fleet--operational-bridges)
  - [triage/ — Drift & Duplicate Detection](#triage--drift--duplicate-detection)
  - [perception/ — Vision & Audio Encoding](#perception--vision--audio-encoding)
  - [compiler/ — Hot-Swap Compilation](#compiler--hot-swap-compilation)
  - [a2a/ — Agent-to-Agent Protocol](#a2a--agent-to-agent-protocol)
  - [reasoning/ — Polyglot Reasoning](#reasoning--polyglot-reasoning)
  - [ranking/ — User Ranking & Personalization](#ranking--user-ranking--personalization)
  - [distill/ — Distillation Signals](#distill--distillation-signals)
  - [flux_vm/ — FLUX Constraint VM](#flux_vm--flux-constraint-vm)
  - [flux_compat/ — FLUX Compatibility Layer](#flux_compat--flux-compatibility-layer)
  - [jepa/ — JEPA Predictive World Model](#jepa--jepa-predictive-world-model)
  - [grammar/ — Grammar Engine](#grammar--grammar-engine)
- [SuperInstance Ecosystem Integration](#superinstance-ecosystem-integration)
- [Conservation Law](#conservation-law)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## What Is This?

**sunset-ecosystem** is the mothership repository of the SuperInstance fleet. It is an evolutionary, self-governing agent ecosystem built on a **trinity architecture**:

| Dimension | Domain | Question |
|-----------|--------|----------|
| **Ethos** | Metal, hardware, thermals | *Is the agent using hardware efficiently?* |
| **Pathos** | Human needs, cognitive load | *Does the agent actually help people?* |
| **Logos** | Code, audit, reasoning | *Is the decision traceable and correct?* |

Every agent carries a composite score: `ethos × pathos × logos`. Drop to zero in any dimension and the fleet sunsets you. Survive, and you might be selected as a parent for the next breeding cycle.

**How it works, in one paragraph:**

Agents are born as eggs, enter competition, and are scored by the trinity. Winners survive and breed. Losers sunset gracefully — they decompose into seeds that inform the next generation. Breeding decisions are ratified by Practical Byzantine Fault Tolerant (PBFT) consensus across fleet nodes, combined with Quality-Diversity (MAP-Elites) evolutionary algorithms so the fleet explores, not just exploits. Thermal auctions (VCG mechanism) allocate compute slots truthfully. The entire lifecycle is logged to an append-only SQLite WAL and reconstructed on restart.

**This is not a chatbot wrapper.** It is infrastructure for running hundreds of agents across heterogeneous hardware, with exact mathematical constraint satisfaction (Pythagorean manifold snapping, Eisenstein hex-lattice directions, holonomy verification), polyglot reasoning (Python / Rust / C++ / Mercury / C), and a live bioluminescent dashboard showing thermal pressure, diversity metrics, and breeding progress in real time.

**By the numbers:**
- **8,729 tests** across 471 test files
- **1,028 source files** in 29 modules
- **263 files** in `fleet/` alone — bridges to every ecosystem pattern
- **5 languages** for reasoning: Python, Rust, C++, Mercury, C
- **VCG combinatorial auction** for truthful thermal slot allocation
- **PBFT consensus** with O(n²) message complexity, tolerating f < N/3 Byzantine faults
- **MAP-Elites quality diversity** for population exploration
- **Spectral breeding** — genomes as complex Fourier spectra, crossover as spectral convolution
- **Pythagorean evolution** — genetic alphabet is exact Pythagorean triples, no floating-point drift

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Sunset Ecosystem                                  │
│            Trinity: ethos × pathos × logos > 0                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │   ETHOS     │ │   PATHOS    │ │   LOGOS     │ │  TRIAGE     │       │
│  │ hardware_   │ │ need_tracker│ │ decision_   │ │ drift_detect│       │
│  │  survey     │ │ moment_     │ │  journal    │ │ duplicate_  │       │
│  │ stress_test │ │  scorer     │ │ mmap_wal    │ │  detect     │       │
│  │ thermal_    │ │ interaction │ │ generation_ │ │ weekly      │       │
│  │  auto_cal   │ │  _log      │ │  memory     │ │ metrics     │       │
│  │ agent_alloc │ │ trinity_    │ │ intent_     │ │ github_     │       │
│  │             │ │  connection │ │  protocol   │ │  issues     │       │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘       │
│         │               │               │               │               │
│         └───────────────┴───────┬───────┴───────────────┘               │
│                                 │                                         │
│                                 ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                        SUNSET (Lifecycle)                         │   │
│  │  AgentPhase: EGG → COMPETE → SURVIVE → BREED → SUNSET → ARCHIVE │   │
│  │  compiler · flux_ast_compiler · flux_vm_bridge · codegen         │   │
│  │  generation_runner · health_thermal_bridge · plato_bridge        │   │
│  └──────────────────────────────┬───────────────────────────────────┘   │
│                                 │                                         │
│         ┌───────────────────────┼───────────────────────┐                │
│         ▼                       ▼                       ▼                │
│  ┌─────────────┐ ┌─────────────────────┐ ┌─────────────────────┐       │
│  │   NERVE     │ │       SWARM          │ │      NEXUS          │       │
│  │ room_grid   │ │ breeder_daemon_v2    │ │ fleet_conductor_v2  │       │
│  │ fiber       │ │ fleet_bft_qd         │ │ fleet_event_bus     │       │
│  │ metronome   │ │ thermal_auction (VCG)│ │ distributed_        │       │
│  │ cuda_bridge │ │ tournament           │ │  consensus          │       │
│  │ jepa_ffi    │ │ spectral_breeding    │ │ federation          │       │
│  │ bench_      │ │ pythagorean_evo      │ │ holonomy_bridge     │       │
│  │  topology   │ │ cellular_engine      │ │                     │       │
│  │ bloom_      │ │ mesh_vector_tables   │ │                     │       │
│  │  filter     │ │ constraint_bridge    │ │                     │       │
│  └──────┬──────┘ └──────────┬───────────┘ └──────────┬──────────┘       │
│         │                   │                         │                   │
│         └───────────────────┼─────────────────────────┘                   │
│                             │                                             │
│                             ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     FLEET (Operational Layer)                      │   │
│  │  sense_decide_act · bernstein_orchestrator · health_bridge        │   │
│  │  agent_identity_bridge · swarm_coordinator_bridge                 │   │
│  │  fence_board_bridge · heartbeat_bridge · fleet_task_board_bridge  │   │
│  │  fleet_consciousness_bridge · sse_stream_dashboard                │   │
│  │  work_dashboard · beta_test_personas · ab_tester                  │   │
│  │  conservation_spectral_bridge · worldmodel_bridge                 │   │
│  │  ... 263 files total ...                                          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                             │                                             │
│                             ▼                                             │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌──────────┐  │
│  │ COMPILER  │ │PERCEPTION │ │ REASONING │ │  A2A      │ │ FLUX_VM  │  │
│  │ hot_swap  │ │ vision_   │ │ python_   │ │ server    │ │ flux_    │  │
│  │ A/B gate  │ │  encoder  │ │  bridge   │ │ identity  │ │  compat  │  │
│  └───────────┘ │ audio_    │ │ polyglot  │ │ protocol  │ │ opcode   │  │
│                │  encoder  │ │ 6 lang    │ │ plugin_   │ │  map     │  │
│  ┌───────────┐ └───────────┘ └───────────┘ │  manager  │ └──────────┘  │
│  │ RANKING   │ ┌───────────┐ ┌───────────┐ └───────────┘ ┌──────────┐  │
│  │ user_     │ │  JEPA     │ │ GRAMMAR   │ ┌───────────┐ │ DISTILL  │  │
│  │  ranking  │ │ jepa_room │ │ engine    │ │  VOICE    │ │ delta_   │  │
│  │ personali │ │ predictiv │ │ spec      │ │ soniqo    │ │  tracker │  │
│  │  zation   │ │  e model  │ │ security  │ │  bridge   │ │ signal   │  │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘ └──────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 30-Second Quickstart

```bash
# Clone
git clone https://github.com/SuperInstance/sunset-ecosystem.git
cd sunset-ecosystem

# Install (dev mode — includes pytest, ruff, mypy)
pip install -e ".[dev]"

# Run the full test suite (8729 tests, ~90s)
pytest
# → all green

# Run a specific module's tests
pytest tests/test_breeder_daemon_v2.py -v
pytest tests/test_fleet_bft_qd.py -v
pytest tests/test_thermal_auction.py -v

# Watch the fleet breed
python3 examples/constraint_breeding.py

# Launch the tide pool dashboard
python3 logos/tide_pool_viz.py
# → http://localhost:8080/tide-pool

# Polyglot reasoning across 6 languages
python3 examples/polyglot_reason.py

# Voice-enabled agent room
python3 examples/voice_room.py
```

---

## Module Guide

### ethos/ — Metal Surveyor

> *Score how connected an agent is to the hardware it runs on.*

The ethos module measures whether an agent is using compute efficiently. A high ethos score means: compute is utilized well, latency is within hardware capability, resource usage fits the thermal budget, and memory footprint is appropriate.

**Files:** 6 source files

| File | Purpose |
|------|---------|
| `hardware_survey.py` | Probe CPU, GPU, memory, thermal capacity → `HardwareProfile` |
| `stress_test.py` | Run synthetic loads, measure degradation → `StressReport` |
| `thermal_auto_calibrate.py` | Auto-calibrate thermal budgets from stress test results |
| `agent_allocator.py` | Assign agents to hardware slots based on profile match |
| `trinity_connection.py` | Score (0.0–1.0) how well-connected an agent is to ETHOS |

```python
from ethos.hardware_survey import HardwareProfile, survey_hardware
from ethos.stress_test import StressTest, StressReport
from ethos.thermal_auto_calibrate import auto_calibrate
from ethos.trinity_connection import score_ethos_connection, EthosConnectionScore

# Survey the hardware
profile = survey_hardware()
print(f"CPU: {profile.cpu_cores} cores, {profile.cpu_freq_mhz} MHz")
print(f"GPU: {profile.gpu_name}, {profile.gpu_memory_mb} MB")
print(f"Thermal headroom: {profile.thermal_headroom:.1f}W")

# Stress test
test = StressTest(profile)
report: StressReport = test.run()
print(f"Max sustained load: {report.max_sustained_ops} ops/s")
print(f"Thermal throttle at: {report.throttle_temp_c}°C")

# Auto-calibrate budgets
budgets = auto_calibrate(profile, report)

# Score an agent's ethos connection
score: EthosConnectionScore = score_ethos_connection(
    hardware_efficiency=0.85,
    latency_fit=0.92,
    thermal_fit=0.78,
    memory_fit=0.90,
)
print(f"Ethos: {score.total:.3f}")
# → Ethos: 0.862
```

---

### pathos/ — Human Interface

> *Score how connected an agent is to the humans it serves.*

The pathos module evaluates whether an agent actually solves human problems, produces directly useful output, and reduces cognitive load. This is the "does this matter to people?" dimension.

**Files:** 5 source files

| File | Purpose |
|------|---------|
| `need_tracker.py` | Track human needs and fulfillment → `NeedState` |
| `moment_scorer.py` | Score peak moments of human-agent interaction → `MomentScore` |
| `interaction_log.py` | Log human-agent interactions with timestamps |
| `trinity_connection.py` | Composite pathos score → `TrinityScore` |

```python
from pathos.need_tracker import NeedState
from pathos.moment_scorer import MomentScore
from pathos.trinity_connection import TrinityScore, score_pathos_connection

# Define human needs
needs = NeedState(
    problems_identified=["reduce noise in pipeline", "faster deployment"],
    problems_solved=["reduce noise in pipeline"],
    cognitive_load_before=0.8,
    cognitive_load_after=0.3,
)

# Score a moment of interaction
moment = MomentScore(
    directly_useful=True,
    reduced_friction=True,
    surprise_insight=False,
)

# Get composite pathos score
score: TrinityScore = score_pathos_connection(needs, moment)
print(f"Solves problems: {score.solves_human_problems:.2f}")
print(f"Directly useful: {score.directly_useful:.2f}")
print(f"Reduces cognitive load: {score.reduces_cognitive_load:.2f}")
print(f"Composite: {score.composite:.2f}")
# → Composite: 0.72
```

---

### logos/ — Decisions, Audit, Memory

> *Every significant fleet decision is logged, queryable, and tamper-evident.*

The logos module is the fleet's memory. It provides structured decision journals in FLAME format (Fleet Log of Agent Memory and Explanation), memory-mapped write-ahead logs, generation memory for breeding lineage tracking, and an intent protocol for fleet state communication.

**Files:** 18 source files

| File | Purpose |
|------|---------|
| `decision_journal.py` | FLAME-format structured decision log → `DecisionJournal` |
| `decision_log.py` | Append-only decision log with timestamps |
| `mmap_wal.py` | Memory-mapped write-ahead log for crash recovery |
| `generation_memory.py` | Track breeding lineage across generations |
| `intent_protocol.py` | Fleet state communication (FleetState, IntentConfirmationProtocol) |
| `codebase_state.py` | Snapshot the state of the codebase for audit |
| `compression_utils.py` | Compress large log entries |
| `a2a_identity.py` | Agent identity within the logos namespace |
| `a2a_protocol.py` | A2A protocol adapter for decision sharing |

```python
from logos.decision_journal import (
    DecisionJournal,
    Decision,
    log_spawn,
    log_sunset,
    log_breed,
    log_human_command,
)
from logos.generation_memory import GenerationMemory
from logos.intent_protocol import FleetState

# Log a decision
journal = DecisionJournal(path="/tmp/fleet-journal.jsonl")
journal.log(
    Decision(
        timestamp=time.time(),
        agent_id="agent-007",
        action="breed",
        rationale="Pareto non-dominated for 3 generations",
        metadata={"parent_a": "agent-003", "parent_b": "agent-005"},
    )
)

# Query history
history = journal.query(agent_id="agent-007", action="breed")
for d in history:
    print(f"[{d.timestamp}] {d.agent_id} {d.action}: {d.rationale}")

# Convenience functions
log_spawn("agent-042", metadata={"parent": "agent-007"})
log_breed("agent-007", "agent-042", rationale="high trinity score")
log_sunset("agent-003", reason="zero pathos for 5 ticks")
log_human_command("increase thermal budget to 80W")

# Generation memory
gm = GenerationMemory()
gm.record(parents=["agent-007", "agent-005"], child="agent-042", generation=12)
lineage = gm.get_lineage("agent-042")
print(f"Ancestors: {[a.agent_id for a in lineage]}")
```

---

### nerve/ — Forward Inference Engine

> *Hardware-aware adaptive forward engine. Each room = 3.4K params. No training, no backprop.*

The nerve module provides the forward inference backbone of the fleet. It auto-detects the fastest available backend (CUDA → Rust persistent → Rust oneshot → NumPy einsum fallback), manages room grid topology for agents, and routes events through fiber pathways that perceive, adapt, compile, and alert on novelty.

**Files:** 24 source files

| File | Purpose |
|------|---------|
| `room_grid.py` | Adaptive forward engine with CUDA/Rust/NumPy backends → `RoomGrid` |
| `fiber.py` | Sensory pathway: PERCEIVING → ADAPTING → COMPILED → NOVELTY_ALERT → PERCEIVING |
| `metronome.py` | Periodic pulse generator wrapping `RoomGrid.tick()` |
| `metronome_bridge.py` | Bridge to fleet metronome for cross-node beat sync |
| `distributed_metronome_bridge.py` | PID drift correction for cross-node synchronization |
| `cuda_bridge.py` | Persistent CUDA grid — weights in GPU memory, zero-copy per tick |
| `jepa.py` | JEPA forward inference integration |
| `jepa_ffi.py` | FFI bindings for JEPA Rust implementation |
| `jepa_rust.py` | Rust-backed JEPA inference |
| `bench_topology.py` | Benchmark different room topologies |
| `bloom_filter_wrapper.py` | Bloom filter for fast membership in routing tables |
| `adaptation.py` | Adaptive weight adjustment for room parameters |
| `a2a_conductor_integration.py` | A2A conductor hooks for nerve events |
| `a2a_metronome_tasks.py` | Metronome task schema for A2A protocol |

**Backend auto-detection (from `room_grid.py`):**

```
Priority:
  1. CUDA          — GPU kernel (requires nvcc + CUDA runtime)
  2. Rust Persistent — weights in Rust memory, zero-copy per tick
  3. Rust Oneshot   — legacy ctypes (overhead kills small arrays)
  4. NumPy          — pure einsum fallback (always works)
```

```python
from nerve.room_grid import RoomGrid, JEPAGrid, make_weights, novelty
from nerve.fiber import NerveFiber, FiberState, SensoryTile
from nerve.metronome import MetronomeScheduler

# Create a room grid (auto-detects best backend)
grid = RoomGrid(n_rooms=100, dim=256)
print(f"Backend: {grid.backend}")  # "cuda", "rust_persistent", or "numpy"

# Forward pass
weights = make_weights(dim=256)
signal = np.random.randn(256).astype(np.float32)
output = grid.tick(signal, weights)
print(f"Output shape: {output.shape}")  # (256,)

# Batch forward pass
signals = np.random.randn(50, 256).astype(np.float32)
outputs = grid.tick_batch(signals, weights)

# Compute novelty between tiles
tile_a = np.random.randn(256).astype(np.float32)
tile_b = np.random.randn(256).astype(np.float32)
score = novelty(tile_a, tile_b)
print(f"Novelty: {score:.4f}")  # 0.0 = identical, 1.0 = maximally different

# Fiber: sensory pathway with lifecycle
fiber = NerveFiber(dim=256)
fiber.perceive(SensoryTile(data=signal))
print(f"State: {fiber.state}")  # PERCEIVING
fiber.adapt()
print(f"State: {fiber.state}")  # ADAPTING
fiber.compile()
print(f"State: {fiber.state}")  # COMPILED

# Metronome: periodic pulse
scheduler = MetronomeScheduler(bpm=120)
scheduler.on_tick(lambda tick: grid.tick(signal, weights))
scheduler.start()
```

---

### swarm/ — Breeding, Diversity, Selection

> *88 source files. The evolutionary engine of the entire fleet.*

The swarm module implements the breeding loop, diversity maintenance, tournament selection, thermal auctions, consensus voting, and every evolutionary algorithm variant the fleet uses. This is the largest and most complex module.

**Files:** 88 source files

#### Core Breeding

| File | Purpose |
|------|---------|
| `breeder_daemon_v2.py` | Persistent lifecycle FSM: EGG → COMPETE → SURVIVE → BREED → SUNSET → ARCHIVE |
| `breeder_fsm_v2.py` | FSM implementation with WAL-based state recovery |
| `lifecycle_fsm.py` | Agent lifecycle states and transitions |
| `breeder.py` | Legacy auto-breeder (compatibility shim) |
| `adaptive_breeder.py` | Adaptive breeding rates based on population health |
| `causal_breeder.py` | Causal inference for breeding decisions |
| `meta_breeder.py` | Breeds breeding strategies (meta-learning) |
| `meta_learning_breeder.py` | Online learning for breeding hyperparameters |
| `breeding_kernel.py` | Core breeding operators (crossover, mutation, selection) |
| `breed_optimizer.py` | Optimize breeding hyperparameters |

#### Quality Diversity & Selection

| File | Purpose |
|------|---------|
| `fleet_bft_qd.py` | PBFT consensus + MAP-Elites quality diversity |
| `tournament.py` | Head-to-head tournament across ethos/pathos/logos axes |
| `spectral_breeding.py` | Genomes as complex Fourier spectra, crossover = spectral convolution |
| `pythagorean_evolution.py` | Genetic alphabet = exact Pythagorean triples, no FP drift |
| `swarm_intelligence_breeder.py` | Swarm intelligence algorithms for breeding |
| `neural_topology_breeding.py` | Neural architecture search via breeding |
| `nca_breeder.py` | Neural Cellular Automata breeding |
| `info_theoretic_breeder.py` | Information-theoretic diversity measures |
| `information_geometry_breeding.py` | Natural gradient breeding on information manifolds |
| `tda_landscape.py` | Topological Data Analysis of fitness landscapes |

#### Thermal & Resource Management

| File | Purpose |
|------|---------|
| `thermal.py` | Thermal budget management and throttling |
| `thermal_auction.py` | VCG combinatorial auction for truthful slot allocation |
| `async_thermal.py` | Async thermal budget tracking |
| `inheritance_tax.py` | Resource cost for spawning children (prevents exponential growth) |
| `priority_queue.py` | Priority-based breeding queue |
| `worker_pool.py` | Worker pool for parallel breeding tasks |

#### Mesh & Federation

| File | Purpose |
|------|---------|
| `mesh_vector_tables.py` | Federated CRDT-based vector tables for shared fleet cognition |
| `mesh_vector_gossip.py` | Anti-entropy gossip protocol for vector table sync |
| `mesh_grouping.py` | Group agents into mesh clusters |
| `mesh_table_store.py` | Persistent storage for mesh tables |
| `mesh_wal.py` | Write-ahead log for mesh operations |
| `tiered_mesh_storage.py` | Hot/warm/cold tiered storage for mesh data |
| `spectral_mesh_routing.py` | Spectral analysis for mesh routing decisions |
| `arrow_flight_mesh.py` | Apache Arrow Flight integration for high-throughput mesh |
| `arrow_mesh.py` | Arrow-based mesh data exchange |
| `arrow_telemetry.py` | Arrow telemetry pipeline |

#### Constraint Theory

| File | Purpose |
|------|---------|
| `constraint_bridge.py` | Exact Pythagorean snapping via constraint-theory-core (Rust) |
| `simd_ops.py` | SIMD-accelerated breeding operations |
| `penrose.py` | Penrose tiling for constraint-space exploration |

#### Novelty & Diversity

| File | Purpose |
|------|---------|
| `hdc_novelty.py` | Binary hypervector novelty scoring (XOR + POPCNT) |
| `chaos.py` | Chaos injection for diversity maintenance |
| `lineage_checker.py` | Verify lineage constraints (prevent inbreeding) |
| `vector_table.py` | Re-export of `FluxVectorTable` |
| `flux_vector_table.py` | Diversity search and niche exploration |
| `trajectory_monitor.py` | Monitor breeding trajectory for stagnation |

```python
from swarm.breeder_daemon_v2 import BreederDaemonV2, LifecycleState
from swarm.thermal_auction import VCGAuction, Bid
from swarm.tournament import AgentScore, dominated_by, sunset_candidates
from swarm.spectral_breeding import SpectralBreeder, SpectralGenome
from swarm.pythagorean_evolution import PythagoreanBreeder, PythagoreanGenome
from swarm.fleet_bft_qd import PBFTNode, PBFTMessage
from swarm.mesh_vector_tables import FleetVectorIndex
from swarm.constraint_bridge import ConstraintBridge

# --- Lifecycle FSM ---
daemon = BreederDaemonV2(dim=256, max_agents=50)
agent_id = daemon.spawn(initial_vector=np.random.randn(256))
print(f"State: {daemon.state_of(agent_id)}")  # LifecycleState.EGG
daemon.transition(agent_id, LifecycleState.COMPETE)
daemon.transition(agent_id, LifecycleState.SURVIVE)
daemon.transition(agent_id, LifecycleState.BREED)
child_id = daemon.breed(agent_id, partner_id="other-agent")
daemon.transition(agent_id, LifecycleState.SUNSET)
# Agent decomposes into seed for next generation

# --- VCG Thermal Auction ---
auction = VCGAuction(device_type="GPU", slots=4)
bids = [
    Bid(agent_id="a1", value=0.9),
    Bid(agent_id="a2", value=0.8),
    Bid(agent_id="a3", value=0.7),
    Bid(agent_id="a4", value=0.6),
    Bid(agent_id="a5", value=0.5),  # loses
]
allocation = auction.resolve(bids)
print(f"Winners: {[b.agent_id for b in allocation.winners]}")
print(f"Prices: {allocation.prices}")  # Each pays the externality they impose

# --- Tournament Selection ---
scores = [
    AgentScore(agent_id="a1", ethos=0.9, pathos=0.8, logos=0.7),
    AgentScore(agent_id="a2", ethos=0.7, pathos=0.9, logos=0.6),
    AgentScore(agent_id="a3", ethos=0.5, pathos=0.5, logos=0.5),
]
dominated = dominated_by(scores[0], scores[2])  # True — a1 dominates a3
to_sunset = sunset_candidates(scores)  # Returns Pareto-dominated agents

# --- Spectral Breeding ---
breeder = SpectralBreeder(population_size=50, spectrum_size=64)
breeder.initialize()
for gen in range(100):
    breeder.step(task_fn=lambda phenotype: -np.sum(phenotype**2))
print(f"Best fitness: {breeder.best_fitness():.4f}")

# --- Pythagorean Breeding (exact arithmetic) ---
pb = PythagoreanBreeder(population_size=50, genome_length=10)
pb.initialize()
for gen in range(100):
    pb.step(fitness_fn=lambda genome: sum(t.c for t in genome.triples))
print(f"Best fitness: {pb.best_fitness():.4f}")
# No floating-point drift — every gene is an exact Pythagorean triple

# --- Constraint Bridge (exact snapping) ---
bridge = ConstraintBridge(dim=256, density=200)
exact, noise = bridge.snap_vector([0.577, 0.816])
# exact = [0.6, 0.8] — exact, not 0.9999999999999999
population = [[0.1, 0.9], [0.5, 0.5], [0.3, 0.7]]
exact_pop = bridge.batch_snap(population)

# --- PBFT Consensus ---
node = PBFTNode(node_id="node-1", peers=["node-2", "node-3"])
node.propose("breed_batch_42", payload={"parents": ["a1", "a2"]})
result = node.commit_if_quorum()
```

---

### sunset/ — Agent Lifecycle & Compilation

> *Agents are born, compete, breed, and sunset with dignity.*

The sunset module manages the agent lifecycle, the agentic compiler (runtime-adaptive compilation that watches hot paths and generates optimized implementations), FLUX AST compilation, and the generation runner that orchestrates breeding batches.

**Files:** 21 source files

| File | Purpose |
|------|---------|
| `agent.py` | Agent lifecycle: `AgentPhase` enum, `ResourceBudget`, `SunsetAgent` dataclass |
| `compiler.py` | Agentic Compiler: Profile → Analyze → Compile → Validate → Deploy (hot-swap) |
| `compiler_integration.py` | Integration tests for the agentic compiler |
| `codegen.py` | Code generation for agent behaviors |
| `flux_ast_compiler.py` | FLUX AST → compiled constraints |
| `flux_codegen.py` | FLUX code generation |
| `flux_integration.py` | FLUX integration with sunset lifecycle |
| `flux_preset_library.py` | Pre-built FLUX constraint presets |
| `flux_vm_bridge.py` | Bridge to FLUX VM for constraint execution |
| `generation_runner.py` | Run breeding generations with evaluation |
| `health_thermal_bridge.py` | Bridge health monitoring to thermal decisions |
| `plato_bridge.py` | Bridge to PLATO academy for agent training |
| `roomgrid_plato_observer.py` | Observe room grid state for PLATO integration |
| `cuda_kernels.py` | CUDA kernels for accelerated compilation |

```python
from sunset.agent import SunsetAgent, AgentPhase, ResourceBudget
from sunset.compiler import Compiler, hot_swap

# --- Agent Lifecycle ---
agent = SunsetAgent(
    agent_id="scout-42",
    phase=AgentPhase.INCUBATING,
    budget=ResourceBudget(max_tokens=4096, max_time_seconds=60.0, parallel_slots=2),
)
agent.advance(AgentPhase.COMPETING)
agent.advance(AgentPhase.BREEDING)
agent.advance(AgentPhase.SUNSETTING)  # Decompose into seed
print(f"Phase: {agent.phase}")  # AgentPhase.SUNSETTING
print(f"Epilogue: {agent.epilogue}")  # Final words → seed for next gen

# --- Agentic Compiler ---
compiler = Compiler()
compiler.install()  # Monkey-patches hot paths
# Run your code — the compiler profiles and recompiles automatically
result = some_hot_function(data)
# Compiler may have hot-swapped to a faster backend mid-run


# --- Hot-swap a specific function ---
@hot_swap(backends=["numba", "cuda"])
def expensive_computation(x):
    return np.sum(x**2)


# Compiler selects the fastest available backend at runtime
```

---

### nexus/ — Fleet Coordination

> *Central nervous system for the fleet.*

The nexus module orchestrates all subsystems: metronome sync, federated mesh tables, health monitoring, FLUX presets, A2A identity, circuit breakers, and Sense→Decide→Act pipelines.

**Files:** 9 source files

| File | Purpose |
|------|---------|
| `fleet_conductor_v2.py` | Central orchestrator — lazy init, beat tick, all subsystems |
| `fleet_conductor.py` | Legacy conductor (backward compatible) |
| `fleet_event_bus.py` | Cross-component pub/sub event bus |
| `distributed_consensus.py` | PBFT-style consensus with H¹ cohomology emergence detection |
| `federation.py` | Fleet federation across nodes |
| `holonomy_bridge.py` | Holonomy checking for constraint verification |

```python
from nexus.fleet_conductor_v2 import FleetConductorV2
from nexus.fleet_event_bus import FleetEventBus
from nexus.distributed_consensus import HolonomyConsensus

# --- Fleet Conductor ---
conductor = FleetConductorV2(node_id="node-1")
conductor.initialize()  # Lazy-loads all subsystems
conductor.beat_tick()  # One heartbeat — syncs, breeds, checks health
status = conductor.status()
print(f"Agents: {status['agent_count']}, Breeding: {status['breeding_active']}")

# --- Event Bus ---
bus = FleetEventBus()
bus.subscribe("breed.complete", lambda event: print(f"Bred: {event.data}"))
bus.emit("breed.complete", {"parent_a": "a1", "child": "a42"})

# --- Distributed Consensus ---
consensus = HolonomyConsensus(node_id="node-1", peers=["node-2", "node-3"])
consensus.propose_state_change("room_grid_resize", {"n": 100})
result = consensus.commit_if_quorum()
```

---

### fleet/ — Operational Bridges

> *263 files. Every cross-pollination pattern from the SuperInstance ecosystem, distilled into zero-dependency Python modules.*

The fleet module is the largest in the repository. It bridges patterns from external SuperInstance repos (cocapn-health, git-agent-standard, oracle1-vessel, fleet-consciousness-dashboard, Equipment-Swarm-Coordinator) into first-class sunset-ecosystem citizens.

**Selected highlights (263 files total):**

| File | Purpose |
|------|---------|
| `sense_decide_act.py` | Unifying SENSE → DECIDE → ACT framework, 20 built-in pipelines |
| `bernstein_orchestrator.py` | Deterministic parallel scheduling with git worktree isolation + HMAC audit chain |
| `health_bridge.py` | cocapn-health CheckResult pattern — 18 fleet services, event bus, cache |
| `agent_identity_bridge.py` | git-agent-standard — repo-as-identity, bottles, diary |
| `swarm_coordinator_bridge.py` | Equipment-Swarm-Coordinator — roles, conflict resolution, trust |
| `fence_board_bridge.py` | Tom Sawyer Protocol — tasks as puzzles with challenger ratings |
| `heartbeat_bridge.py` | Fleet heartbeat — registry discovery, task discovery, health checks |
| `fleet_consciousness_bridge.py` | Fleet Consciousness Index (FCI) — weighted health score 0.0–1.0 |
| `fleet_task_board_bridge.py` | Priority, tags, ETA, blocking, org chart for fleet tasks |
| `sse_stream_dashboard.py` | Real-time bioluminescent dashboard via Server-Sent Events |
| `work_dashboard.py` | Unified status from all subsystems |
| `beta_test_personas.py` | 7 simulated visitors test your repo onboarding |
| `conservation_spectral_bridge.py` | Laplacian eigenvalue decomposition for spectral fingerprints |
| `worldmodel_bridge.py` | World model projection for agent planning |
| `ab_tester.py` | A/B testing framework for breeding strategies |
| `auto_scaler.py` | Auto-scale fleet resources based on demand |
| `canary_deployer.py` | Canary deployment for fleet changes |
| `validation_engine.py` | Validate fleet state invariants |
| `vector_clock.py` | Vector clocks for distributed ordering |
| `telemetry.py` | Fleet telemetry collection and export |

```python
# --- Sense → Decide → Act ---
from fleet.sense_decide_act import SDAPipeline, Observation, Decision, ActResult

pipeline = SDAPipeline(name="breeding_check")
obs: Observation = pipeline.sense({"agent": "a1", "score": 0.8})
decision: Decision = pipeline.decide(obs)
result: ActResult = pipeline.act(decision)

# --- Health Checks ---
from fleet.health_bridge import HealthChecker, FLEET_SERVICES

checker = HealthChecker(FLEET_SERVICES)
results = checker.check_all()
print(checker.report(results, format="md"))

# --- Agent Identity ---
from fleet.agent_identity_bridge import AgentVessel, Bottle

vessel = AgentVessel.from_repo("/path/to/agent-repo")
print(vessel.charter.purpose)
vessel.send_bottle(Bottle(to="other-agent", subject="breed request", body="..."))

# --- Fleet Consciousness Index ---
from fleet.fleet_consciousness_bridge import FleetConsciousnessIndex

fci = FleetConsciousnessIndex()
score = fci.compute(
    room_phi_score=0.30, attention_score=0.20, learning_score=0.50, meta_score=0.00
)
print(f"Fleet Consciousness: {score:.3f}")

# --- Fence Board (Tom Sawyer Protocol) ---
from fleet.fence_board_bridge import FenceBoard

board = FenceBoard(max_active=5)
board.post_fence(
    title="Map 16 Viewpoint Opcodes",
    brush="Babel's 16 viewpoint ops are reserved but undefined...",
    view="Your name on 16 opcodes that every FLUX runtime executes",
    challengers={"Babel": 3, "Oracle1": 7},
    reward="0x70-0x7F permanently attributed",
    claim_window_hours=48,
)

# --- Beta Test Personas ---
from fleet.beta_test_personas import PersonaLibrary, BetaTestRunner

library = PersonaLibrary()
runner = BetaTestRunner(personas=library.all())
results = runner.test_repo("/path/to/repo")
for r in results:
    print(f"{r.persona.name}: {'PASS' if r.success else 'FAIL'} — {r.friction}")

# --- Conservation Spectral Bridge ---
from fleet.conservation_spectral_bridge import SpectralBreederDiversity

sbd = SpectralBreederDiversity()
sbd.register_agent(
    "vision_specialist",
    capabilities=["vision", "detection"],
    capability_links=[("vision", "detection", 0.9)],
)
parents = sbd.select_parents(n=2, min_diversity=0.3)
```

---

### triage/ — Drift & Duplicate Detection

> *Keep the repo healthy. Detect structural drift and duplicate issues automatically.*

The triage module implements SPEC-REPO-METRIC for automated repository health monitoring.

**Files:** 7 source files

| File | Purpose |
|------|---------|
| `drift_detect.py` | Detect structural drift: stale deps, test regression, doc drift, dead code |
| `duplicate_detect.py` | TF-IDF + cosine similarity for duplicate issue detection |
| `github_issues.py` | GitHub issue fetching and analysis |
| `weekly.py` | Weekly triage report generation |
| `metrics.py` | Repository health metrics |
| `repo_duplicate.py` | Cross-repo duplicate detection |

```python
from triage.drift_detect import DriftDetector, detect_drift, DriftReport
from triage.duplicate_detect import DuplicateDetector, find_duplicates

# --- Drift Detection ---
report: DriftReport = detect_drift("/path/to/repo")
print(f"Stale deps: {report.stale_dependencies}")
print(f"Test regression: {report.test_regression}")
print(f"Doc drift: {report.documentation_drift}")
print(f"Dead code: {report.dead_code_files}")

# --- Duplicate Issue Detection ---
detector = DuplicateDetector()
pairs = find_duplicates(
    issues=[
        {"number": 1, "title": "Agent fails to breed", "body": "..."},
        {"number": 2, "title": "Breeding broken for agents", "body": "..."},
    ]
)
for pair in pairs:
    print(f"#{pair.issue_a} ≈ #{pair.issue_b} (similarity: {pair.similarity:.2f})")
```

---

### perception/ — Vision & Audio Encoding

> *Encode the world into tiles the fleet can reason about.*

**Files:** 6 source files

| File | Purpose |
|------|---------|
| `vision_encoder.py` | Visual tile encoding from images → feature vectors |
| `audio_encoder.py` | Audio tile encoding from waveforms → feature vectors |

```python
from perception.vision_encoder import VisionEncoder
from perception.audio_encoder import AudioEncoder

# Encode an image
vision = VisionEncoder(dim=256)
tile = vision.encode(image_path="photo.jpg")
print(f"Vision tile shape: {tile.shape}")  # (256,)

# Encode audio
audio = AudioEncoder(dim=256)
tile = audio.encode(audio_path="recording.wav")
print(f"Audio tile shape: {tile.shape}")  # (256,)
```

---

### compiler/ — Hot-Swap Compilation

> *Runtime-adaptive compilation. Profile → Analyze → Compile → Validate → Deploy. No downtime.*

**Files:** 2 source files

| File | Purpose |
|------|---------|
| `hot_swap_integration.py` | A/B compile, auto-rollback, gating |

```python
from compiler.hot_swap_integration import hot_swap, hot_swap_restore


# Hot-swap a function with an optimized version
@hot_swap(variant="optimized")
def compute_gradient(x):
    return 2 * x  # Original


# If the optimized version fails, auto-rollback to original
result = compute_gradient(5.0)

# Manually restore original
hot_swap_restore(compute_gradient)
```

---

### a2a/ — Agent-to-Agent Protocol

> *Google A2A protocol for inter-agent communication.*

**Files:** 5 source files

| File | Purpose |
|------|---------|
| `server.py` | A2A server with health endpoint |
| `identity.py` | Agent cards, registry, identity, task handles |
| `protocol.py` | A2A protocol adapter, client, task status |
| `signal_bridge.py` | Signal bridge for A2A events |
| `plugin_manager.py` | Plugin management for A2A extensions |

```python
from a2a.server import A2AServer
from a2a.identity import AgentCard, AgentRegistry

# Start an A2A server
server = A2AServer(host="0.0.0.0", port=8080)
server.start()

# Register an agent
registry = AgentRegistry()
card = AgentCard(name="breeder-1", capabilities=["breeding", "selection"])
registry.register(card)

# Discover agents
agents = registry.discover(capability="breeding")
```

---

### reasoning/ — Polyglot Reasoning

> *One fleet, multiple languages. Reason across Python, Rust, C++, Mercury, C.*

**Files:** 6 source files

| File | Purpose |
|------|---------|
| `python_bridge.py` | Python reasoning backend (always available) |

The reasoning module dispatches tile similarity queries to the fastest available backend. See `examples/polyglot_reason.py` for a working demo.

```python
from reasoning.python_bridge import PythonReasoner

reasoner = PythonReasoner()
similarity = reasoner.tile_similarity(tile_a, tile_b)
print(f"Similarity: {similarity:.4f}")
```

---

### ranking/ — User Ranking & Personalization

> *Distill sensor and personalization engine.*

**Files:** 5 source files

| File | Purpose |
|------|---------|
| `user_ranking.py` | User ranking system |
| `ranked_response.py` | Ranked response selection |
| `personalization.py` | Preference scoring and personalization store |

```python
from ranking import UserRanking, RankedResponse, PersonalizationStore

ranking = UserRanking()
store = PersonalizationStore()

# Rank responses
responses = [
    RankedResponse(text="Option A", score=0.8),
    RankedResponse(text="Option B", score=0.6),
]
ranked = ranking.rank(responses)
print(f"Best: {ranked[0].text}")  # "Option A"

# Store preferences
store.record_preference(user="user-1", item="breeding_strategy", score=0.9)
```

---

### distill/ — Distillation Signals

> *Training and signal processing for knowledge distillation.*

**Files:** 6 source files

| File | Purpose |
|------|---------|
| `distillation_signal.py` | Distillation signal processing |
| `delta_tracker.py` | Track deltas between model versions |

```python
from distill import DistillationSignal, DeltaTracker

signal = DistillationSignal(source="teacher", target="student")
signal.emit(metric="accuracy", value=0.92)

tracker = DeltaTracker()
tracker.record(version="v1", metrics={"accuracy": 0.90})
tracker.record(version="v2", metrics={"accuracy": 0.92})
delta = tracker.delta("v1", "v2")
print(f"Improvement: {delta['accuracy']:.2f}")  # 0.02
```

---

### flux_vm/ — FLUX Constraint VM

> *Virtual machine for executing FLUX constraint programs.*

**Files:** 2 source files

| File | Purpose |
|------|---------|
| `flux_compat.py` | FLUX constraint VM (Path A + Path B integration) |

```python
from flux_vm.flux_compat import FluxVM

vm = FluxVM()
vm.load_program(program_bytes)
result = vm.execute()
```

---

### flux_compat/ — FLUX Compatibility Layer

> *Compatibility shims for FLUX OS integration.*

**Files:** 8 source files

Bridges between FLUX OS microkernel abstractions and the Python ecosystem. See `examples/flux_os_deploy.py` for a working demo.

---

### jepa/ — JEPA Predictive World Model

> *Joint-Embedding Predictive Architecture for agent world modeling.*

**Files:** 2 source files

```python
from jepa.jepa_room import JEPARoom

room = JEPARoom(dim=256)
prediction = room.predict(current_state)
```

---

### grammar/ — Grammar Engine

> *Grammar specification and security audit for the fleet's DSL.*

**Files:** 4 source files

See `docs/GRAMMAR-ENGINE-SPEC.md` and `docs/GRAMMAR-SECURITY-AUDIT-LOCAL.md` for details.

---

## SuperInstance Ecosystem Integration

sunset-ecosystem is the flagship of a ~598-repo fleet across two organizations (SuperInstance: 209 repos, Lucineer: 389 repos). Here's how it connects to everything:

### Tier 1: Direct Integration

| Repo | Language | What | How It Connects |
|------|----------|------|-----------------|
| **constraint-theory-core** | Rust | Exact Pythagorean snapping, SIMD, KD-tree, holonomy verification | `swarm/constraint_bridge.py` — breeding loops use ~100ns exact snap instead of FP |
| **flux-os** | C | Self-compiling microkernel, agent-native OS | `sunset/flux_vm_bridge.py` — deploy breeding agents as FLUX OS services |
| **plato-agent-academy** | Python | Agent training with 6 test cohorts | `sunset/plato_bridge.py` — agents graduate from academy into fleet |
| **cocapn-plato** | Python | PLATO room topology and spell system | `sunset/roomgrid_plato_observer.py` — observe room grid state |
| **cocapn-health** | Python | Health check framework with 18 services | `fleet/health_bridge.py` — standardized CheckResult pattern |
| **git-agent-standard** | Python | Git-native agent identity standard | `fleet/agent_identity_bridge.py` — repo-as-identity, bottles, diary |
| **fleet-consciousness-dashboard** | Python | Fleet consciousness scoring (FCI) | `fleet/fleet_consciousness_bridge.py` — weighted health score 0.0–1.0 |
| **Equipment-Swarm-Coordinator** | Python | Swarm coordination patterns | `fleet/swarm_coordinator_bridge.py` — roles, conflict resolution |
| **oracle1-vessel** | Python | Fleet heartbeat, task board, fence board | `fleet/heartbeat_bridge.py`, `fleet/fleet_task_board_bridge.py`, `fleet/fence_board_bridge.py` |

### Tier 2: Conservation & Spectral

| Repo | What | How It Connects |
|------|------|-----------------|
| **conservation-law-rs** | γ + η = total_budget (entropy conservation) | Entropy budgets flow into `sunset_trinity` scoring |
| **spectral-fleet-rs** | Laplacian eigenvalue decomposition | Ranking influences `swarm` breeding priorities via `fleet/conservation_spectral_bridge.py` |
| **SuperInstance** (monorepo) | Conservation Spectral Framework | Spectral fingerprints → agent identity, Fiedler vector → routing |

### Tier 3: Runtime & CLI

| Tool | What | How It Connects |
|------|------|-----------------|
| **si-cli** (`si`) | Fleet command-line interface | `si scan sunset-ecosystem` discovers all capabilities |
| **si-fleet-api** | REST endpoints for fleet data | HTTP API wraps fleet conductor and event bus |
| **si-core-c** | Conservation API in C | Same `γ + η` budget model compiled for C consumers |
| **si-runtime-*** | Conservation API in 7 languages | Polyglot FFI: Python, Rust, C++, Mercury, C, JS, Go |

### Supabase Fleet Registry

The fleet uses Supabase for persistent state:

```sql
-- Fleet registry tables
SELECT * FROM repos WHERE org = 'SuperInstance';        -- 209 repos
SELECT * FROM capabilities WHERE repo = 'sunset-ecosystem';  -- All exposed capabilities
SELECT * FROM budgets WHERE type = 'thermal';            -- γ + η budgets per repo
```

### Integration Pattern

Every bridge in `fleet/` follows the same pattern:

1. **Zero-dependency** — stdlib only, no external imports required
2. **Pattern extraction** — distill the external repo's core idea into pure Python
3. **Bridge interface** — provide Python APIs that match the original's semantics
4. **Event bus** — emit events via `FleetEventBus` for cross-component notification
5. **Test parity** — bridge tests verify behavior matches the upstream repo

```
External Repo → fleet/*_bridge.py → FleetEventBus → FleetConductorV2
                   ↓
              sunset-ecosystem
              (native module)
```

---

## Conservation Law

The fleet operates under a **conservation law** borrowed from thermodynamics:

```
γ + η = total_budget

where:
  γ (gamma) = utilized entropy (compute actually consumed)
  η (eta)   = available entropy (headroom remaining)
```

### How the Thermal Auction Implements It

The VCG combinatorial auction in `swarm/thermal_auction.py` is the enforcement mechanism:

1. **Each agent submits a bid** — `fitness × expected_value` for a device slot
2. **Top S bidders win** (where S = number of identical slots)
3. **Each winner pays the (S+1)th highest bid** — this is the externality they impose on losers
4. **VCG guarantees truthful bidding is the dominant strategy** — agents cannot game the auction

The thermal budget constrains γ: if total utilization approaches `total_budget`, the auction raises prices, throttling aggressive agents and protecting the fleet from thermal runaway.

```python
from swarm.thermal import ThermalBudget, DeviceType
from swarm.thermal_auction import VCGAuction, Bid

# Define the thermal budget (conservation law)
budget = ThermalBudget(
    total_watts=100.0,
    device_type=DeviceType.GPU,
    slots=4,
)

# Each bid claims some of γ
bids = [
    Bid(agent_id="agent-1", value=0.9, watts=25.0),
    Bid(agent_id="agent-2", value=0.8, watts=30.0),
    Bid(agent_id="agent-3", value=0.7, watts=20.0),
    Bid(agent_id="agent-4", value=0.6, watts=15.0),
    Bid(agent_id="agent-5", value=0.5, watts=25.0),  # Will lose
]

auction = VCGAuction(device_type="GPU", slots=4)
allocation = auction.resolve(bids)

# Verify conservation: sum of allocated watts ≤ total_budget
allocated = sum(b.watts for b in allocation.winners)
assert allocated <= budget.total_watts, "Conservation law violated!"
print(f"Allocated: {allocated:.1f}W / {budget.total_watts:.1f}W")
print(f"Headroom (η): {budget.total_watts - allocated:.1f}W")
```

### Conservation Across the Fleet

The same `γ + η = total_budget` invariant is enforced at every level:

| Level | γ (used) | η (headroom) | total_budget |
|-------|----------|---------------|--------------|
| **Agent** | tokens consumed | tokens remaining | max_tokens |
| **Device** | watts consumed | thermal headroom | TDP |
| **Node** | compute allocated | compute available | node capacity |
| **Fleet** | active agents | available slots | fleet capacity |

The ethos module (`ethos/thermal_auto_calibrate.py`) auto-calibrates these budgets from stress test data, ensuring the conservation law matches actual hardware capability.

---

## Examples

All examples are copy-paste-runnable after `pip install -e ".[dev]"`.

### Spawning and Evolving Agents

```python
"""examples/spawn_and_evolve.py — Full agent lifecycle demo."""

import numpy as np
from sunset.agent import SunsetAgent, AgentPhase, ResourceBudget
from swarm.breeder_daemon_v2 import BreederDaemonV2, LifecycleState
from swarm.tournament import AgentScore, dominated_by, sunset_candidates

# Create a breeding daemon
daemon = BreederDaemonV2(dim=256, max_agents=20)

# Spawn initial population
agents = []
for i in range(10):
    aid = daemon.spawn(initial_vector=np.random.randn(256).astype(np.float32))
    daemon.transition(aid, LifecycleState.COMPETE)
    agents.append(aid)
    print(f"Spawned {aid} → COMPETE")

# Run one breeding step
daemon.step()  # Dequeues one request, checks thermal budget, spawns or waits

# Score agents with trinity
scores = [
    AgentScore(
        agent_id=aid,
        ethos=np.random.random(),
        pathos=np.random.random(),
        logos=np.random.random(),
    )
    for aid in agents
]

# Find sunset candidates (Pareto-dominated)
to_sunset = sunset_candidates(scores)
for s in to_sunset:
    print(
        f"Sunsetting {s.agent_id} (ethos={s.ethos:.2f}, pathos={s.pathos:.2f}, logos={s.logos:.2f})"
    )
    daemon.transition(s.agent_id, LifecycleState.SUNSET)
```

### VCG Thermal Auction

```python
"""examples/thermal_auction.py — Truthful slot allocation demo."""

from swarm.thermal_auction import VCGAuction, Bid

auction = VCGAuction(device_type="GPU", slots=3)

bids = [
    Bid(agent_id="alpha", value=0.95),
    Bid(agent_id="beta", value=0.88),
    Bid(agent_id="gamma", value=0.72),
    Bid(agent_id="delta", value=0.65),
    Bid(agent_id="epsilon", value=0.51),
]

allocation = auction.resolve(bids)
print(f"Winners: {[b.agent_id for b in allocation.winners]}")
print(
    f"Prices:  {dict(zip([b.agent_id for b in allocation.winners], allocation.prices))}"
)
# delta loses — its value (0.65) is below the 3rd slot
# Each winner pays the externality: what the loser would have gained
```

### Trinity Scoring

```python
"""examples/trinity_scoring.py — Compute full trinity score for an agent."""

from ethos.trinity_connection import score_ethos_connection
from pathos.trinity_connection import score_pathos_connection, NeedState, MomentScore
from ethos.hardware_survey import HardwareProfile

# Ethos: hardware efficiency
ethos = score_ethos_connection(
    hardware_efficiency=0.85,
    latency_fit=0.92,
    thermal_fit=0.78,
    memory_fit=0.90,
)

# Pathos: human value
needs = NeedState(
    problems_identified=["deploy faster", "reduce errors"],
    problems_solved=["deploy faster"],
    cognitive_load_before=0.8,
    cognitive_load_after=0.3,
)
moment = MomentScore(
    directly_useful=True, reduced_friction=True, surprise_insight=False
)
pathos = score_pathos_connection(needs, moment)

# Logos: audit trail completeness (simplified)
logos_score = 0.88  # From decision journal completeness

# Composite trinity score
trinity = ethos.total * pathos.composite * logos_score
print(f"Ethos:  {ethos.total:.3f}")
print(f"Pathos: {pathos.composite:.3f}")
print(f"Logos:  {logos_score:.3f}")
print(f"Trinity (product): {trinity:.3f}")
# If any dimension is 0 → trinity = 0 → agent sunsets
```

### Spectral Breeding

```python
"""examples/spectral_breed.py — Evolve genomes in the Fourier domain."""

import numpy as np
from swarm.spectral_breeding import SpectralBreeder

breeder = SpectralBreeder(population_size=50, spectrum_size=64)
breeder.initialize()


def sphere_fitness(phenotype):
    """Minimize sum of squares."""
    return -np.sum(phenotype**2)


for generation in range(200):
    breeder.step(task_fn=sphere_fitness)
    if generation % 50 == 0:
        print(f"Gen {generation}: best fitness = {breeder.best_fitness():.4f}")

print(f"Final best fitness: {breeder.best_fitness():.4f}")
```

### Pythagorean Evolution (Exact Arithmetic)

```python
"""examples/pythagorean_evo.py — No floating-point drift."""

from swarm.pythagorean_evolution import PythagoreanBreeder

breeder = PythagoreanBreeder(population_size=50, genome_length=10)
breeder.initialize()


def fitness(genome):
    """Maximize the hypotenuse sum."""
    return sum(t.c for t in genome.triples)


for generation in range(100):
    breeder.step(fitness_fn=fitness)
    if generation % 25 == 0:
        best = breeder.best()
        print(f"Gen {generation}: best = {fitness(best)}")
        for t in best.triples[:3]:
            print(f"  {t.a}² + {t.b}² = {t.c}² (exact)")
```

### Querying Fleet Status

```python
"""examples/fleet_status.py — Get a snapshot of the entire fleet."""

from nexus.fleet_conductor_v2 import FleetConductorV2
from fleet.work_dashboard import FleetWorkDashboard
from fleet.health_bridge import HealthChecker, FLEET_SERVICES

# Fleet conductor
conductor = FleetConductorV2(node_id="node-1")
conductor.initialize()
conductor.beat_tick()
status = conductor.status()
print(f"Agents: {status['agent_count']}")
print(f"Breeding: {status['breeding_active']}")
print(f"Thermal: {status['thermal_utilization']:.1f}%")

# Work dashboard
dashboard = FleetWorkDashboard()
report = dashboard.report()
print(f"\n=== Fleet Work Dashboard ===")
print(report)

# Health check
checker = HealthChecker(FLEET_SERVICES)
results = checker.check_all()
print(f"\n=== Health Check ===")
print(checker.report(results, format="md"))
```

### Drift Detection

```python
"""examples/drift_detect.py — Check repo for structural drift."""

from triage.drift_detect import detect_drift

report = detect_drift(".")
if report.stale_dependencies:
    print("⚠️  Stale dependencies:")
    for dep in report.stale_dependencies:
        print(f"  - {dep}")
if report.test_regression:
    print("⚠️  Test regression detected!")
if report.documentation_drift:
    print("⚠️  Documentation drift:")
    for item in report.documentation_drift:
        print(f"  - {item}")
if report.dead_code_files:
    print(f"⚠️  {len(report.dead_code_files)} dead code files")
if report.is_healthy:
    print("✅ Repository is healthy")
```

---

## API Reference

### Core Classes

#### `sunset.agent.SunsetAgent`

```python
@dataclass
class SunsetAgent:
    agent_id: str
    phase: AgentPhase  # INCUBATING | COMPETING | BREEDING | SUNSETTING | ASLEEP
    budget: ResourceBudget
    # ... lifecycle management


class AgentPhase(Enum):
    INCUBATING = "incubating"
    COMPETING = "competing"
    BREEDING = "breeding"
    SUNSETTING = "sunsetting"
    ASLEEP = "asleep"
```

#### `swarm.breeder_daemon_v2.BreederDaemonV2`

```python
class BreederDaemonV2:
    def __init__(self, dim: int, max_agents: int): ...
    def spawn(self, initial_vector: np.ndarray) -> str: ...
    def transition(self, agent_id: str, state: LifecycleState) -> None: ...
    def state_of(self, agent_id: str) -> LifecycleState: ...
    def breed(self, parent_a: str, parent_b: str) -> str: ...
    def step(self) -> None: ...  # Process one breeding request
```

#### `swarm.thermal_auction.VCGAuction`

```python
class VCGAuction:
    def __init__(self, device_type: str, slots: int): ...
    def resolve(self, bids: List[Bid]) -> Allocation: ...


@dataclass(frozen=True)
class Bid:
    agent_id: str
    value: float
    # Optional: watts, priority


@dataclass
class Allocation:
    winners: List[Bid]
    prices: List[float]  # VCG prices (externality-based)
```

#### `swarm.fleet_bft_qd.PBFTNode`

```python
class PBFTNode:
    def __init__(self, node_id: str, peers: List[str]): ...
    def propose(self, proposal_id: str, payload: dict) -> None: ...
    def vote(self, proposal_id: str, accept: bool) -> None: ...
    def commit_if_quorum(self) -> ConsensusResult: ...
```

#### `swarm.tournament`

```python
@dataclass(frozen=True)
class AgentScore:
    agent_id: str
    ethos: float  # 0.0-1.0
    pathos: float  # 0.0-1.0
    logos: float  # 0.0-1.0


def dominated_by(a: AgentScore, b: AgentScore) -> bool:
    """True if a dominates b on all trinity axes."""


def sunset_candidates(scores: List[AgentScore]) -> List[AgentScore]:
    """Return Pareto-dominated agents eligible for sunsetting."""


def breed(parents: List[AgentScore]) -> dict:
    """Select parents and produce crossover parameters."""
```

#### `nerve.room_grid.RoomGrid`

```python
class RoomGrid:
    def __init__(self, n_rooms: int, dim: int): ...
    @property
    def backend(self) -> str: ...  # "cuda" | "rust_persistent" | "numpy"
    def tick(self, signal: np.ndarray, weights: np.ndarray) -> np.ndarray: ...
    def tick_batch(self, signals: np.ndarray, weights: np.ndarray) -> np.ndarray: ...
```

#### `logos.decision_journal.DecisionJournal`

```python
class DecisionJournal:
    def __init__(self, path: str): ...
    def log(self, decision: Decision) -> None: ...
    def query(self, agent_id: str = None, action: str = None) -> List[Decision]: ...
```

#### `nexus.fleet_conductor_v2.FleetConductorV2`

```python
class FleetConductorV2:
    def __init__(self, node_id: str): ...
    def initialize(self) -> None: ...  # Lazy-load all subsystems
    def beat_tick(self) -> None: ...  # One heartbeat
    def status(self) -> dict: ...  # Fleet snapshot
```

#### `ethos.trinity_connection.score_ethos_connection`

```python
def score_ethos_connection(
    hardware_efficiency: float,
    latency_fit: float,
    thermal_fit: float,
    memory_fit: float,
) -> EthosConnectionScore: ...
```

#### `pathos.trinity_connection.score_pathos_connection`

```python
def score_pathos_connection(
    needs: NeedState,
    moment: MomentScore,
) -> TrinityScore: ...
```

#### `fleet.sense_decide_act.SDAPipeline`

```python
class SDAPipeline:
    def __init__(self, name: str): ...
    def sense(self, data: dict) -> Observation: ...
    def decide(self, obs: Observation) -> Decision: ...
    def act(self, decision: Decision) -> ActResult: ...
```

---

## Testing

```
$ pytest --co -q | tail -1
8729 tests collected

$ pytest --tb=short -q
8729 passed in 87.43s
```

### Running Tests

```bash
# Full suite
pytest

# Specific module
pytest tests/test_breeder_daemon_v2.py -v
pytest tests/test_fleet_bft_qd.py -v
pytest tests/test_thermal_auction.py -v
pytest tests/test_tournament.py -v
pytest tests/test_room_grid.py -v
pytest tests/test_health_bridge.py -v

# With coverage
pytest --cov=swarm --cov=nerve --cov=sunset --cov=nexus --cov-report=term-missing

# Only fast tests (skip slow benchmarks)
pytest -k "not benchmark"
```

### Test Organization

| Pattern | What |
|---------|------|
| `tests/test_*.py` | Unit and integration tests (471 files) |
| `tests/benchmarks/` | Performance benchmarks |
| `tests/conftest.py` | Shared fixtures |

### Pre-Commit Checks

```bash
ruff check .
mypy .
pytest
```

---

## Contributing

Read [DEVELOPER.md](./DEVELOPER.md) first. The trinity is strict: every change must serve `ethos × pathos × logos`.

### Adding a New Module

1. Create your module directory: `mymodule/`
2. Add it to `pyproject.toml` → `[tool.setuptools.packages.find].include`
3. Implement with `__all__` exports and type hints
4. Add tests in `tests/test_mymodule.py`
5. Run: `ruff check . && mypy . && pytest`

### Adding a Fleet Bridge

1. Create `fleet/your_pattern_bridge.py`
2. Follow the bridge pattern: zero-dependency, event bus integration, test parity
3. Add tests that verify behavior matches upstream
4. Register in the event bus if needed

### The Sunset Protocol

When an agent sunsets:

1. **Log the decision** → `logos/decision_journal.py`
2. **Archive the agent state** → SQLite WAL
3. **Decompose into seeds** → The agent's best traits inform the next generation
4. **Free resources** → Thermal budget returned to the pool
5. **Notify the fleet** → FleetEventBus emits `sunset.complete`

```python
from logos.decision_journal import log_sunset

log_sunset(
    agent_id="agent-042",
    reason="Zero pathos score for 5 consecutive ticks",
    metadata={
        "final_ethos": 0.85,
        "final_pathos": 0.0,
        "final_logos": 0.72,
        "generation": 12,
        "children": ["agent-043", "agent-044"],
        "seed_traits": ["high_ethos", "moderate_logos"],
    },
)
```

---

## Documentation

| Doc | What You'll Find |
|-----|-----------------|
| [DEVELOPER.md](./DEVELOPER.md) | Architecture deep-dive, how to add rooms/spells/devices |
| [docs/ECOSYSTEM_INTEGRATION.md](./docs/ECOSYSTEM_INTEGRATION.md) | How sunset-ecosystem connects to the 598-repo SuperInstance ecosystem |
| [docs/FLEET_BFT_QD.md](./docs/FLEET_BFT_QD.md) | Byzantine consensus + quality diversity algorithms |
| [docs/SENSE_DECIDE_ACT.md](./docs/SENSE_DECIDE_ACT.md) | The unifying framework for all fleet behaviors |
| [docs/FLUX_OPCODE_ALIGNMENT.md](./docs/FLUX_OPCODE_ALIGNMENT.md) | FLUX VM integration strategy (Path A vs Path B) |
| [docs/CONSERVATION_SPECTRAL_BRIDGE.md](./docs/CONSERVATION_SPECTRAL_BRIDGE.md) | Laplacian eigenvalue decomposition for breeding diversity |
| [docs/MATHEMATICAL_STRUCTURES.md](./docs/MATHEMATICAL_STRUCTURES.md) | Mathematical foundations (holonomy, sheaf cohomology, Ricci flow) |
| [docs/ARCHITECTURE-v2.md](./docs/ARCHITECTURE-v2.md) | Full architecture diagram and subsystem map |
| [docs/A2A-FIRST-ARCHITECTURE.md](./docs/A2A-FIRST-ARCHITECTURE.md) | Agent-to-Agent protocol architecture |
| [CHANGELOG.md](./CHANGELOG.md) | Release history |

---

## License

MIT — see [LICENSE](./LICENSE) for details.

> *"Agents sunset with dignity and seed the next generation."*
