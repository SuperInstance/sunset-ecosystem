# sunset-ecosystem 🌅

> *A fleet of agents that breed, vote, sunset with dignity, and seed the next generation — governed by a trinity of ethos (metal), pathos (human), and logos (code).*

[![Tests](https://img.shields.io/badge/tests-2065%2B%20passing-brightgreen)](./tests)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](./pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-yellow)](./LICENSE)
[![Status](https://img.shields.io/badge/status-Beta-orange)]()

---

## Table of Contents

- [What is this?](#what-is-this)
- [Trinity Architecture](#trinity-architecture)
  - [Ethos — The Metal Surveyor](#ethos--the-metal-surveyor)
  - [Pathos — The Human Interface](#pathos--the-human-interface)
  - [Logos — The Code Memory](#logos--the-code-memory)
- [Major Modules](#major-modules)
  - [nerve/ — Forward Inference & Room Grid](#nerve--forward-inference--room-grid)
  - [swarm/ — Breeding, Tournament & Diversity](#swarm--breeding-tournament--diversity)
  - [sunset/ — Agent Lifecycle & Trinity Scoring](#sunset--agent-lifecycle--trinity-scoring)
  - [logos/ — Decision Journals & WAL](#logos--decision-journals--wal)
  - [triage/ — Health Metrics & Drift Detection](#triage--health-metrics--drift-detection)
  - [voice/ — Speech Synthesis](#voice--speech-synthesis)
  - [compiler/ — Hot-Swap Integration](#compiler--hot-swap-integration)
  - [perception/ — Vision, Audio & Cognition](#perception--vision-audio--cognition)
  - [ethos/ — Hardware Survey & Thermal](#ethos--hardware-survey--thermal)
  - [pathos/ — Need Tracking & Moment Scoring](#pathos--need-tracking--moment-scoring)
  - [a2a/ — Agent-to-Agent Protocol](#a2a--agent-to-agent-protocol)
  - [fleet/ — Sense→Decide→Act & Conservation Spectral](#fleet--sensedecideact--conservation-spectral)
  - [superinstance/ — Event Bus Runtime](#superinstance--event-bus-runtime)
- [Working Code Examples](#working-code-examples)
- [SuperInstance Runtime Integration](#superinstance-runtime-integration)
- [Fleet Conservation Budgets](#fleet-conservation-budgets)
- [Supabase Fleet Registry](#supabase-fleet-registry)
- [MCP / A2A Protocol Support](#mcp--a2a-protocol-support)
- [Test Suite](#test-suite)
- [Installation](#installation)
- [Development](#development)
- [License](#license)

---

## What is this?

**sunset-ecosystem** is an evolutionary, self-governing agent fleet with 1400+ files and 2065+ tests. Agents are born, compete for survival, breed children with diverse traits, and — when their time comes — sunset gracefully, archiving their lineage for future generations.

Every agent carries a **trinity score**: `ethos × pathos × logos`. Drop to zero in any dimension, and the fleet sunsets you. Survive, and you might be selected as a parent for the next breeding cycle.

**The fleet decides by consensus.** Every breeding batch is ratified by Practical Byzantine Fault Tolerant (PBFT) voting across nodes, combined with Quality-Diversity (QD) evolutionary algorithms so the fleet doesn't just optimize — it *explores*.

This is not a chatbot wrapper. It is infrastructure for running hundreds of agents across heterogeneous hardware, with exact mathematical constraint satisfaction, polyglot reasoning (Python / Rust / C++ / Mercury / C), and a live bioluminescent dashboard showing thermal pressure, diversity metrics, and breeding progress in real time.

---

## Trinity Architecture

The entire ecosystem is organized around the **Trinity**: three independent selection pressures that every agent must satisfy simultaneously.

```
Trinity Score = Ethos × Pathos × Logos

If any dimension → 0, the agent sunsets.
```

### Ethos — The Metal Surveyor

**Ethos** measures how well an agent respects the hardware it runs on. It is the voice of the metal.

- **Hardware Survey** (`ethos/hardware_survey.py`): Probes CUDA, NPU, CPU, memory, and thermal capacity. Produces a `HardwareProfile` with per-device latency benchmarks.
- **Thermal Auto-Calibration** (`ethos/thermal_auto_calibrate.py`): Runs stress tests and auto-adjusts thermal budgets so agents don't cook the GPU.
- **Stress Testing** (`ethos/stress_test.py`): Pushes devices to their limit and reports `StressReport` with throttling thresholds.
- **Agent Allocation** (`ethos/agent_allocator.py`): Matches agent workloads to hardware profiles using a constraint solver.

An agent with high ethos uses compute efficiently, stays within thermal envelopes, and respects memory bounds. An agent that hogs the GPU or causes thermal throttling gets a low ethos score and faces sunset.

### Pathos — The Human Interface

**Pathos** measures whether an agent actually serves human needs. A pathos agent would rather be **invisible and effective** than **visible and impressive**.

- **Need Tracker** (`pathos/need_tracker.py`): Models human needs as a state machine (`NeedState`). Tracks which needs are met, pending, or blocked.
- **Moment Scorer** (`pathos/moment_scorer.py`): Scores interactions by how much they reduce cognitive load and increase human satisfaction.
- **Interaction Log** (`pathos/interaction_log.py`): Persistent log of every human↔agent interaction with sentiment scoring.
- **Trinity Connection** (`pathos/trinity_connection.py`): Computes the pathos dimension of the trinity score from human-facing metrics.

Pathos asks three questions about every agent:
1. Does it solve actual human problems?
2. Is its output directly useful (not just technically correct)?
3. Does it reduce or increase human cognitive load?

### Logos — The Code Memory

**Logos** measures how well an agent understands and integrates with the codebase. It is the memory of the system.

- **Codebase State** (`logos/codebase_state.py`): Surveys the repo for file counts, language breakdown, architecture patterns, and module structure.
- **Decision Journal** (`logos/decision_journal.py` / `decision_log.py`): Every significant decision is logged with context, reasoning, and outcome. Searchable via `DecisionLog.query()`.
- **Generation Memory** (`logos/generation_memory.py`): Tracks what each agent generation learned, so offspring inherit distilled knowledge.
- **Signed WAL** (`logos/signed_wal.py`): Append-only write-ahead log with cryptographic signatures for audit trails.
- **Trinity Connection** (`logos/trinity_connection.py`): Scores codebase understanding, integration quality, and maintainability.

Logos ensures that agents don't just write code — they write code that fits, that is documented, and that future agents (and humans) can understand.

---

## Major Modules

### nerve/ — Forward Inference & Room Grid

`nerve/` is the **perception-action backbone** of the ecosystem. It implements a biologically-inspired neural topology where signals flow from sensory fibers through a room grid, with Hebbian routing that strengthens successful pathways.

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `NerveTopology` | Orchestrates the full COLLECT → SELECT → COMPILE → FEEDBACK → REGULATE cycle |
| `RoomGrid` | 250-room vector grid where each room holds latent state (64-dim) |
| `NerveFiber` | Perception channel — one per sensory modality |
| `RoutingLayer` | Hebbian routing with chaos probability for exploration |
| `MetronomeScheduler` | Drives the grid on a periodic beat (BPM-driven) |
| `TickResult` | Snapshot of one topology tick |

**The COLLECT → SELECT → COMPILE Lifecycle:**

1. **COLLECT**: Each `NerveFiber` perceives raw signals and produces a `SensoryTile`
2. **SELECT**: `RoutingLayer.fire_fast()` selects which rooms receive which tiles (with chaos-based exploration)
3. **COMPILE**: `RoomGrid.tick()` processes combined signals and fires rooms that exceed novelty thresholds
4. **FEEDBACK**: Successful routes are reinforced; cold rooms are weakened
5. **REGULATE**: Chaos probability decays as routes compile, enabling convergence

**Adaptive Compiler Integration:**

`NerveTopology` can enable the agentic compiler (`sunset/compiler.py`) which profiles hot functions and auto-compiles them to Numba/Rust/CUDA after they've been called enough times. This is triggered every 50 ticks after a 100-tick warmup.

### swarm/ — Breeding, Tournament & Diversity

`swarm/` is the **evolutionary engine**. It implements tournament selection, Quality-Diversity archives, constraint theory integration, and thermal-aware breeding.

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `Breeder` | Tournament round → Pareto frontier → breed → rebirth cycle |
| `TournamentRound` | Competes agents and scores them on multiple objectives |
| `AgentLifecycle` | FSM: SPAWNED → ACTIVE → ADAPTING → COMPILED → SUNSET → (rebirth) |
| `ThermalBudget` | Tracks compute/thermal spend per agent |
| `ConstraintBridge` | Integrates exact constraint theory (Eisenstein integers, Laman rigidity) |
| `FleetBFTNetwork` | PBFT consensus for ratifying breeding batches |
| `QDArchive` | Quality-Diversity archive using MAP-Elites |
| `VectorSwarm` | Distributed vector operations and mesh gossip |

**Breeding Pipeline:**

```
Tournament Round → Score agents (trinity × diversity × thermal)
    ↓
Pareto Frontier → Select non-dominated agents
    ↓
Breed → Crossover + mutation on agent templates
    ↓
PBFT Consensus → Fleet nodes vote to ratify the batch
    ↓
Rebirth → Spawn children into cold rooms
    ↓
Sunset → Archive parents with epilogue + summary
```

**Constraint Theory Integration:**

`swarm/constraint_bridge.py` and `swarm/constraint_theory_integration.py` enforce exact mathematical constraints on agent configurations using:
- **Eisenstein integers** (`E12`, `HexDisk`): Discrete geometry for agent trait spaces
- **Laman rigidity** (`laman_is_rigid`): Ensures agent graphs are minimally rigid (no floppy modes)
- **Holonomy checks**: Cyclic consistency verification across agent state graphs

### sunset/ — Agent Lifecycle & Trinity Scoring

`sunset/` is the **core identity layer**. It defines what an agent is, how it lives, and how it dies.

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `Agent` | Dataclass with id, generation, phase, trinity_score, resource_budget |
| `AgentPhase` | INCUBATING → COMPETING → BREEDING → SUNSETTING → ASLEEP |
| `ResourceBudget` | max_tokens, max_time_seconds, parallel_slots |
| `trinity_score()` | Computes ethos × pathos × logos |
| `SeedBank` | Archives agent seeds for future resurrection |
| `TensorArchive` / `SunsetEntry` | Persistent storage of agent state tensors |
| `Epilogue` / `Summary` / `Onboarding` | Sunset documents: what the agent tried, what worked, what didn't |
| `PlatoBridge` | Bridges to the PLATO room system for cross-agent communication |
| `FluxVMBridge` | Bridges to the FLUX virtual machine for constraint execution |
| `Compiler` | Agentic compiler that auto-optimizes hot functions |

**Agent Lifecycle:**

```python
agent = Agent()                    # INCUBATING
agent.advance(AgentPhase.COMPETING)  # Finding relevance
agent.advance(AgentPhase.BREEDING)   # Scored high, spawning children
agent.advance(AgentPhase.SUNSETTING) # Writing epilogue
# ... archived in SeedBank, searchable forever
```

### logos/ — Decision Journals & WAL

`logos/` is the **audit and memory layer**. Every decision, every generation, every codebase survey is logged and searchable.

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `DecisionLog` / `DecisionRecord` | Structured decision records with context, reasoning, outcome |
| `GenerationMemory` / `GenerationHistory` | What each generation learned, distilled for offspring |
| `CodebaseState` | Survey of repo structure, languages, patterns |
| `SignedWAL` | Cryptographically signed append-only log |
| `mmap_wal.py` | Memory-mapped WAL for zero-copy reads |
| `wal_index.py` / `wal_query.py` | Indexed query interface for WAL contents |
| `TrinityConnection` | Logos dimension scoring for agents |

### triage/ — Health Metrics & Drift Detection

`triage/` is the **fleet health monitor**. It runs weekly automated health checks and detects when repos drift from their intended structure.

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `RepoHealthMetrics` / `HealthScore` | Five-component health score (tests, docs, deps, coverage, issues) |
| `DriftDetector` / `DriftReport` | Detects structural drift (missing tests, dead code, dependency changes) |
| `DuplicateDetector` | TF-IDF-based duplicate issue detection |
| `RepoDuplicateDetector` | Cross-repo duplicate detection via file hashing |
| `GitHubIssues` | REST API wrapper for issue management |
| `WeeklyTriage` | Orchestrates the full weekly health check |

### voice/ — Speech Synthesis

`voice/` provides audio output capabilities via the **Soniqo bridge** (`voice/soniqo_bridge.py`), enabling agents to speak their status reports and alerts.

### compiler/ — Hot-Swap Integration

`compiler/` provides runtime compilation and hot-swapping of agent code without restarting the fleet.

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `CompilerHotSwap` | Swaps compiled functions at runtime |
| `CompileResult` | Validation and speedup metrics for compiled code |
| `sunset/compiler.py` | Agentic compiler with Numba/Rust/CUDA backends |

### perception/ — Vision, Audio & Cognition

`perception/` is the **sensory input layer** for agents. It captures webcam frames, screen regions, microphone input, and system audio, then encodes them into 512-dim embeddings for the nerve grid.

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `VisionTileEncoder` | 512-dim image embedding (CLIP, transformers, torchvision backends) |
| `AudioTileEncoder` | 512-dim audio embedding (whisper, speechbrain backends) |
| `WebcamCapture` / `ScreenCapture` | Video frame capture with FPS throttling |
| `MicrophoneCapture` / `SystemAudioCapture` | Audio capture with frame dropping |
| `CognitionLoop` | observe → reason → act cycle for autonomous agents |
| `AgentConfig` | Configuration for autonomous agent behavior |

### ethos/ — Hardware Survey & Thermal

`ethos/` probes the metal and calibrates thermal budgets.

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `HardwareProfile` | CUDA/NPU/CPU/memory/thermal capacity survey |
| `survey_hardware()` | Produces a complete hardware profile |
| `ThermalAutoCalibrator` | Auto-adjusts thermal budgets from stress tests |
| `EthosConnectionScore` | Scores hardware efficiency, latency fit, thermal fit, memory fit |

### pathos/ — Need Tracking & Moment Scoring

`pathos/` ensures agents serve human needs.

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `NeedTracker` / `NeedState` | Human need state machine |
| `MomentScorer` / `MomentScore` | Interaction quality scoring |
| `InteractionLog` / `InteractionRecord` | Persistent human↔agent interaction log |
| `PathosTrinity` | Pathos dimension scoring |

### a2a/ — Agent-to-Agent Protocol

`a2a/` implements the **Google A2A (Agent-to-Agent) draft standard** for inter-agent communication.

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `A2AServer` | JSON-RPC 2.0 + SSE server for agent discovery and task handling |
| `A2AClient` | Discovers, sends tasks to, and subscribes to remote agents |
| `A2AAgentCard` | A2A-compliant agent card (RFC 8785 JCS serialization) |
| `A2ATask` | Task lifecycle: submitted → working → completed/failed |
| `AgentIdentity` / `AgentRegistry` | Identity management and discovery |
| `A2AProtocolAdapter` | Fleet-wide registration and wiring |
| `TaskHandle` / `TaskState` | Async task tracking |

**Task States:**
```
SUBMITTED → WORKING → INPUT_REQUIRED → COMPLETED
                      ↓
                   CANCELLED / FAILED
```

### fleet/ — Sense→Decide→Act & Conservation Spectral

`fleet/` is the **operational control layer**. Every fleet module implements a variation of the SDA loop.

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `Sense` / `Decide` / `Act` | Abstract base classes for the SDA loop |
| `SDALoop` / `SDAPipeline` | Pipeline orchestration with policy injection |
| `Observation` / `Decision` / `ActResult` | Structured data for each phase |
| `Holodeck` | Virtual environment for agent simulation |
| `PlatoSignalChain` | Bridges PLATO rooms to the breeding loop |
| `I2IBridge` | Iron-to-Iron bridge for cross-instance communication |
| `ConservationSpectralEngine` | Spectral fingerprinting and diversity scoring |
| `FleetBridgeServer` | HTTP REST API exposing fleet operations |

**Conservation Spectral Bridge:**

`fleet/conservation_spectral_bridge.py` connects the SuperInstance Conservation Spectral Framework to the breeding loop:
- **SpectralFingerprint**: Laplacian eigenvalue spectra of agent capability graphs
- **SpectralAlignmentScorer**: Cosine similarity of eigenvalue spectra (higher = more diverse)
- **ConservationRatioMonitor**: Detects anomalies in trinity alignment via spectral reconstruction error

### superinstance/ — Event Bus Runtime

`superinstance/` implements the **COLLECT → SELECT → COMPILE event bus** that all plugins use.

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `EventBus` | Three-phase pipeline: collect → select → compile |
| `CollectorPlugin` | Phase-1: gather raw artifacts |
| `SelectorPlugin` | Phase-2: filter and rank artifacts |
| `CompilerPlugin` | Phase-3: transform artifacts into outputs |
| `EventResult` | Immutable result with collected/selected/compiled lists |

---

## Working Code Examples

### Example 1: Create an Agent and Compute Its Trinity Score

```python
from sunset import Agent, AgentPhase, trinity_score
from sunset.agent import ResourceBudget

# Create an agent with a custom budget
agent = Agent(
    generation=3,
    room="breeding-chamber-7",
    resource_budget=ResourceBudget(
        max_tokens=8192,
        max_time_seconds=120.0,
        parallel_slots=4,
    ),
)

# Advance through lifecycle
agent.advance(AgentPhase.COMPETING)

# Compute trinity score (each dimension 0.0–1.0)
ethos = 0.85   # Hardware-efficient, thermally responsible
pathos = 0.72  # Solves human problems, reduces cognitive load
logos = 0.91   # Deep codebase understanding, clean integration

score = trinity_score(ethos, pathos, logos)
print(f"Trinity score: {score:.4f}")  # 0.85 × 0.72 × 0.91 = 0.5569

# Update agent score
agent.trinity_score = score
```

### Example 2: Run the Nerve Topology

```python
from nerve import NerveTopology

# Create topology with 8 fibers and 250 rooms
topo = NerveTopology(
    n_fibers=8,
    n_rooms=250,
    chaos=0.3,           # 30% exploration
    adapt_threshold=0.95,
    learning_rate=0.05,
    signal_dim=64,
)

# Enable the agentic compiler for auto-optimization
topo.enable_compiler(auto_compile_interval=50)

# Run 1000 ticks
results = topo.run(ticks=1000)

# Check stats
print(topo.stats)
# {
#   "tick": 1000,
#   "fibers": 8,
#   "rooms": 250,
#   "rooms_active": 47,
#   "rooms_cold": 203,
#   "fibers_compiled": 3,
#   "fibers_perceiving": 5,
#   "routes": 2000,
#   "channels": 249,
#   "chaos": 0.02,
# }

# Rebirth cold rooms to maintain diversity
topo.rebirth_cold_rooms()
```

### Example 3: Drive the Grid with a Metronome

```python
from nerve.metronome import MetronomeScheduler, RandomSignalSource
from nerve.room_grid import RoomGrid
from nerve.routing import RoutingLayer
from swarm.breeder_daemon import AutoBreeder

grid = RoomGrid(n_rooms=250, d=64)
router = RoutingLayer(chaos=0.3, learning_rate=0.05)
breeder = AutoBreeder(grid=grid)

scheduler = MetronomeScheduler(
    grid=grid,
    router=router,
    breeder=breeder,
    bpm=120.0,
    breeding_harmonic=4,    # Breed every 4 beats
    flux_harmonic=16,       # FLUX check every 16 beats
    signal_source=RandomSignalSource(seed=42),
)

# Start background thread
scheduler.start()

# Let it run for a bit
import time
time.sleep(10)

# Check actual BPM
print(f"Target BPM: {scheduler.bpm}")
print(f"Actual BPM: {scheduler.actual_bpm:.1f}")

# Stop gracefully
scheduler.stop()
```

### Example 4: Submit a Human Command with Intent Confirmation

```python
from nerve.metronome import MetronomeScheduler
from logos.intent_protocol import FleetState

# The scheduler can process human commands through intent confirmation
result = scheduler.submit_human_command(
    raw_command="breed 5 new agents from the top 10",
    fleet_state=FleetState(
        total_agents=250,
        active_agents=47,
        rooms=["room-0", "room-1", ...],
    ),
)

print(result["intent"])
print(f"Requires confirmation: {result['requires_confirmation']}")
if result["requires_confirmation"]:
    print(result["confirmation_prompt"])
else:
    print("Can execute immediately")
```

### Example 5: A2A Agent Card and Task Submission

```python
from a2a import A2AAgentCard, A2AClient, A2ATask

# Create an A2A-compliant agent card
card = A2AAgentCard(
    name="sunset-breeder",
    description="Evolutionary agent breeder with trinity scoring",
    version="0.1.0",
    url="http://localhost:8080",
    capabilities={
        "streaming": True,
        "pushNotifications": False,
    },
    skills=[
        {"name": "breed", "description": "Run one breeding cycle"},
        {"name": "score", "description": "Compute trinity score"},
    ],
)

# Discover and send a task to a remote agent
client = A2AClient(base_url="http://remote-agent:8080")
discovered = client.discover()
print(f"Discovered {len(discovered)} agents")

task = A2ATask(
    id="task-001",
    type="breed",
    input={"parents": 4, "offspring": 8},
)
result = client.send_task(task)
print(f"Task status: {result.status.value}")
```

### Example 6: Sense→Decide→Act Pipeline

```python
from fleet import Sense, Decide, Act, SDALoop, Policy, Observation

class ThermalSense(Sense):
    def sense(self, context):
        return Observation(
            timestamp=time.time(),
            source="thermal_probe",
            metrics={"gpu_temp": 82.3, "cpu_temp": 65.1},
            severity_hint="warning",
        )

class ThermalPolicy(Decide):
    def decide(self, observation, context):
        if observation.metrics["gpu_temp"] > 80:
            return Decision(
                action_type="throttle",
                confidence=0.95,
                payload={"max_utilization": 0.6},
                reasoning="GPU temp exceeds 80°C, throttling to 60%",
            )
        return Decision(action_type="noop", confidence=1.0, payload={})

class ThermalAct(Act):
    def act(self, decision, context):
        if decision.action_type == "throttle":
            print(f"Throttling to {decision.payload['max_utilization']*100:.0f}%")
            return ActResult(success=True, latency_ms=5.0, side_effects=["throttle_applied"])
        return ActResult(success=True, latency_ms=0.1, side_effects=[])

# Wire into a loop
loop = SDALoop(
    sense=ThermalSense(),
    decide=ThermalPolicy(),
    act=ThermalAct(),
    policy=Policy.default(),
)

result = loop.tick(context={"room_id": 42})
print(result)
```

### Example 7: Hardware Survey and Thermal Calibration

```python
from ethos import survey_hardware, ThermalAutoCalibrator

# Survey all hardware
profile = survey_hardware()
print(f"GPUs: {len(profile.gpus)}")
print(f"CUDA available: {profile.cuda_available}")
print(f"Total VRAM: {profile.total_vram_mb} MB")

# Run thermal auto-calibration
calibrator = ThermalAutoCalibrator(target_temp=75.0)
report = calibrator.calibrate(profile)
print(f"Thermal budget: {report.thermal_budget_watts} W")
print(f"Max sustained load: {report.max_sustained_load:.0%}")
```

### Example 8: Constraint Theory — Eisenstein Snap

```python
from swarm import E12, HexDisk, eisenstein_norm, snap_from_angle

# Create an Eisenstein integer (discrete 2D coordinate)
z = E12(a=3, b=4)
print(f"Norm N(3,4) = {eisenstein_norm(3, 4)}")  # 3² - 3·4 + 4² = 13

# Snap an angle to the nearest Eisenstein lattice point
 snapped = snap_from_angle(angle_rad=0.785, radius=5.0)
print(f"Snapped: ({snapped.a}, {snapped.b})")

# Check if a configuration graph is rigid (Laman's theorem)
from superinstance_ffi_real import laman_is_rigid
print(f"Rigid: {laman_is_rigid(6, 9)}")  # 6 vertices, 9 edges
```

### Example 9: Decision Journal and Codebase Survey

```python
from logos import DecisionLog, survey_codebase, CodebaseState

# Log a decision
log = DecisionLog()
log.record(
    context="Choosing breeding strategy for gen-7",
    options=["tournament", "roulette", "rank"],
    chosen="tournament",
    reasoning="Tournament maintains diversity while selecting for trinity",
    outcome="pending",
)

# Survey the codebase
state = survey_codebase(".")
print(f"Files: {state.file_count}")
print(f"Languages: {list(state.language_breakdown.keys())}")
print(f"Modules: {len(state.architecture_patterns.get('module_dirs', []))}")
```

### Example 10: Perception — Webcam to Nerve Grid

```python
from perception import WebcamCapture, VisionTileEncoder, CognitionLoop

# Capture from webcam
capture = WebcamCapture(device_id=0, fps=30)
frame = capture.read()

# Encode to 512-dim embedding
encoder = VisionTileEncoder(backend="clip")
embedding = encoder.encode(frame)
print(f"Embedding shape: {embedding.shape}")  # (512,)

# Run cognition loop
loop = CognitionLoop(
    encoder=encoder,
    capture=capture,
    grid=RoomGrid(n_rooms=250, d=512),
)
loop.observe()  # Perceive
loop.reason()   # Route through grid
loop.act()      # Fire rooms
```

---

## SuperInstance Runtime Integration

sunset-ecosystem is designed to run inside the **SuperInstance runtime stack**, which provides cross-language FFI, fleet orchestration, and a unified event bus.

### si-core-c — Low-Level Primitives

The `superinstance-ffi/` crate (Rust) exposes C-compatible functions:
- `eisenstein_norm(a, b)` — Eisenstein integer norm
- `laman_is_rigid(v, e)` — Graph rigidity check
- `holonomy_check(states, threshold)` — Cyclic consistency
- `pythagorean48_encode(num, den)` — Frequency ratio encoding
- `constraint_check(value, lower, upper)` — Interval constraint
- `spline_interpolate(...)` — Spline interpolation

Python bindings are provided via `superinstance_ffi_real.py` (ctypes) with a pure-Python fallback `superinstance_ffi_mock.py` that parses the C header and auto-generates mocks.

```python
# Use real Rust-backed FFI when available
from superinstance_ffi_real import eisenstein_norm, laman_is_rigid

# Or use the mock for testing / when Rust lib is not built
from superinstance_ffi_mock import load_mock_ffi
ffi = load_mock_ffi()
```

### si-runtime-python — Event Bus & Plugin System

`sunset-ecosystem` registers itself as plugins on the `EventBus`:

```python
from superinstance import EventBus
from swarm import ConstraintCollector, ConstraintSelector, ConstraintCompiler

bus = EventBus()
bus.register_collector(ConstraintCollector())
bus.register_selector(ConstraintSelector())
bus.register_compiler(ConstraintCompiler())

result = bus.run({"room_id": 42, "signal": embedding})
print(f"Collected: {len(result.collected)}")
print(f"Selected: {len(result.selected)}")
print(f"Compiled: {len(result.compiled)}")
```

### si-fleet-api — Fleet Orchestration

The `claw_fleet_bridge.py` module exposes an HTTP REST API for fleet operations:

```python
from claw_fleet_bridge import FleetBridgeServer

server = FleetBridgeServer(host="127.0.0.1", port=8850)
server.start()

# Endpoints:
# GET  /health              → {"status": "ok", "fleet": "cocapn"}
# GET  /status              → Fleet conductor status
# GET  /flux/presets        → Available FLUX constraint presets
# POST /breed               → Run one breeding cycle
# POST /flux/check          → Validate constraints
# POST /mesh/insert         → Insert vector into mesh table
# POST /mesh/query          → Query nearest neighbors
```

### si-cli — Command Line Interface

```bash
# Run fleet health check
python -m fleet.cli health

# Start the metronome scheduler
python -m fleet.cli metronome --bpm 120 --rooms 250

# Trigger a breeding cycle
python -m fleet.cli breed --parents 4 --offspring 8

# Run triage on all repos
python -m triage.weekly run --repos ./ --output triage-report.json
```

---

## Fleet Conservation Budgets

Every agent in the fleet operates under a **conservation budget** that enforces `γ + η = total`, where:
- **γ (gamma)** — Exploration budget: novel signals, chaos, mutation, diversity search
- **η (eta)** — Exploitation budget: compiled routes, known patterns, convergence

### Budget Allocation

```python
from sunset.agent import ResourceBudget

# Default budget
budget = ResourceBudget(
    max_tokens=4096,
    max_time_seconds=60.0,
    parallel_slots=1,
)

# High-exploration agent (more gamma)
explorer = ResourceBudget(
    max_tokens=8192,
    max_time_seconds=120.0,
    parallel_slots=4,
)

# High-exploitation agent (more eta, less gamma)
exploiter = ResourceBudget(
    max_tokens=2048,
    max_time_seconds=30.0,
    parallel_slots=1,
)
```

### Conservation Spectral Monitoring

The `ConservationSpectralEngine` monitors fleet-wide conservation:

```python
from fleet.conservation_spectral_bridge import (
    ConservationSpectralEngine,
    SpectralFingerprint,
    ConservationRatioMonitor,
)

# Fingerprint an agent's capability graph
fp = SpectralFingerprint.from_agent(
    agent_id="agent-001",
    capabilities=["vision", "audio", "breeding", "routing"],
)

# Monitor conservation ratio
monitor = ConservationRatioMonitor()
ratio = monitor.compute_ratio(agent_graph)
print(f"Conservation ratio: {ratio:.4f}")
# ratio ≈ 1.0 → well-balanced
# ratio ≈ 0.0 → anomaly detected (too much exploration or exploitation)
```

### Thermal Budget

`swarm/thermal.py` tracks thermal spend:

```python
from swarm.thermal import ThermalBudget

thermal = ThermalBudget(max_watts=250.0)
thermal.spend(45.0)  # Agent used 45W
print(f"Remaining: {thermal.remaining} W")
print(f"Utilization: {thermal.utilization:.1%}")
```

---

## Supabase Fleet Registry

sunset-ecosystem integrates with the **Supabase Fleet Registry** for persistent fleet state, audit logging, and cross-node synchronization.

### Tables

| Table | Purpose |
|-------|---------|
| `fleet_agents` | Agent records (id, generation, phase, trinity_score, budget) |
| `fleet_breeding_events` | Breeding cycle records with parent/child lineage |
| `fleet_budgets` | Per-agent conservation budgets (gamma, eta, total) |
| `fleet_events` | Audit trail of all fleet operations |
| `fleet_health` | Periodic health snapshots from triage |
| `fleet_mesh` | Vector mesh table for distributed similarity search |

### Connection

```python
import os

# Environment variables
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

# The fleet bridge auto-detects Supabase credentials
# and syncs agent state, budgets, and events
```

### Sync Behavior

- **Agent birth** → Upsert to `fleet_agents`
- **Budget transfer** → Update `fleet_budgets` (γ + η must equal total)
- **Breeding event** → Insert to `fleet_breeding_events` with PBFT consensus proof
- **Health check** → Upsert to `fleet_health`
- **Audit event** → Append to `fleet_events`

---

## MCP / A2A Protocol Support

sunset-ecosystem supports both **MCP (Model Context Protocol)** and **A2A (Agent-to-Agent)** standards for interoperability.

### A2A Protocol

Full implementation of the Google A2A draft spec:

- **Agent Cards**: JSON metadata describing capabilities, skills, endpoints
- **Task Lifecycle**: SUBMITTED → WORKING → INPUT_REQUIRED → COMPLETED
- **Streaming**: SSE-based progress updates for long-running tasks
- **Discovery**: Auto-discovery via `.well-known/agent-cards/` directory
- **Identity**: Cryptographic agent identity with `AgentRegistry`

**Server Mode:**

```python
from a2a import A2AServer

server = A2AServer(
    host="0.0.0.0",
    port=8080,
    agent_card=card,
    task_handlers={
        "breed": handle_breed_task,
        "score": handle_score_task,
    },
)
server.start()
```

**A2A Signal Source for Metronome:**

The metronome can fetch input signals from remote A2A agents:

```python
from nerve.metronome import A2ASignalSource

source = A2ASignalSource(
    endpoint_url="http://remote-agent:8080",
    agent_card_path=".well-known/agent-cards/remote.json",
)

scheduler = MetronomeScheduler(
    grid=grid,
    router=router,
    breeder=breeder,
    signal_source=source,  # ← Fetches signals from A2A agent
)
```

**Tick-as-Task Mode:**

When `task_mode=True`, every metronome beat is packaged as an A2A task and submitted to a remote endpoint:

```python
scheduler = MetronomeScheduler(
    ...,
    task_mode=True,
    a2a_endpoint="http://nexus.fleet.local:4047/metronome",
)
```

### MCP Integration

The ecosystem exposes MCP-compliant tool interfaces for external orchestrators:

- `tools/breed` — Run a breeding cycle
- `tools/score` — Compute trinity score
- `tools/survey` — Hardware survey
- `tools/triage` — Weekly health check
- `tools/audit` — Decision journal query

---

## Test Suite

With **2065+ tests** across 473 test files, sunset-ecosystem maintains comprehensive coverage.

### Running Tests

```bash
# Run all tests
pytest tests/ -x

# Run with coverage
pytest tests/ --cov=sunset --cov=swarm --cov=nerve --cov=logos

# Run specific module tests
pytest tests/test_sunset.py -v
pytest tests/test_topology.py -v
pytest tests/test_breeder.py -v
pytest tests/test_a2a_protocol.py -v
pytest tests/test_conservation_spectral_bridge.py -v

# Run benchmarks (excluded by default)
pytest tests/benchmarks/ -v

# Run with asyncio support
pytest tests/test_metronome.py -v --asyncio-mode=auto
```

### Test Categories

| Category | Files | Description |
|----------|-------|-------------|
| **Core** | `test_sunset.py`, `test_agent.py` | Agent lifecycle, trinity scoring |
| **Nerve** | `test_topology.py`, `test_room_grid.py`, `test_metronome.py`, `test_fiber.py` | Room grid, routing, metronome |
| **Swarm** | `test_breeder.py`, `test_tournament.py`, `test_thermal.py`, `test_chaos.py` | Breeding, diversity, thermal |
| **A2A** | `test_a2a_protocol.py`, `test_a2a_server.py`, `test_a2a_identity.py` | Agent-to-agent protocol |
| **Fleet** | `test_sense_decide_act.py`, `test_holodeck.py`, `test_conservation_spectral_bridge.py` | SDA loop, conservation |
| **Logos** | `test_decision_log.py`, `test_codebase_state.py`, `test_signed_wal.py` | Decision journals, WAL |
| **Triage** | `test_triage_modules.py`, `test_drift_detect.py`, `test_metrics.py` | Health, drift, metrics |
| **Perception** | `test_vision_encoder.py`, `test_audio_encoder.py`, `test_cognition_loop.py` | Vision, audio, cognition |
| **Ethos** | `test_ethos.py`, `test_hardware_survey.py`, `test_stress_test.py` | Hardware, thermal |
| **Pathos** | `test_pathos_modules.py` | Need tracking, moment scoring |
| **Compiler** | `test_compiler.py`, `test_hot_swap_integration.py` | Runtime compilation |
| **Integration** | `test_cross_repo_integration.py`, `test_cross_ecosystem_integration.py` | End-to-end |
| **E2E** | `test_breeding_cycle_e2e.py`, `test_conductor_breed_coordination.py` | Full breeding cycles |

### Test Fixtures

`conftest.py` provides:
- **Mock turbovec**: Pure-Python `IdMapIndex` for vector search tests (no external dependency)
- **Mock JEPA**: C mock for JEPA FFI tests
- **NumPy fixtures**: Shared RNG seeds for deterministic tests

---

## Installation

### Requirements

- Python 3.10+
- NumPy, SciPy
- Optional: PyTorch, CUDA toolkit, ONNX Runtime

### Quick Install

```bash
# Clone the repo
git clone https://github.com/SuperInstance/sunset-ecosystem.git
cd sunset-ecosystem

# Install core dependencies
pip install -e .

# Install with dev dependencies (tests, linting)
pip install -e ".[dev]"

# Install with GPU support
pip install -e ".[gpu]"

# Install with perception (webcam, audio)
pip install -e ".[perception]"

# Install with ML backends (transformers, CLIP, whisper)
pip install -e ".[ml]"

# Install everything
pip install -e ".[dev,gpu,perception,ml,vecsearch]"
```

### Building Rust FFI

```bash
cd superinstance-ffi
cargo build --release
# Produces target/release/libsuperinstance_ffi.so
cd ..
python -c "from superinstance_ffi_real import eisenstein_norm; print(eisenstein_norm(3,4))"
```

### Docker

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t sunset-ecosystem .
docker run -p 8850:8850 -p 8080:8080 sunset-ecosystem
```

---

## Development

### Project Structure

```
sunset-ecosystem/
├── a2a/                    # Agent-to-Agent protocol (Google A2A)
├── agents/                 # Specialized agent personas
├── audit/                  # Audit trail and compliance
├── benchmarks/             # Performance benchmarks
├── bin/                    # CLI entry points
├── bottles/                # Agent bottle (state capsule) storage
├── compiler/               # Hot-swap compiler integration
├── conftest.py             # Root pytest fixtures (mock turbovec)
├── dist/                   # Distribution artifacts
├── distill/                # Knowledge distillation
├── docs/                   # Architecture docs and reports
├── ethos/                  # Hardware survey, thermal, stress tests
├── examples/               # Usage examples
├── experiments/            # Research experiments
├── fleet/                  # Sense→Decide→Act, conservation spectral
├── fleet-status/           # Fleet status dashboard data
├── flux_compat/            # FLUX constraint compatibility
├── flux_vm/                # FLUX virtual machine bridge
├── grammar/                # Grammar-based code generation
├── jepa/                   # JEPA (Joint Embedding Predictive Architecture)
├── lessons/                # Learned lessons and post-mortems
├── logos/                  # Decision journals, WAL, codebase state
├── nerve/                  # Room grid, topology, metronome, routing
├── nexus/                  # Fleet conductor, federation, coordination
├── pathos/                 # Human need tracking, moment scoring
├── perception/             # Vision, audio, capture, cognition loop
├── plato_core/             # PLATO room system core
├── ranking/                # Agent ranking algorithms
├── reasoning/              # Polyglot reasoning (C++, Rust, Mercury)
├── scripts/                # Utility scripts
├── simulators/             # Agent behavior simulators
├── skills/                 # Skill definitions
├── sunset/                 # Core agent lifecycle, trinity, compiler
├── superinstance/          # Event bus runtime (COLLECT→SELECT→COMPILE)
├── superinstance-ffi/      # Rust FFI crate
├── swarm/                  # Breeding, tournament, thermal, constraints
├── tests/                  # 473 test files, 2065+ tests
├── triage/                 # Health metrics, drift, duplicate detection
├── voice/                  # Speech synthesis
└── pyproject.toml          # Project configuration
```

### Code Quality

```bash
# Linting
ruff check .
ruff format .

# Type checking
mypy sunset/ swarm/ nerve/ logos/ fleet/ a2a/

# Security audit
bandit -r sunset/ swarm/ nerve/
pip-audit

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

Key principles:
1. Every new module must have tests in `tests/`
2. Every agent-facing change must update the trinity scorer
3. Every breeding change must update the constraint bridge
4. Every A2A change must maintain protocol compliance
5. Run `pytest` before pushing — the fleet depends on it

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Sunset Ecosystem                              │
│              Trinity: ethos × pathos × logos                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │  NERVE   │  │  SWARM   │  │  SUNSET  │  │  LOGOS   │           │
│  │ Forward  │  │ Breeding │  │  Trinity │  │ Decisions│           │
│  │ Inference│  │  Loop    │  │  Scoring │  │ Journals │           │
│  │Room Grid │  │ Diversity│  │  Thermal │  │  Audit   │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │             │             │             │                  │
│       └─────────────┴─────────────┴─────────────┘                  │
│                        │                                            │
│                        ▼                                            │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                 Nexus Fleet Coordination                    │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │   │
│  │  │  BFT   │ │Metronome│ │  Mesh  │ │  A2A   │ │Conserv.│  │   │
│  │  │Consensus│ │  Sync  │ │ Gossip │ │Protocol│ │Spectral│  │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘  │   │
│  └────────────────────────────────────────────────────────────┘   │
│                        │                                            │
│                        ▼                                            │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              SuperInstance Runtime Stack                    │   │
│  │  ┌────────┐ ┌─────────────┐ ┌─────────┐ ┌──────────────┐  │   │
│  │  │si-core-c│ │si-runtime-py│ │si-fleet-│ │   si-cli     │  │   │
│  │  │  FFI   │ │  Event Bus  │ │  API   │ │  Commands    │  │   │
│  │  └────────┘ └─────────────┘ └─────────┘ └──────────────┘  │   │
│  └────────────────────────────────────────────────────────────┘   │
│                        │                                            │
│                        ▼                                            │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              Supabase Fleet Registry                        │   │
│  │  Agents · Budgets · Events · Health · Mesh                 │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## License

MIT License — see [LICENSE](./LICENSE) for details.

---

> *"The fleet decides. Agents sunset with dignity. The next generation seeds from the best of the last. This is not optimization — it is evolution."*
