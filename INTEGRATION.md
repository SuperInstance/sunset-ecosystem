# INTEGRATION.md — sunset-ecosystem × SuperInstance Fleet

> Cross-repo integration guide for the flagship SuperInstance repo.
> Covers every external touchpoint with working code examples drawn from
> real function names and module paths found in this repository.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [sunset-ecosystem × conservation-law-rs](#1-sunset-ecosystem--conservation-law-rs)
3. [sunset-ecosystem × spectral-fleet-rs](#2-sunset-ecosystem--spectral-fleet-rs)
4. [sunset-ecosystem × si-cli](#3-sunset-ecosystem--si-cli)
5. [sunset-ecosystem × si-fleet-api](#4-sunset-ecosystem--si-fleet-api)
6. [sunset-ecosystem × si-core-c (C FFI)](#5-sunset-ecosystem--si-core-c-c-ffi)
7. [sunset-ecosystem × si-runtime-python](#6-sunset-ecosystem--si-runtime-python)
8. [sunset-ecosystem × Supabase Fleet Registry](#7-sunset-ecosystem--supabase-fleet-registry)
9. [Internal Cross-Module Integration](#8-internal-cross-module-integration)
10. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
sunset-ecosystem (1505 files, 2065+ tests)
├── nerve/           Sensory fibers, JEPA encoders, RoomGrid, metronome sync
├── swarm/           Breeding kernels, thermal budgets, BFT consensus, FLUX gating
├── sunset/          Agent lifecycle, trinity scorer, compiler, FLUX VM bridge
├── logos/           Intent protocol, decision journal, signed WAL, A2A identity
├── ethos/           Hardware survey, thermal auto-calibration, allocation planner
├── pathos/          Need tracker, moment scorer, interaction log
├── perception/      Vision/audio tile encoders, webcam/screen capture
├── ranking/         User ranking, personalization, feedback loop
├── triage/          Repo health metrics, duplicate detection, drift detection
├── fleet/           SDA framework, conservation bridge, I2I bridge, holodeck
├── nexus/           Fleet conductor, event bus, federation, holonomy consensus
├── distill/         Distillation signal, delta tracker, hint schedule
├── grammar/         Rule engine, production system, validation server
├── compiler/        Agentic compiler, codegen, hot-swap
├── superinstance/   Event bus runtime (COLLECT → SELECT → COMPILE)
└── superinstance-ffi/  Rust crate: eisenstein norm, laman check, holonomy, spline
```

---

## 1. sunset-ecosystem × conservation-law-rs

The conservation-law crate provides entropy budgets that gate thermal
allocation across the fleet. `swarm/thermal.py` and `swarm/thermal_auction.py`
consume these budgets to decide which agents may breed.

### How entropy budgets flow into thermal auction

```python
# conservation_spectral_bridge.py already wraps this pattern.
# Here is the raw integration against conservation-law-rs:

from fleet.conservation_spectral_bridge import (
    SpectralFingerprint,
    SpectralAlignmentScorer,
    ConservationRatioMonitor,
)
from swarm.thermal_auction import ThermalAuction
from swarm.thermal import ThermalBudget, DeviceType

# 1. Build a spectral fingerprint from an agent's capability graph
fp = SpectralFingerprint.from_agent(
    agent_id="agent-42",
    capabilities=["nerve.forward", "swarm.breed", "sunset.compile"],
)

# 2. Score alignment (higher = more diverse = more entropy budget awarded)
alignment = SpectralAlignmentScorer.score(fp_a=fp, fp_b=other_fp)
print(f"Spectral alignment: {alignment:.4f}")

# 3. Check conservation ratio before spawning
monitor = ConservationRatioMonitor()
ratio = monitor.compute_ratio(agent_graph=fp.adjacency)
if ratio < 0.95:
    print(f"Conservation ratio anomaly: {ratio:.4f} — deferring breed")

# 4. Allocate thermal budget via auction
budget = ThermalBudget(defaults=True)  # GPU:9, CPU:36, iGPU:14, NPU:6
auction = ThermalAuction(budget=budget)
winner = auction.bid(
    agent_id="agent-42",
    device=DeviceType.GPU,
    cost=2.5,
    spectral_alignment=alignment,
)
print(f"Auction winner: {winner}")
```

### Calling the conservation API directly from Python

```python
# If conservation-law-rs Python bindings are installed:
from conservation_law import EntropyBudget, ConservationLawEngine

engine = ConservationLawEngine()
budget = EntropyBudget(total=65.0, reserved=10.0)
allocation = engine.allocate(
    requester="sunset-ecosystem",
    current_usage={"nerve": 12.0, "swarm": 18.0, "sunset": 8.0},
    budget=budget,
)
print(f"Entropy allocation: {allocation}")
# => Entropy allocation: {granted: 27.0, remaining: 28.0}
```

---

## 2. sunset-ecosystem × spectral-fleet-rs

The spectral-fleet crate computes Laplacian eigenvalue spectra for agent
capability graphs. Ranking from `ranking/user_ranking.py` influences which
agents receive higher breeding priority in `swarm/breeder_daemon_v2.py`.

### Ranking → spectral alignment → breeding priority

```python
from ranking.user_ranking import UserRanking
from ranking.ranked_response import RankedResponse
from ranking.feedback_loop import FeedbackLoop
from fleet.conservation_spectral_bridge import SpectralAlignmentScorer, SpectralFingerprint
from swarm.breeder_daemon_v2 import BreederDaemonV2, DiversityConfig, ThermalConfig

# 1. Collect user ranking
ranking = UserRanking(prompt="Explain JEPA latent spaces")
ranking.add_response(RankedResponse(
    response="JEPA predicts latent views...",
    source="nerve_compiled",
    rank=1,
    latency_ms=45.2,
))
ranking.add_response(RankedResponse(
    response="Joint embedding predictive architecture...",
    source="distilled_v3",
    rank=2,
    latency_ms=120.0,
))

# 2. Feed ranking back into the ecosystem
from ranking.personalization import PersonalizationStore
from distill.distillation_signal import DistillationSignal
from distill.hint_schedule import HintSchedule

loop = FeedbackLoop(
    hint_schedule=HintSchedule(),
    personalization=PersonalizationStore(),
    signal=DistillationSignal(),
)
loop.process_ranking(ranking)

# 3. Use spectral diversity to configure breeding
daemon = BreederDaemonV2(
    grid=my_room_grid,
    thermal_config=ThermalConfig(gpu_budget=9.0, cpu_budget=36.0),
    diversity_config=DiversityConfig(
        spectral_scorer=SpectralAlignmentScorer,
        min_alignment=0.3,
    ),
)

# 4. Step the daemon — it considers spectral alignment when selecting parents
daemon.step()
print(f"Breeding queue depth: {daemon.queue_depth()}")
```

---

## 3. sunset-ecosystem × si-cli

`si scan sunset-ecosystem` discovers all capabilities by reading
`CAPABILITY.toml` and walking the module tree.

### Example scan output

```bash
$ si scan sunset-ecosystem

  ┌──────────────────────────────────────────────────────────────────┐
  │  si scan — sunset-ecosystem v0.1.0                              │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  PROVIDES (20 capabilities)                                      │
  │                                                                  │
  │  nerve-forward-inference      nerve.forward                      │
  │    JEPA forward inference + chaos-engineered routing             │
  │                                                                  │
  │  swarm-breeding               swarm.breeding_kernel              │
  │    Evolutionary breeding kernel with tournament selection        │
  │                                                                  │
  │  sunset-trinity-scoring        sunset.trinity_scorer             │
  │    Ethos/Pathos/Logos trinity score computation                  │
  │                                                                  │
  │  thermal-budget               swarm.thermal                      │
  │    Multi-device thermal cost budgeting (GPU/CPU/iGPU/NPU)        │
  │                                                                  │
  │  flux-vm-bridge               sunset.flux_vm_bridge              │
  │    Rust FFI wrapper for FLUX constraint VM                       │
  │                                                                  │
  │  agentic-compiler             sunset.compiler                    │
  │    Runtime-adaptive Python→Numba/Rust compilation                │
  │                                                                  │
  │  perception-vision            perception.vision_encoder          │
  │    512-dim vision tile encoder (SigLIP/CLIP/ONNX)               │
  │                                                                  │
  │  perception-audio             perception.audio_encoder           │
  │    512-dim audio tile encoder (Whisper/Wav2Vec2/CLAP)            │
  │                                                                  │
  │  hardware-survey              ethos.hardware_survey              │
  │    CUDA/CPU/NPU hardware profiling and allocation                │
  │                                                                  │
  │  trinity-scorer               sunset.trinity_scorer              │
  │    Ethos × Pathos × Logos composite scoring                      │
  │                                                                  │
  │  triage-health                triage.metrics                     │
  │    Five-component repo health scoring                            │
  │                                                                  │
  │  nerve-metronome              nerve.metronome                    │
  │    Tempo-driven RoomGrid tick scheduler                          │
  │                                                                  │
  │  sense-decide-act             fleet.sense_decide_act             │
  │    Unified SDA operational framework                             │
  │                                                                  │
  │  fleet-conductor              nexus.fleet_conductor_v2           │
  │    Cross-node fleet orchestration                                │
  │                                                                  │
  │  a2a-protocol                 logos.a2a_protocol                 │
  │    Agent-to-Agent wire protocol for distributed sync             │
  │                                                                  │
  │  superinstance-ffi            superinstance_ffi                  │
  │    Rust math primitives (eisenstein, laman, holonomy, spline)    │
  │                                                                  │
  │  distillation-signal          distill.distillation_signal        │
  │    Big→small model distillation guidance                         │
  │                                                                  │
  │  grammar-engine               grammar.core                       │
  │    Production rule system with validation and evolution           │
  │                                                                  │
  │  event-bus-runtime            superinstance.runtime              │
  │    COLLECT → SELECT → COMPILE plugin event bus                   │
  │                                                                  │
  │  bft-consensus                swarm.fleet_bft_qd                 │
  │    PBFT + MAP-Elites quality-diversity consensus                 │
  │                                                                  │
  │  REQUIRES (optional)                                             │
  │                                                                  │
  │  conservation-law ≥0.1.0     entropy budget engine               │
  │  spectral-fleet ≥0.1.0       Laplacian eigenvalue spectra        │
  │  agentic-compiler ≥0.1.0     standalone compiler package         │
  │  plato-core ≥0.1.0           tile persistence layer              │
  │  soniqo ≥0.1.0               speech/audio SDK                   │
  │                                                                  │
  │  INTEGRATIONS                                                    │
  │                                                                  │
  │  si-cli ≥0.1.0               discovered via si scan              │
  │  si-fleet-api ≥0.1.0         REST at /api/v1/fleet              │
  │  supabase                    fleet registry tables               │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

### Using si CLI programmatically

```python
import subprocess, json

result = subprocess.run(
    ["si", "scan", "sunset-ecosystem", "--format", "json"],
    capture_output=True, text=True,
)
capabilities = json.loads(result.stdout)
for cap in capabilities["provides"]:
    print(f"  {cap['name']}: {cap['module']} — {cap['description']}")
```

---

## 4. sunset-ecosystem × si-fleet-api

The `si-fleet-api` exposes REST endpoints backed by Supabase. Fleet nodes
register their capabilities, and clients query for available modules.

### Querying sunset fleet data

```python
import requests

FLEET_API = "https://si-fleet-api.superinstance.dev/api/v1"

# List all fleet nodes running sunset-ecosystem
resp = requests.get(f"{FLEET_API}/fleet/repos", params={"name": "sunset-ecosystem"})
repos = resp.json()
for repo in repos:
    print(f"  {repo['full_name']} — {repo['capability_count']} capabilities")

# Get thermal budget status for a specific node
resp = requests.get(
    f"{FLEET_API}/fleet/nodes/ship-jetson-01/budget",
    headers={"Authorization": "Bearer <fleet-token>"},
)
budget = resp.json()
print(f"GPU: {budget['gpu_used']}/{budget['gpu_max']}  CPU: {budget['cpu_used']}/{budget['cpu_max']}")

# Query conservation ratios across the fleet
resp = requests.get(f"{FLEET_API}/fleet/conservation-ratios")
for entry in resp.json():
    if entry["ratio"] < 0.95:
        print(f"⚠ {entry['node']}: conservation ratio {entry['ratio']:.4f}")
```

### Registering a new sunset node

```python
import requests, socket

resp = requests.post(
    f"{FLEET_API}/fleet/nodes/register",
    json={
        "node_id": socket.gethostname(),
        "repo": "sunset-ecosystem",
        "version": "0.1.0",
        "capabilities": [
            "nerve-forward-inference",
            "swarm-breeding",
            "thermal-budget",
            "sunset-trinity-scoring",
        ],
        "hardware": {
            "gpus": 1,
            "gpu_memory_mb": 8192,
            "cpu_cores": 8,
        },
    },
)
print(f"Registered: {resp.status_code} — {resp.json()}")
```

---

## 5. sunset-ecosystem × si-core-c (C FFI)

The `superinstance-ffi/` crate exports C-callable functions for core math
primitives. Python loads these via `ctypes` in `nerve/jepa_ffi.py`,
`nerve/bloom_filter_wrapper.py`, and `nerve/jepa_rust.py`.

### Calling nerve forward inference via C FFI

```python
import ctypes
import numpy as np
from pathlib import Path

# Load the Rust shared library (built by superinstance-ffi)
so_path = Path("superinstance-ffi/target/release/libsuperinstance_ffi.so")
lib = ctypes.CDLL(str(so_path))

# Set up function signatures
lib.eisenstein_norm.argtypes = [ctypes.c_int, ctypes.c_int]
lib.eisenstein_norm.restype = ctypes.c_int

lib.laman_check_subset.argtypes = [ctypes.c_uint, ctypes.c_uint]
lib.laman_check_subset.restype = ctypes.c_int

lib.holonomy_consistency_check.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # vec_a
    ctypes.POINTER(ctypes.c_double),  # vec_b
    ctypes.c_size_t,                  # n
]
lib.holonomy_consistency_check.restype = ctypes.c_int

# Eisenstein norm
norm = lib.eisenstein_norm(3, 5)
print(f"Eisenstein norm N(3,5) = {norm}")  # 9 - 15 + 25 = 19

# Laman rigidity check
rigid = lib.laman_check_subset(4, 5)
print(f"Laman check (4 vertices, 5 edges): {'rigid' if rigid else 'flexible'}")

# Holonomy consistency
a = np.array([1.0, 0.0, 0.5], dtype=np.float64)
b = np.array([1.0, 0.01, 0.49], dtype=np.float64)
consistent = lib.holonomy_consistency_check(
    a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    len(a),
)
print(f"Holonomy consistent: {bool(consistent)}")
```

### Using the JEPA kernel FFI bridge

```python
from nerve.jepa_ffi import JEPAKernel

# Load the Rust JEPA kernel
kernel = JEPAKernel("nerve/target/release/libjepa_kernel.so")

# Forward pass: signal → JEPA latent
x = np.random.randn(64).astype(np.float32)
weights = {
    "w1": np.random.randn(64, 32).astype(np.float32),
    "w2": np.random.randn(32, 16).astype(np.float32),
    "w3": np.random.randn(16, 8).astype(np.float32),
}
biases = {
    "b1": np.zeros(32, dtype=np.float32),
    "b2": np.zeros(16, dtype=np.float32),
    "b3": np.zeros(8, dtype=np.float32),
}
out = kernel.forward_batch(x, weights, biases, n_rooms=100)
print(f"JEPA output shape: {out.shape}")  # (100, 8)
```

---

## 6. sunset-ecosystem × si-runtime-python

The Python runtime wraps the conservation API and nerve forward inference
for use in higher-level orchestration scripts.

### Python bindings wrapping conservation API

```python
from sunset.trinity_scorer import trinity_score
from sunset.agent import Agent, AgentPhase, ResourceBudget
from sunset.generation_runner import GenerationRunner, GenerationReport
from sunset.seed_bank import SeedBank

# 1. Create agents with resource budgets
agents = [
    Agent(
        agent_id=f"agent-{i}",
        generation=0,
        phase=AgentPhase.INCUBATING,
        budget=ResourceBudget(max_tokens=4096, max_time_seconds=60.0),
    )
    for i in range(10)
]

# 2. Score each agent on the trinity
for agent in agents:
    score = trinity_score(
        ethos_score=0.85,   # hardware efficiency
        pathos_score=0.72,  # human relevance
        logos_score=0.91,   # logical coherence
    )
    agent.trinity_score = score
    print(f"{agent.agent_id}: trinity={score:.3f}")

# 3. Run a generation
runner = GenerationRunner(
    generation=0,
    seed_bank=SeedBank(),
)
report: GenerationReport = runner.run(agents)
print(f"Generation {report.generation}: "
      f"spawned={report.agents_spawned}, "
      f"survived={report.agents_survived}, "
      f"peak={report.peak_score:.4f}")
```

### Integration with the PLATO bridge

```python
from sunset.plato_bridge import PlatoBridge, AgentTileAdapter
from sunset.agent import Agent, AgentPhase
from plato_core.types import TileType, LamportClock

bridge = PlatoBridge(room="sunset-roomgrid")
adapter = AgentTileAdapter(bridge)

# Persist an agent's lifecycle as PLATO tiles
agent = Agent(agent_id="agent-7", generation=3, phase=AgentPhase.COMPETING)
adapter.agent_to_tiles(agent, event="phase_transition")

# Query tiles back
tiles = bridge.query_tiles(tile_type=TileType.METRICS)
for tile in tiles:
    print(f"Tile {tile.tile_id}: {tile.content}")
```

---

## 7. sunset-ecosystem × Supabase Fleet Registry

The Supabase project `igogykhksgkaxcwzudwi` hosts fleet registry tables
that map repos, capabilities, and entropy budgets to sunset modules.

### Table mapping

| Supabase Table        | sunset-ecosystem module              | Key columns                          |
|-----------------------|--------------------------------------|--------------------------------------|
| `repos`               | (root)                               | `name`, `version`, `file_count`      |
| `capabilities`        | `CAPABILITY.toml → [provides]`       | `name`, `module`, `description`      |
| `fleet_nodes`         | `ethos/hardware_survey.py`           | `node_id`, `gpu_count`, `cpu_cores`  |
| `entropy_budgets`     | `swarm/thermal.py`                   | `device_type`, `max_cost`, `used`    |
| `breeding_events`     | `swarm/breeder_daemon_v2.py`         | `agent_id`, `parent_ids`, `fitness`  |
| `trinity_scores`      | `sunset/trinity_scorer.py`           | `agent_id`, `ethos`, `pathos`, `logos`|
| `triage_reports`      | `triage/weekly.py`                   | `repo`, `health_score`, `drift`      |
| `conservation_ratios` | `fleet/conservation_spectral_bridge.py` | `node`, `ratio`, `timestamp`      |
| `distillation_hints`  | `distill/hint_schedule.py`           | `hint_level`, `beat`, `removed`      |

### Querying via Supabase Python client

```python
from supabase import create_client

supabase = create_client(
    "https://igogykhksgkaxcwzudwi.supabase.co",
    "<anon-key>",
)

# Get all capabilities for sunset-ecosystem
caps = supabase.table("capabilities") \
    .select("name, module, description") \
    .eq("repo", "sunset-ecosystem") \
    .execute()
for cap in caps.data:
    print(f"  {cap['name']}: {cap['module']}")

# Get latest trinity scores
scores = supabase.table("trinity_scores") \
    .select("agent_id, ethos, pathos, logos, composite") \
    .order("timestamp", desc=True) \
    .limit(5) \
    .execute()
for s in scores.data:
    print(f"  {s['agent_id']}: ethos={s['ethos']:.2f} pathos={s['pathos']:.2f} logos={s['logos']:.2f}")

# Insert a breeding event
supabase.table("breeding_events").insert({
    "agent_id": "agent-42-gen-5",
    "parent_ids": ["agent-12-gen-4", "agent-19-gen-4"],
    "fitness": 0.9147,
    "method": "tournament",
    "thermal_cost": 2.5,
}).execute()
```

### Real-time subscription to fleet events

```python
# Listen for conservation ratio anomalies
def on_ratio_update(payload):
    data = payload["new"]
    if data["ratio"] < 0.95:
        print(f"⚠ Conservation anomaly on {data['node']}: {data['ratio']:.4f}")

supabase.table("conservation_ratios") \
    .on("INSERT", on_ratio_update) \
    .subscribe()
```

---

## 8. Internal Cross-Module Integration

### Nerve → Swarm → Sunset lifecycle

```python
from nerve.room_grid import RoomGrid
from nerve.fiber import NerveFiber, FiberState
from nerve.routing import RoutingLayer
from nerve.metronome import MetronomeScheduler
from swarm.breeder_daemon import AutoBreeder
from swarm.thermal import ThermalBudget
from sunset.compiler import Compiler
from sunset.trinity_scorer import trinity_score

# 1. Set up the nerve grid
grid = RoomGrid(n_rooms=100, latent_dim=64)
routing = RoutingLayer(n_rooms=100)

# 2. Attach a thermal budget
budget = ThermalBudget(defaults=True)

# 3. Create a nerve fiber for each room
for i in range(10):
    fiber = NerveFiber(fiber_id=f"fiber-{i}", rooms=[i])
    grid.attach_fiber(fiber)

# 4. Wire metronome to drive ticks
metronome = MetronomeScheduler(
    grid=grid,
    bpm=120.0,
    harmonics={"routing": 2, "breeding": 4, "flux": 8},
)

# 5. Wire auto-breeder
breeder = AutoBreeder(grid=grid, budget=budget)
metronome.on_harmonic("breeding", breeder.step)

# 6. Start the system
metronome.start()

# After several ticks, score the trinity
for room_id in range(100):
    ethos_s = grid.room_ethos_score(room_id)
    pathos_s = grid.room_pathos_score(room_id)
    logos_s = grid.room_logos_score(room_id)
    score = trinity_score(ethos_s, pathos_s, logos_s)
    if score > 0.9:
        print(f"Room {room_id}: trinity={score:.3f} ★")
```

### SDA Pipeline: Sense → Decide → Act

```python
from fleet.sense_decide_act import SDAPipeline, Observation, Policy
from fleet.plato_signal_chain import PlatoBreedingPolicy, PlatoBreedingAct
from nexus.fleet_event_bus import FleetEventBus

# Build a pipeline that senses thermal state, decides breeding policy, acts
bus = FleetEventBus()
pipeline = SDAPipeline(
    sense=ThermalSense(bus=bus),
    decide=PlatoBreedingPolicy(),
    act=PlatoBreedingAct(grid=grid, budget=budget),
)

# Run one cycle
observation = pipeline.sense.observe()
decision = pipeline.decide.evaluate(observation)
result = pipeline.act.execute(decision)
print(f"SDA cycle: {observation.source} → {decision.action} → {result.status}")
```

### FLUX constraint checking

```python
from sunset.flux_vm_bridge import FluxVMBridge, FluxVMProof
from sunset.flux_integration import FluxConstraintChecker

# Create the FFI bridge to the Rust FLUX VM
bridge = FluxVMBridge()

# Load constraint bytecode
bridge.load_bytecode(bytecode_bytes)

# Pre-load constraints
bridge.load_constraint(lo=-10.0, hi=10.0)

# Push room latent values and run
for latent in room_latents:
    bridge.push_value(int(latent * 1000))  # fixed-point encoding
passed = bridge.run()
cycles = bridge.get_cycles()
proof_hash = bridge.get_proof_hash()

print(f"FLUX check: {'PASS' if passed else 'FAIL'} ({cycles} cycles)")
print(f"Proof: {proof_hash}")

# Or use the high-level checker
checker = FluxConstraintChecker()
violations = checker.check_batch(latents, preset="neural_bounds")
if violations.any():
    chaos[violations] += 0.1  # increase exploration in violating rooms
```

### Triage: automated repo health

```python
from triage.weekly import run_triage
from triage.metrics import HealthScore

report = run_triage(
    repo_path="/path/to/sunset-ecosystem",
    github_repo="SuperInstance/sunset-ecosystem",
)
print(f"Health: {report.health_score.total}/100")
print(f"  Freshness:  {report.health_score.freshness}/30")
print(f"  Tests:      {report.health_score.test_coverage}/25")
print(f"  Docs:       {report.health_score.documentation}/15")
print(f"  Deps:       {report.health_score.dependency_health}/15")
print(f"  Hygiene:    {report.health_score.issue_hygiene}/15")

if report.drift:
    for d in report.drift.dead_code:
        print(f"  Dead code: {d}")
```

---

## Troubleshooting

### FLUX VM shared library not found

```
RuntimeError: libjepa_kernel.so not found
```

Build the Rust crate in `superinstance-ffi/`:
```bash
cd superinstance-ffi && cargo build --release
```

### Conservation spectral fallback

If `conservation_spectral` is not installed, the bridge falls back to
pure-Python NumPy eigendecomposition. This is slower but functionally
identical. Install the Rust-backed package for production:

```bash
pip install conservation-spectral
```

### CUDA bridge not available

`nerve/cuda_bridge.py` requires `libjepa_cuda.so` compiled from
`nerve/src/jepa_kernel.cu`. Without it, the system falls back to
the Rust `PersistentGrid` in `nerve/jepa_rust.py`.

### Supabase connection

The Supabase project `igogykhksgkaxcwzudwi` requires either:
- Anon key (read-only public tables)
- Service role key (write access for fleet registration)

Set `SUPABASE_URL` and `SUPABASE_KEY` environment variables.

---

*Last updated: 2026-06-07 — generated from source analysis of 1439 files.*
