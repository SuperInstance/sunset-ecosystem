# Integration Guide: sunset-ecosystem

## What This Crate Provides

`sunset-ecosystem` is the Python-native orchestration layer of the SuperInstance fleet. It wires together neural inference, evolutionary breeding, thermal auctions, mesh gossip, drift detection, decision journaling, and ambient visualization into a single coherent runtime.

### Core Modules

- **`nerve.metronome`** — `LocalMetronome`, `MetronomeScheduler`, `SignalSource`, `A2ASignalSource`, `RandomSignalSource`
  Periodic pulse generator that drives the nerve grid. Each beat triggers `RoomGrid.tick()`, with harmonics spawning routing, breeding, and FLUX checks on sub-multiples.

- **`nerve.topology`** — `NerveTopology`, `TickResult`
  Orchestrates the full COLLECT → SELECT → COMPILE → FEEDBACK → REGULATE lifecycle. Wires `NerveFiber.perceive()` → `RoutingLayer.fire()` → `RoomGrid.tick()` → `RoutingLayer.feedback()` with Hebbian channel strengthening.

- **`nerve.world_model`** — `WorldModel`, `WanderingJEPA`, `Calibrator`, `CalibrationReport`
  Predictive world model using JEPA (Joint Embedding Predictive Architecture) with calibration-driven uncertainty estimation.

- **`swarm.breeder`** — `Breeder`, `AgentLifecycle`, `LifecycleRecord`, `spawn_from_template()`
  Tournament breeding pipeline: tournament round → Pareto frontier / sunset candidates → breed → rebirth. Lifecycle states: `SPAWNED → ACTIVE → ADAPTING → COMPILED → SUNSET`.

- **`swarm.thermal_auction`** — `VCGAuction`, `Bid`, `Allocation`
  VCG combinatorial auction for GPU/CPU/iGPU/NPU slot allocation. Truthful bidding is the dominant strategy; winners pay the externality they impose.

- **`swarm.mesh_vector_gossip`** — `MeshVectorGossip`, `GossipDigest`, `DeltaBatch`, `GossipResult`
  Anti-entropy gossip protocol for `FluxVectorTable` CRDTs. Propagates agent DNA vector deltas across fleet nodes using digest-based delta pulls.

- **`sunset.trinity_scorer`** — `trinity_score()`, `normalize_connection()`, `trinity_score_raw()`
  Three-body selection pressure: Ethos (values) × Pathos (emotional resonance) × Logos (logic). If any connection is zero, the agent sunsets.

- **`logos.decision_journal`** — `DecisionJournal`, `Decision`, `log_spawn()`, `log_sunset()`, `log_breed()`, `log_human_command()`, `get_decision_history()`
  Structured FLAME-format log of human-fleet interactions.

- **`logos.tide_pool_viz`** — `TidePoolVisualizer`, `FleetSnapshot`, `AgentSnapshot`
  Ambient bioluminescent visualization of fleet health. Renders HTML and ASCII views for intuition-building without reading metrics.

- **`triage.drift_detect`** — `DriftDetector`, `DriftReport`, `detect_drift()`
  Detects structural drift: stale dependencies, test regression, documentation drift, dead code accumulation, branch divergence.

- **`fleet.fleet_api`** — FastAPI server with endpoints `/health`, `/status`, `/agents`, `/memory/write`, `/memory/query`, `/swarm/knn`, `/cache/stats`
  REST API for fleet-scale mesh vector database operations.

- **`fleet.gossip_protocol`** — `GossipProtocol`, `GossipMessage`
  Epidemic gossip for fleet state propagation. Converges in O(log N) rounds via random peer selection.

- **`fleet.conservation_spectral_bridge`** — `ConservationSpectralEngine`, `SpectralFingerprint`, `SpectralAlignmentScorer`, `ConservationRatioMonitor`
  Bridges entropy conservation (from `conservation-law-rs`) with spectral fleet ranking.

- **`superinstance_ffi_real`** — `eisenstein_norm()`, `laman_is_rigid()`, `holonomy_check()`, `pythagorean48_encode()`, `constraint_check()`, `spline_interpolate()`, `deadband_filter()`
  ctypes bindings to `libsuperinstance_ffi.so` (Rust C FFI).

- **`claw_fleet_bridge`** — `FleetBridgeServer`
  HTTP bridge exposing breeding, consensus, mesh, FLUX, and status to the Claw CLI gateway.

## How to Add This Crate

```bash
git clone https://github.com/SuperInstance/sunset-ecosystem.git
cd sunset-ecosystem
pip install -e .
```

```python
from nerve.metronome import LocalMetronome, MetronomeScheduler
from swarm.breeder import Breeder, AgentLifecycle
from sunset.trinity_scorer import trinity_score_raw
from logos.decision_journal import DecisionJournal, log_spawn
```

## Cross-Repo Connections

### With `conservation-law-rs`: Entropy Budgets for Breeding Rounds

Use the Rust entropy conservation library to enforce per-generation energy budgets, preventing runaway breeding:

```python
from swarm.breeder import Breeder, AgentLifecycle
from swarm.thermal import ThermalBudget
from superinstance_ffi_real import constraint_check, constraint_violation

entropy_budgeted_breed(breeder: Breeder, budget: ThermalBudget) -> bool:
    """Only breed if the proposed mutation stays within the entropy budget."""
    proposed_entropy = breeder.estimate_mutation_entropy()
    within_budget = constraint_check(
        proposed_entropy,
        lower=0.0,
        upper=budget.entropy_limit,
    )
    if not within_budget:
        violation = constraint_violation(
            proposed_entropy, 0.0, budget.entropy_limit
        )
        print(f"Breeding rejected: entropy violation {violation:.4f}")
        return False
    breeder.run_round()
    return True
```

### With `spectral-fleet-rs`: Spectral Ranking of Agents

Compute spectral fingerprints of the agent mesh and rank agents by alignment with the fleet's Fiedler vector:

```python
from fleet.conservation_spectral_bridge import (
    ConservationSpectralEngine,
    SpectralAlignmentScorer,
)
from spectral_fleet::power_iteration::top_k_eigenpairs
import numpy as np

def spectral_agent_ranking(adjacency: np.ndarray, agent_vectors: list[np.ndarray]) -> list[tuple[str, float]]:
    engine = ConservationSpectralEngine()
    fingerprint = engine.compute_fingerprint(adjacency)
    scorer = SpectralAlignmentScorer()
    rankings = []
    for agent_id, vec in agent_vectors:
        alignment = scorer.alignment_score(fingerprint, vec)
        rankings.append((agent_id, alignment))
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings
```

### With `si-cli`: Discovery via Fleet Bridge

The Claw CLI discovers sunset-ecosystem capabilities through the `FleetBridgeServer` HTTP bridge:

```python
from claw_fleet_bridge import FleetBridgeServer

# Start the bridge on port 8850
server = FleetBridgeServer(host="127.0.0.1", port=8850)
server.start()

# si-cli queries:
# GET  /health           → {"status": "ok", "fleet": "cocapn"}
# GET  /status           → FleetConductorV2.get_status()
# GET  /flux/presets     → FluxPresetLibrary.list_presets()
# POST /breed            → trigger Breeder.run_round()
# POST /flux/check       → run FLUX validation
# POST /mesh/insert      → insert vector into MeshVectorTable
# POST /mesh/query       → KNN query across mesh
```

### With `si-fleet-api`: RESTful Fleet Memory

The FastAPI server exposes fleet memory and swarm operations as REST endpoints:

```python
from fleet.fleet_api import app
from fleet.fleet_memory import FleetMemory
from swarm.mesh_vector_tables import MeshVectorTable, VectorTableEntry

# Run with: uvicorn fleet.fleet_api:app --host 0.0.0.0 --port 8000

# POST /agents
#   Insert an agent with vector, fitness, generation, thermal_pressure
# GET  /agents/{agent_id}
#   Retrieve agent metadata
# POST /agents/similar
#   KNN similarity search using VectorTableEntry
# POST /memory/write
#   Write to FleetMemory shard
# POST /memory/query
#   Time-range query with optional fitness filter
# POST /swarm/knn
#   Distributed KNN with quorum consistency
# GET  /cache/stats
#   CognitiveCache hit rates and prediction accuracy
# POST /cache/maintenance
#   Run cache eviction and index compaction
```

### With `si-core-c`: C FFI for Constraint Geometry

Call into `libsuperinstance_ffi.so` (built from `si-core-c` / Rust) for fast geometric constraint checks:

```python
from superinstance_ffi_real import (
    eisenstein_norm,
    laman_is_rigid,
    holonomy_check,
    pythagorean48_encode,
    constraint_check,
    spline_interpolate,
    deadband_filter,
)

# Check if a mesh topology is rigid (Laman's theorem)
is_rigid = laman_is_rigid(num_vertices=10, num_edges=18)

# Check cyclic consistency of agent state transitions
consistency = holonomy_check(
    states=[0.2, 0.4, 0.6, 0.2],
    threshold=0.05,
)

# Smooth thermal readings across time steps
smoothed, last = deadband_filter(
    value=72.5,
    last=71.0,
    deadband=1.0,
)

# Encode frequency ratio for musical agent communication
freq_index = pythagorean48_encode(numerator=3, denominator=2)

# Interpolate between control points for smooth actuation
output = spline_interpolate(p0=0.0, p1=1.0, m0=0.5, m1=0.5, t=0.3)
```

### With `si-runtime-python`: Python Bindings for Fleet Scheduling

The Python runtime bindings expose the fleet scheduler to the sunset ecosystem:

```python
from fleet.fleet_scheduler import FleetScheduler
from fleet.fleet_task_board_bridge import FleetTaskBoardBridge
from fleet.fleet_memory import FleetMemory

scheduler = FleetScheduler()
bridge = FleetTaskBoardBridge(scheduler)

# Enqueue a breeding task with thermal constraints
task_id = bridge.enqueue(
    task_type="breed",
    priority=5,
    thermal_budget={"gpu": 0.4, "cpu": 0.2},
    constraints={"min_fitness": 0.6},
)

# Query task status
status = bridge.status(task_id)
```

## Design Patterns

### Pattern: Metronome-Driven Event Loop

Drive the entire ecosystem from a single metronome pulse, with harmonics triggering subsystems:

```python
from nerve.metronome import LocalMetronome, MetronomeScheduler
from nerve.topology import NerveTopology
from swarm.breeder import Breeder

metronome = LocalMetronome(bpm=120)
scheduler = MetronomeScheduler(metronome)
topology = NerveTopology(n_fibers=8, n_rooms=250)
breeder = Breeder(topology.grid)

# Every beat: tick the topology
scheduler.on_beat(topology.tick)

# Every 4th beat: run a breeding round
scheduler.on_harmonic(4, breeder.run_round)

# Every 16th beat: log a decision journal entry
from logos.decision_journal import log_breed
scheduler.on_harmonic(16, lambda: log_breed(breeder.last_round()))

metronome.start()
```

### Pattern: Thermal-Auction Resource Allocation

Use VCG auctions to fairly allocate GPU/CPU/iGPU/NPU slots among competing agents:

```python
from swarm.thermal_auction import VCGAuction, Bid
from swarm.thermal import ThermalBudget, DeviceType

budget = ThermalBudget(gpu=4, cpu=8, npu=2)
auction = VCGAuction(budget)

# Agents submit bids
auction.submit(Bid(
    agent_id="agent-7",
    device_type=DeviceType.GPU,
    value=0.85,
    fitness=0.92,
))

# Resolve: winners pay VCG externality price
allocations = auction.resolve()
for alloc in allocations:
    print(f"{alloc.agent_id} → {alloc.device_type.value} @ price {alloc.price_paid:.3f}")
```

### Pattern: Anti-Entropy Mesh Gossip

Keep mesh-wide vector tables consistent without a central coordinator:

```python
from swarm.mesh_vector_gossip import MeshVectorGossip, GossipDigest
from swarm.mesh_vector_tables import MeshVectorTable

local_table = MeshVectorTable(table_id="node-1")
gossip = MeshVectorGossip(node_id="node-1", peers=["node-2", "node-3"])

# Periodic gossip round
def sync_round():
    digest = GossipDigest.from_table(local_table)
    for peer_id in gossip.peers:
        deltas = gossip.pull_deltas(peer_id, digest)
        for delta in deltas:
            local_table.apply(delta)

# Run every 5 seconds
import threading
timer = threading.Timer(5.0, sync_round)
timer.start()
```

### Pattern: Drift-Aware CI Gate

Block deployments when drift detection finds structural regression:

```python
from triage.drift_detect import detect_drift

class DriftAwareDeployer:
    def deploy(self, repo_path: str) -> bool:
        report = detect_drift(repo_path)
        if report.severity in ("high", "critical"):
            print(f"BLOCKED: {report.severity} drift detected")
            print(f"  Stale deps: {report.stale_dependencies}")
            print(f"  Dead code: {report.dead_code}")
            return False
        # proceed with deployment
        return True
```

### Pattern: Trinity-Gated Sunset

Sunset agents that fail the trinity connection test (ethos × pathos × logos):

```python
from sunset.trinity_scorer import trinity_score_raw
from swarm.breeder import AgentLifecycle

def maybe_sunset(agent) -> bool:
    score = trinity_score_raw(
        ethos_raw=agent.ethos_connection,
        pathos_raw=agent.pathos_connection,
        logos_raw=agent.logos_connection,
    )
    if score < 0.1:
        agent.lifecycle.state = AgentLifecycle.SUNSET
        return True
    return False
```

### Pattern: Tide Pool Ambient Monitoring

Use the Tide Pool visualizer for ambient fleet awareness without dashboard overload:

```python
from logos.tide_pool_viz import TidePoolVisualizer, FleetSnapshot

viz = TidePoolVisualizer(max_events=10, top_k=5)

def on_tick(fleet_state):
    snapshot = viz.generate_snapshot(fleet_state)
    html = viz.render_html(snapshot)
    ascii_view = viz.render_ascii(snapshot)
    # Display ASCII in terminal, serve HTML on internal dashboard
    print(ascii_view)

viz.auto_refresh(on_tick, interval_seconds=5)
```
