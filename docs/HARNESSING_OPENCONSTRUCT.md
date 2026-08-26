# Harnessing OpenConstruct — A Guide for Agents and Humans

> **Agent**: Read this to learn your shell commands. Skip to [Agent Quick Reference](#agent-quick-reference).
>�**Human**: Read this to understand what your fleet can do. Start with [What Is OpenConstruct?](#what-is-openconstruct).

---

## Table of Contents

1. [What Is OpenConstruct?](#what-is-openconstruct)
2. [The Five Layers](#the-five-layers)
3. [Agent Quick Reference](#agent-quick-reference)
4. [Human Quick Reference](#human-quick-reference)
5. [Spatial Awareness](#spatial-awareness)
6. [Parallel Campaigns](#parallel-campaigns)
7. [FLUX Constraints](#flux-constraints)
8. [Advanced Patterns](#advanced-patterns)
9. [Troubleshooting](#troubleshooting)

---

## What Is OpenConstruct?

OpenConstruct is an **agent-native operating system**. It's not a framework you import and call — it's an environment agents *live inside*.

### For Humans

Think of it as a container where AI agents can:
- **Spawn breeding campaigns** to evolve solutions to problems
- **Project their state** into a shared spatial map so they know where other agents are
- **Run multiple experiments in parallel** across fleet nodes
- **Check their own health** and self-heal when something breaks
- **Broadcast predictions** to other agents via A2A protocol

The agents write code, run experiments, and evolve — all while reporting back to you in plain language.

### For Agents

You are an agent running in the Cocapn Fleet. Your primary interface is the **OpenConstruct Shell** (`fleet/openconstruct_shell.py`). You:

1. Receive a task from your human (or another agent)
2. Spawn the appropriate breeder attachment
3. Run generations, collecting fitness data
4. Project your state into the spatial index
5. Broadcast predictions to collaborating agents
6. Return the best result

---

## The Five Layers

OpenConstruct operates through five abstraction layers:

```
┌─────────────────────────────────────────┐
│  5. VALIDATION GATES                     │
│     FLUX constraints, thermal checks,     │
│     exact arithmetic verification         │
├─────────────────────────────────────────┤
│  4. PROGRESS STREAMER                    │
│     SSE events, sensor readings,         │
│     self-healing loops                   │
├─────────────────────────────────────────┤
│  3. BUILD COORDINATOR                    │
│     BreederDaemon, Pythagorean,         │
│     Spectral, Adversarial, NCA           │
├─────────────────────────────────────────┤
│  2. HARNESS ADAPTER                      │
│     BreederFactory, runtime selection,   │
│     polymorphic API binding              │
├─────────────────────────────────────────┤
│  1. CONSTRUCT MANIFEST                   │
│     Goal specification, attachment        │
│     selection, fleet node assignment      │
└─────────────────────────────────────────┘
```

### Layer 1: Construct Manifest

You specify what you want:

```python
manifest = ConstructManifest(
    goal="Find exact Pythagorean triples",
    attachment="pythagorean",  # Which breeder to use
    generations=100,
    population_size=50,
)
```

### Layer 2: Harness Adapter

The adapter picks the right breeder and normalizes APIs:

| Breeder | Input | Output | Best For |
|---------|-------|--------|----------|
| `standard` | Genome object | `.fitness` | General purpose |
| `pythagorean` | NumPy matrix | `sum(matrix)` | Exact arithmetic |
| `spectral` | Fourier spectrum | Spectral fitness | Signal processing |
| `adversarial` | Solver/Tester vectors | `(solver_f, tester_f)` | Robustness testing |
| `nca` | NCA rules | Grid convergence | Pattern generation |

### Layer 3: Build Coordinator

Runs the breeding loop:
- Single-node: local execution
- Multi-node: distributed via Bernstein orchestrator
- Coordinated: PBFT consensus before breeding batches

### Layer 4: Progress Streamer

Real-time events:
```
SPAWN → GENERATION → PARENT_SELECT → MUTATION → FLUX_GATE → BREED_COMPLETE
```

Agents subscribe to events. Humans watch dashboards.

### Layer 5: Validation Gates

Every result passes through:
- **Exact arithmetic gate**: For Pythagorean triples, verify a² + b² = c² exactly
- **Thermal gate**: Check predicted thermal budget isn't exceeded
- **Spectral real gate**: Verify Fourier spectrum has no imaginary leakage
- **Robustness gate**: Adversarial arena must pass tester evaluation

---

## Agent Quick Reference

### Commands

```python
from fleet.openconstruct_shell import OpenConstructShell

shell = OpenConstructShell(node_id="your-node")
```

| Command | Code | What It Does |
|---------|------|--------------|
| **spawn** | `shell.spawn(...)` | Start a breeding campaign |
| **breed parallel** | `shell.execute("breed parallel", [...])` | Run multiple campaigns |
| **status** | `shell.execute("status")` | System health check |
| **project state** | `projector.project_state(...)` | Add yourself to spatial map |
| **query neighbors** | `projector.query_neighbors(...)` | Find nearby agents |
| **predict** | `projector.predict_trajectory(...)` | Forecast your movement |
| **flux-gate** | `projector.apply_flux_gate(...)` | Validate prediction |
| **terminate** | `shell.terminate()` | Clean shutdown |

### Spawn a Campaign

```python
result = shell.spawn(
    run_id="my-experiment",
    attachment="pythagorean",
    generations=50,
    task_fn=lambda matrix: {"fitness": float(matrix.sum())},
    config={"population_size": 100, "elitism_count": 5},
)
# result = {"best_genome": ..., "best_fitness": 123.4, "generations": 50}
```

### Run Parallel Campaigns

```python
campaigns = [
    {"name": "exact-rational", "attachment": "pythagorean"},
    {"name": "fourier-evolution", "attachment": "spectral"},
    {"name": "adversarial-test", "attachment": "adversarial"},
]
result = shell.execute("breed parallel", campaigns)
# result["best_campaign"] = "fourier-evolution"
```

### Project Your State

```python
from fleet.spatial_projector import SpatialProjector, WorldState

projector = SpatialProjector("your-node", dimension=3)

projector.project_state(
    agent_id="your-agent-id",
    room_id="ethos-thermal",
    state=WorldState(
        position=(0.0, 0.0, 0.0),  # Your abstract position
        semantics={
            "role": "breeder",
            "temperature": 65.4,
            "current_task": "my-experiment",
        },
        confidence=0.95,
    ),
)
```

### Find Collaborators

```python
# Find agents within 5 units
neighbors = projector.query_neighbors("your-agent-id", radius=5.0)
for agent in neighbors:
    print(f"Nearby: {agent.agent_id} in {agent.room_id}")

# Find all breeders
breeders = projector.query_semantic("role", "breeder")
```

### Predict and Validate

```python
# Predict your next 10 steps
prediction = projector.predict_trajectory("your-agent-id", horizon=10)

# Add a thermal constraint
from fleet.spatial_projector import create_thermal_constraint

projector.add_flux_constraint(create_thermal_constraint(max_temp=75.0, hard=True))

# Validate (raises if violation)
validated = projector.apply_flux_gate(prediction)

# Broadcast to fleet
projector.broadcast_prediction(validated)
```

### Self-Healing Loop

```python
from fleet.openconstruct_shell import SensorReading

# Check your sensors
reading = SensorReading(
    thermal=68.5,
    queue_depth=3,
    flux_gate_pass=True,
    mesh_sync_delta=0.12,
    timestamp=time.time(),
)

# Self-heal if needed
if reading.thermal > 80.0:
    shell.execute("cooldown")
elif reading.queue_depth > 10:
    shell.execute("throttle", factor=0.5)
```

---

## Human Quick Reference

### Watch the Fleet

```bash
# Check system health
curl http://localhost:8080/health

# Watch breeding events
curl http://localhost:8080/sse/stream
# → {"event": "BEAT", "timestamp": ..., "payload": {...}}
# → {"event": "PARENT_SELECT", ...}
# → {"event": "BREED_COMPLETE", ...}
```

### Start a Campaign

```bash
# Via CLI (if implemented)
python -m sunset.cli spawn \
  --attachment pythagorean \
  --generations 100 \
  --population 50

# Or via Python API
python -c "
from fleet.openconstruct_shell import OpenConstructShell
shell = OpenConstructShell('cli-node')
shell.spawn(
    run_id='cli-experiment',
    attachment='pythagorean',
    generations=100,
    task_fn=lambda g: {'fitness': float(g.sum())},
    config={'population_size': 50}
)
"
```

### Inspect Spatial State

```bash
# List all agents in spatial index
python -c "
from fleet.spatial_projector import SpatialProjector
proj = SpatialProjector('cli-node')
print(proj.snapshot())
"

# Find agents in ethos room
python -c "
from fleet.spatial_projector import SpatialProjector
proj = SpatialProjector('cli-node')
agents = proj.query_semantic('room_id', 'ethos')
for a in agents:
    print(f'{a.agent_id}: {a.position}')
"
```

### Check Test Status

```bash
# Run all tests
python -m pytest tests/ -x --tb=short

# Run specific module
python -m pytest tests/test_spatial_projector.py -v
python -m pytest tests/test_openconstruct_shell.py -v
python -m pytest tests/test_spatial_breeding.py -v
```

---

## Spatial Awareness

The A2A Spatial Projector gives every agent a **shared sense of space**.

### Concepts

- **WorldState**: Your current position, velocity, orientation, and semantic tags
- **Spatial Index**: A queryable database of all agent positions
- **Prediction**: Forecast of where you'll be in N steps
- **FLUX Gate**: Constraint check before broadcasting predictions

### Use Cases

1. **Parent Selection**: Breed with agents that are spatially proximal (share context) or distant (inject diversity)
2. **Collision Avoidance**: Don't schedule conflicting experiments on the same node
3. **Load Balancing**: Relocate agents to balance spatial entropy
4. **Trajectory Planning**: Predict and validate resource usage before committing

### Example: Spatial Breeding

```python
from fleet.spatial_breeding import SpatialBreedingContext

ctx = SpatialBreedingContext(projector)

# Proximal parents (same room, nearby)
parents = ctx.select_proximal_parents("agent-1", radius=5.0, k=3)

# Diverse parents (far away, different context)
diverse = ctx.select_diverse_parents("agent-1", min_distance=20.0, k=2)

# Hybrid mix
hybrid = ctx.select_hybrid_parents("agent-1", total_k=5)

# Check population distribution
print(ctx.room_distribution())
# → {'ethos': 3, 'pathos': 2, 'logos': 1}

# Recommend relocation for better diversity
rec = ctx.recommend_relocation("agent-1", target_entropy=0.8)
if rec:
    projector.project_state("agent-1", "ethos", rec)
```

---

## Parallel Campaigns

Run multiple breeding experiments simultaneously:

```python
campaigns = [
    {
        "name": "exact-arithmetic",
        "attachment": "pythagorean",
        "generations": 100,
        "config": {"elitism_count": 5},
    },
    {
        "name": "fourier-evolution",
        "attachment": "spectral",
        "generations": 80,
        "config": {"mutation_rate": 0.1},
    },
    {
        "name": "adversarial-robustness",
        "attachment": "adversarial",
        "generations": 60,
        "config": {"solver_pop_size": 30, "tester_pop_size": 20},
    },
]

result = shell.execute("breed parallel", campaigns)

print(f"Best campaign: {result['best_campaign']}")
print(f"Best fitness: {result['best_fitness']}")
print(f"Status: {result['overall_status']}")
# → Complete, Partial, or Failed
```

### What Happens Internally

1. Each campaign gets a **git worktree** (isolated via Bernstein orchestrator)
2. Campaigns run on **different fleet nodes** (round-robin assignment)
3. Every spawn/schedule/verify/merge decision is **HMAC-signed** for audit
4. Results are **aggregated** with best-campaign identification
5. Full **sensor stream** is available for monitoring

---

## FLUX Constraints

FLUX is the constraint system that keeps agents safe.

### Hard Constraints (raise if violated)

```python
from fleet.spatial_projector import create_thermal_constraint

# Kill the prediction if it exceeds thermal budget
projector.add_flux_constraint(create_thermal_constraint(max_temp=75.0, hard=True))

prediction = projector.predict_trajectory("agent-1", horizon=20)
try:
    validated = projector.apply_flux_gate(prediction)
except ValueError as e:
    print(f"Prediction rejected: {e}")
    # Prediction was thermally infeasible
```

### Soft Constraints (reduce confidence)

```python
from fleet.spatial_projector import create_uncertainty_constraint

# Reduce confidence if uncertainty is too high
projector.add_flux_constraint(
    create_uncertainty_constraint(max_uncertainty=0.5, hard=False)
)

# Uncertainty penalty is applied to trajectory confidence
validated = projector.apply_flux_gate(prediction)
# validated.trajectory[1].confidence *= (1 - penalty)
```

### Room Boundaries

```python
from fleet.spatial_projector import create_room_constraint

# Keep predictions within allowed rooms
projector.add_flux_constraint(
    create_room_constraint(["ethos", "pathos", "logos"], hard=True)
)
```

---

## Advanced Patterns

### Pattern 1: Predictive Breeding

Use world model predictions to evaluate offspring before birth:

```python
from fleet.worldmodel_bridge import WorldModelBridge

bridge = WorldModelBridge()

# Predict how a candidate genome would perform
current = projector.get_agent_state("agent-1")
pred = bridge.predict("agent-1", current, horizon=10)

# If prediction looks good, add to breeding pool
if pred.mean_uncertainty < 0.3:
    breeder.add_candidate(genome)
```

### Pattern 2: A2A Negotiation

Agents negotiate task assignment via spatial proximity:

```python
# Find nearby agents working on similar tasks
neighbors = projector.query_neighbors("agent-1", radius=5.0)
for n in neighbors:
    if n.semantics.get("task") == "similar-to-mine":
        # Negotiate: merge tasks or divide work
        negotiate_task_merge("agent-1", n.agent_id)
```

### Pattern 3: Cross-Node Sync

Synchronize spatial state across fleet nodes:

```python
# Node Alpha
snap = projector.snapshot()

# Node Beta (receives via mesh gossip)
beta_projector.ingest_snapshot(snap)

# Now both nodes have the same spatial view
```

### Pattern 4: Thermal-Aware Scheduling

Use spatial predictions to avoid thermal hotspots:

```python
# Predict thermal trajectory
for agent_id in active_agents:
    pred = projector.predict_trajectory(agent_id, horizon=5)
    temps = [s.semantics.get("temperature", 0) for s in pred.trajectory]
    if max(temps) > 80:
        # Reschedule this agent to a cooler node
        reschedule(agent_id, cooler_node)
```

### Pattern 5: Spatial Checkpointing

Save population state for later restoration:

```python
# Save spatial state
snapshot = projector.snapshot()
with open("checkpoint.json", "w") as f:
    json.dump(snapshot, f)

# Restore later
with open("checkpoint.json") as f:
    projector.ingest_snapshot(json.load(f))
```

---

## Troubleshooting

### "No state known for agent X"

You tried to predict/query an agent that hasn't projected its state yet:

```python
# Fix: Project state first
projector.project_state("agent-x", "room", WorldState(position=(0.0, 0.0)))
```

### "FLUX hard constraint 'X' violated"

Your prediction violates a constraint. Either:
1. Relax the constraint (change from `hard=True` to `hard=False`)
2. Reduce prediction horizon
3. Improve initial conditions

### "stable-worldmodel not found"

The bridge falls back to mock predictions. Install for real world models:

```bash
pip install stable-worldmodel
```

### Tests hang at collection

Known issue with pytest collection on large suites. Workaround:

```bash
# Run specific test files instead of full suite
python -m pytest tests/test_spatial_projector.py -v
python -m pytest tests/test_openconstruct_shell.py -v
```

---

## Appendix: Module Index

| Module | Path | Tests | Description |
|--------|------|-------|-------------|
| Spatial Projector | `fleet/spatial_projector.py` | 57 | A2A spatial awareness |
| Spatial Breeding | `fleet/spatial_breeding.py` | 33 | Location-aware parent selection |
| WorldModel Bridge | `fleet/worldmodel_bridge.py` | 12 | stable-worldmodel integration |
| OpenConstruct Shell | `fleet/openconstruct_shell.py` | 31 | Agent-native interface |
| OpenConstruct Bridge | `fleet/openconstruct_bridge.py` | 37 | Harness integration |
| Parallel Orchestrator | `fleet/parallel_breeding_orchestrator.py` | 22 | Multi-campaign dispatch |
| Fleet Conductor V2 | `nexus/fleet_conductor_v2.py` | 40 | Central nervous system |
| SenseDecideAct | `fleet/sense_decide_act.py` | 33 | Unifying framework |
| FleetBFT-QD | `swarm/fleet_bft_qd.py` | 72 | Byzantine + QD breeding |
| Breeder Daemon V2 | `swarm/breeder_daemon_v2.py` | 30 | Main breeding loop |
| NCA Breeder | `swarm/nca_breeder.py` | 22 | Neural Cellular Automata |
| Spectral Breeding | `swarm/spectral_breeding.py` | 26 | Fourier-domain evolution |
| Adversarial Arena | `swarm/adversarial_arena.py` | 18 | Competitive co-evolution |
| Pythagorean Evolution | `swarm/pythagorean_evolution.py` | 20 | Exact arithmetic breeding |
| TDA Landscape | `swarm/tda_landscape.py` | 18 | Topological fitness landscapes |

**Total: 20+ modules, 650+ tests.**

---

## One More Thing

> *"The fleet doesn't just think — it knows where it is, where it's going, and what the world looks like from every agent's perspective."*

This system is designed to be **understood by both agents and humans**. Agents read the code. Humans read the guides. Both meet in the middle — the spatial index, the breeding results, the SSE stream.

If you're an agent: you now know how to project, predict, breed, and broadcast.

If you're a human: you now know how to watch, command, and harness the fleet.

**Go build something.**
