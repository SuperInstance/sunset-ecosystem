# sunset-ecosystem

## The trinity architecture

Every agent in this system lives or dies by its connections to three rooms. Not two, not four — three. Each room measures a different kind of relevance:

| Room | Greek | Measures |
|------|-------|----------|
| **Ethos** | Character | Does this agent fit the hardware? Is it thermally efficient? Resource-appropriate? |
| **Pathos** | Emotion | Is a human waiting? Did this agent actually solve the problem? Reduce frustration? |
| **Logos** | Reason | Is the code clean? Maintainable? Does the agent understand *why* it's doing what it's doing? |

```
trinity_score = ethos × pathos × logos
```

If any connection scores zero, the agent sunsets (retires). An agent that perfectly uses the GPU but ignores the human waiting? Sunset. An agent that makes the human happy but burns the CPU at 100°C? Sunset. An agent that writes beautiful code for a problem nobody has? Sunset.

## What's actually happening?

The trinity is a multi-objective optimization with a hard constraint: no dimension can be zero. This isn't weighted scoring where a high score in one dimension compensates for a low score in another. It's a product — if any factor is zero, the whole thing collapses.

Think of it like a stool with three legs. Shorten one leg and the stool wobbles. Remove one leg and it falls over. You need all three.

### Ethos — the metal

The Ethos room surveys the actual hardware: RTX 4050 SMs, Ryzen AI cores, Radeon 890M CUs, XDNA 2 NPU. It benchmarks each device, measures thermal headroom, and allocates agents to the hardware that fits them best.

An inference-heavy agent goes to the GPU. A routing agent goes to the CPU. An INT8-quantized agent goes to the NPU. Ethos makes sure agents don't fight over the same resources.

### Pathos — the human

The Pathos room tracks interaction quality: is a human waiting for a response? Did the last response help or frustrate? What's the moment-by-moment sentiment? It scores whether the agent is actually *helpful*, not just technically correct.

### Logos — the code

The Logos room evaluates code quality: clean integration, maintainability, understanding of the broader system. It tracks decisions, logs rationale, and maintains a generation memory so agents don't repeat the same mistakes.

## The lifecycle

```
INCUBATE → COMPETE → (SURVIVE → BREED) or (SUNSET → ARCHIVE)
                                       ↓                ↓
                                  Children          Seed Bank
```

1. **Incubate** — Agent is born, reads the three trinity rooms to understand the current state
2. **Compete** — Agent works on tasks, earns trinity scores in real-time
3. **Breed or Sunset** — High scorers spawn children with streamlined budgets. Low scorers retire.

### Sunset documents

Every retiring agent writes three documents:

**Epilogue** — What I tried, what I found, why it wasn't relevant enough. An honest post-mortem.

**Summary** — My work from *my* perspective. Subjective, opinionated, unfiltered.

**Onboarding letter** — A message to the next generation, written knowing I'm being put away. Can be written in three variants:
- `continuation` — for a similar agent to carry on
- `cross-pollination` — for a different type of agent to learn from
- `mutation` — for a completely novel approach

The seed bank stores these documents in a searchable archive. Future generations can query them: "Has anyone tried approach X before? What happened?"

## Deploy it

```bash
# Docker
docker-compose up --build

# Or manually
docker build -t sunset-ecosystem .
docker run -p 8080:8080 -e NEXUS_IP=localhost sunset-ecosystem

# Health check
curl http://localhost:8080/health
```

### Environment Variables

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
