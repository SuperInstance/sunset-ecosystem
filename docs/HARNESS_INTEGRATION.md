# Harness Integration Architecture

## Design Goals

Make sunset-ecosystem a first-class breeding backend for any harnessing system (OpenConstruct, OpenHarness, etc.). The harness provides the "Define + Orchestrate" layer; sunset-ecosystem provides the "Execute + Validate" layer with novel breeding algorithms.

## Core Abstraction: The Construct

A Construct is a declarative breeding specification that a harness can spawn and monitor:

```python
@dataclass
class ConstructManifest:
    name: str                    # Unique identifier
    breeder_type: str            # "pythagorean" | "spectral" | "adversarial" | "standard"
    goal: str                  # Human-readable objective
    population_size: int
    generations: int
    constraints: List[str]     # FLUX constraint names
    qd_dimensions: List[Tuple[int, int, int]]  # For QD archives
    resources: Dict[str, Any]  # Node count, GPU, etc.
```

## Integration Points

### 1. Harness Adapter
- Translates harness construct manifests into sunset-ecosystem breeder instances
- Adapts harness task definitions to fitness functions
- Reports progress back to harness in real-time

### 2. Build Coordinator
- Multi-node BFT consensus for distributed breeding
- Each node runs breeder replica; PBFT ensures identical batches
- 2f+1 nodes must agree before generation commits

### 3. Progress Streamer
- Real-time SSE/WebSocket event broadcasting
- Events: generation, best_fitness, qd_coverage, consensus_status, flux_results
- Harness UI consumes these for visual progress

### 4. Validation Gates
- FLUX constraints as build gates
- Hard gates (fail build) vs soft gates (warn)
- Pre-built gates: exact_arithmetic, holonomic, spectral_real, robustness

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    HARNESS (OpenConstruct)                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Define    │  │  Orchestrate │  │     UI      │     │
│  │  Manifests  │  │   DAG/Flow   │  │  Progress   │     │
│  └──────┬──────┘  └──────┬──────┘  └──────▲──────┘     │
└─────────┼────────────────┼────────────────┼──────────────┘
          │                │                │
          ▼                ▼                │
┌─────────────────────────────────────────────────────────┐
│              SUNSET-ECOSYSTEM BRIDGE                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Harness   │  │    Build    │  │   Progress  │     │
│  │   Adapter   │  │ Coordinator │  │  Streamer   │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐     │
│  │  Breeder    │  │    BFT      │  │   FLUX      │     │
│  │  Factory    │  │  Consensus  │  │   Gates     │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
└─────────┼────────────────┼────────────────┼──────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────┐
│              SUNSET-ECOSYSTEM CORE                      │
│  PythagoreanBreeder  SpectralBreeder  AdversarialArena  │
│  FleetBreederConsensus  FLUX VM  OperationalTrap        │
│  MeshVectorTables  MetronomeBridge  AgentIdentity       │
└─────────────────────────────────────────────────────────┘
```

## Implementation Plan

1. `fleet/openconstruct_bridge.py` — Core integration module
2. `tests/test_openconstruct_bridge.py` — 30+ tests covering all integration points
3. `docs/INTEGRATION_MAP.md` — Updated with harness integration docs

## Key Design Decisions

1. **Protocol Agnostic**: Adapter pattern — works with any harness API
2. **Event-Driven**: Progress streaming via callbacks + SSE
3. **Fault Tolerant**: BFT consensus handles node failures
4. **Type Safe**: Dataclass manifests with validation
5. **Extensible**: New breeder types register via factory pattern
