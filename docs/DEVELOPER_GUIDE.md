# Sunset Ecosystem — Developer Guide

> A fleet-scale mesh vector database with emergent applications for distributed agent systems.
>
> **160 tests passing. 23 modules. 6 bridges.**

## Table of Contents

1. [Philosophy](#philosophy)
2. [Core Architecture](#core-architecture)
3. [Mesh Vector Tables](#mesh-vector-tables)
4. [Storage Layers](#storage-layers)
5. [Emergent Applications](#emergent-applications)
6. [xlang-foundation Integration](#xlang-foundation-integration)
7. [Development Patterns](#development-patterns)
8. [Testing Strategy](#testing-strategy)

---

## Philosophy

The sunset ecosystem is built on a simple premise: **agents are vectors, and vectors deserve a database.**

Most vector databases are built for embeddings — static, batch-loaded, query-and-forget. Our agents are alive: they mutate, they migrate between nodes, they breed, they die. The database must reflect this reality.

Design principles:
- **No GIL bottlenecks**: Every layer is designed to bypass Python's GIL where possible (C++ extensions, HNSW, SQLite)
- **CRDT by default**: Conflicts are resolved by physics, not by coordination. Timestamp + signature hash = winner.
- **Fail gracefully**: No C++ dependency is mandatory. Every acceleration has a Python fallback.
- **Emergent over explicit**: Applications like FleetMemory arise from composing primitives, not from hardcoding use cases.

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Emergent Applications                 │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌──────────┐ │
│  │FleetMemory│ │LevelRunner│ │  Pincher   │ │VectorSwarm│ │
│  └────┬─────┘ └────┬─────┘ └─────┬──────┘ └────┬─────┘ │
│       │            │             │              │        │
│  ┌────┴────────────┴─────────────┴──────────────┴─────┐  │
│  │              Tiered Storage + HNSW                │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │  │
│  │  │  Hot Tier   │  │  Warm Tier   │  │ Cold Tier │ │  │
│  │  │  (MeshTable)│  │  (SQLite)   │  │(Compressed)│ │  │
│  │  └─────────────┘  └─────────────┘  └──────────┘ │  │
│  └──────────────────────────────────────────────────┘  │
│                    Mesh Vector Tables                    │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐   │
│  │  Grouping   │  │  SceneTracker│  │  WAL + CRDT  │   │
│  │  (Cluster)  │  │  (Patterns)  │  │  (Recovery)  │   │
│  └─────────────┘  └─────────────┘  └──────────────┘   │
│                    xlang-foundation Bridges              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐   │
│  │Quanta VDB   │  │  xlang Agent  │  │  caslang     │   │
│  │  (HNSW)     │  │  (Flow Engine)│  │  (Sandbox)   │   │
│  └─────────────┘  └─────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Mesh Vector Tables

The foundation is `MeshVectorTable` — a CRDT-backed vector store for agent states.

### What Makes It Different

| Feature | Standard Vector DB | MeshVectorTable |
|---------|-------------------|-----------------|
| Conflict resolution | Last-write-wins | CRDT with physics (timestamp + signature) |
| Multi-node sync | Centralized | Merkle-tree payload + async merge |
| Fitness queries | None | Built-in index for evolutionary selection |
| Diversity queries | None | Spatial spread + cluster silhouette |
| Identity verification | None | Optional Ed25519 signatures per entry |
| GIL bypass | Rare | Native HNSW, SQLite, C++ bindings |

### Entry Model

```python
@dataclass(frozen=True)
class VectorTableEntry:
    agent_id: str
    vector: np.ndarray          # Agent state vector
    timestamp: float            # Physics time (monotonic)
    node_id: str                # Origin node
    generation: int             # Evolutionary generation
    fitness: float              # [0.0, 1.0]
    capability_mask: int = 0  # Bitfield for skills
    thermal_pressure: float = 0.0
    signature: str = ""         # Ed25519 or hash
```

### CRDT Resolution

When two nodes merge conflicting entries for the same `agent_id`:

1. **Higher timestamp wins** — newer state is more correct
2. **Tie-break by signature hash** — deterministic, no central clock needed
3. **Vector distance threshold** — if vectors diverge > 0.5 cosine, both are kept (divergent evolution)

This is the `_crdt_winner` method. It runs on every merge, no coordination required.

### Query Patterns

```python
# Exact lookup by agent ID
entry = table.query("agent_42")

# Top-k by fitness (for breeding selection)
parents = table.query_by_fitness(k=5, min_fitness=0.8)

# Diversity spread (avoid inbreeding)
spread = table.query_by_diversity(k=10)

# Similarity search (with HNSW if available)
neighbors = table.query_similarity_sorted(query_vector, k=5)

# Population summary
summary = table.get_population_summary()
# Returns: size, mean/std fitness, mean/std vector, spatial spread
```

### When to Use What

| Query | Method | Use Case |
|-------|--------|----------|
| "Who is agent_42?" | `query()` | Agent state recall |
| "Best parents for next generation" | `query_by_fitness()` | Evolutionary breeding |
| "Most diverse set" | `query_by_diversity()` | Avoid convergence |
| "Agents similar to this" | `query_similarity_sorted()` | Pattern matching |
| "Fleet health overview" | `get_population_summary()` | Monitoring |

---

## Storage Layers

### Three-Tier Architecture

**Hot Tier**: `MeshVectorTable` (in-memory, CRDT, O(1) ID lookup)
- All recent entries live here
- No serialization overhead for queries
- Automatic promotion from warm tier on access

**Warm Tier**: SQLite (`TieredMeshStorage`)
- Batched writes, indexed by agent_id and fitness
- WAL mode for crash recovery
- Promotes to hot tier after 3 accesses

**Cold Tier**: Compressed JSONL archives
- Daily snapshots, gzip-compressed
- Demoted after 24h of no access
- Background maintenance thread

### HNSW Hybrid Index

The `HnswMeshTable` overlays an HNSW index on the base `MeshVectorTable`:

```python
# Automatic: uses hnswlib if available, falls back to brute-force
hnsw = HnswMeshTable(base_table, rebuild_threshold=0.2)

# Rebuild is lazy (only when >20% of entries are new)
# Rebuild is background (threaded, doesn't block queries)

# Stats show both indexes
stats = hnsw.stats()
# {
#   "entry_count": 10000,
#   "hnsw_index_count": 10000,
#   "hnsw_available": True,
#   "last_rebuild": 1717800000.0,
#   "novelty": 0.15,
#   "density": 0.85
# }
```

### SceneTracker — Query Pattern Optimization

`SceneTracker` watches query patterns and learns:

- **Hot queries**: Frequently accessed agent IDs → promote to hot tier
- **Co-occurrence**: If agent A is always queried after agent B → preload B when A is accessed
- **Scene detection**: Bursts of related queries indicate a "scene" (e.g., "breeding batch", "health check")

```python
tracker = SceneTracker(table, strategy=CacheStrategy(
    hot_threshold_accesses=3,
    scene_timeout_seconds=60.0,
))

# Every query is tracked
tracker.track_query("by_id", "none", result_size=1, latency_ms=10.0,
                   query_params={"agent_id": "agent_42"})

# Get recommendations for preloading
recs = tracker.get_cache_recommendations()
# {"agent_42": ["agent_43", "agent_44"]}  # co-occurrence based
```

---

### CognitiveCache

Adaptive cache that learns from query patterns.

```python
base = MeshVectorTable(table_id="cache")
storage = TieredMeshStorage(base_table=base)
tracker = SceneTracker(base, strategy=CacheStrategy(
    hot_threshold_accesses=3,
    scene_timeout_seconds=60.0,
))
cache = CognitiveCache(storage, tracker)

# Queries are automatically tracked
entry = cache.query("agent_42")

# Predictive preloading: co-occurring entries are preloaded
# When agent_42 is queried, agent_43 is preloaded (learned pattern)
```

**Why it emerges**: SceneTracker detects patterns. TieredMeshStorage handles hot/warm/cold. CognitiveCache connects them so the cache learns what to keep hot.

---

## Emergent Applications

These are not standalone services. They are **compositions** of the primitives above.

### FleetMemory

Time-partitioned memory for long-lived fleets.

```python
memory = FleetMemory(
    shard_duration=86400.0,  # 1 day per shard
    max_shards=30,           # Keep 30 days
)

# Write is O(1) — routed to current shard
memory.write("agent_42", vector, timestamp=now)

# Temporal query spans shards automatically
results = memory.query(
    start_time=now - 3600,
    end_time=now,
    filter_fn=lambda e: e.fitness > 0.8,
)

# LRU eviction: oldest shard is dropped when max_shards exceeded
```

**Why it emerges**: A single `MeshVectorTable` would grow unbounded. Time sharding + LRU + tiered storage = sustainable long-term memory.

### VectorSwarm

Distributed search across multiple nodes.

```python
router = SwarmRouter()
router.register_node("node1", ["shard_1a", "shard_1b"], table1)
router.register_node("node2", ["shard_2a"], table2)

swarm = VectorSwarm(router)

# Query routes to all nodes, results are deduplicated and ranked
results = swarm.query_knn(
    query_vector,
    k=10,
    consistency="quorum",  # Only need 50%+1 nodes
)

# Consensus ranking: nodes vote on results
ranked = swarm.consensus_rank(results, query_vector)
```

**Why it emerges**: Breeding pools span nodes. A single-node query is insufficient for fleet-wide parent selection.

### LevelRunner

Simulation/game level execution engine.

```python
# Combines xlang events, Quanta VDB state, caslang deterministic AI
runner = LevelRunner()
runner.register_system("physics", PhysicsSystem())
runner.register_system("ai", CaslangAI(caslang_script))

# Each tick: event processing → state update → AI decisions
runner.tick(dt=0.016)
```

**Why it emerges**: xlang provides event-driven execution, caslang provides sandboxed AI, Quanta VDB tracks spatial state. Together they form a simulation engine.

### Pincher

Selective data extraction with pattern matching.

```python
# Extract fields from unstructured records using regex
card = Pincher.Card()
card.add_query("temperature", "sensor", r"TEMP:\s*(\d+\.?\d*)", transform="float")

# Run against a data source
results = pincher.extract([{"raw": "TEMP: 23.5"}], card)
# {"temperature": 23.5}
```

**Why it emerges**: Fleet agents produce unstructured logs. Pincher converts them to structured vectors for the database.

---

## xlang-foundation Integration

Four Apache-2.0 repositories, four complementary roles:

### Quanta VDB (C++ Streaming Vector Database)

- **Role**: High-performance vector storage with GIL bypass
- **Integration**: `quanta_vdb_bridge.py` — CRDT manifest + SQLite shadow + HNSW ANN
- **Key insight**: Quanta's `PartitionedVdb` uses `VdbBucket` per partition. Our `FleetMemory` shards map directly to this.

### xlang (Distributed Runtime)

- **Role**: Event-driven agent execution, no GIL
- **Integration**: `xlang_agent_bridge.py` — AgentFlow graph → YAML blueprint
- **Key insight**: xlang's `X::Runtime` is the execution engine. Our `LevelRunner` is the orchestration layer above it.

### xMind (Agent Flow Orchestration)

- **Role**: Visual agent workflow design
- **Integration**: AgentFlow YAML blueprints from our `LevelRunner` configs
- **Key insight**: xMind's `AgentFlow` is the design-time tool. Our bridge is the runtime execution.

### caslang (Constrained Execution)

- **Role**: Sandbox deterministic AI execution
- **Integration**: `caslang_executor.py` — JSONL command stream + validation
- **Key insight**: caslang's `Guard` prevents runtime violations. We use it for AI system safety in `LevelRunner`.

### When to Use What

| Component | Use When | Don't Use When |
|-----------|----------|----------------|
| Quanta VDB | Need C++ speed, GIL bypass | Can't compile C++ extensions |
| xlang | Event-driven, distributed agents | Simple single-threaded logic |
| caslang | Untrusted AI code, deterministic replay | Trusted, high-performance code |
| MeshVectorTable | CRDT, fleet sync, fitness queries | Pure speed, no distribution |
| HNSW Mesh | Similarity search at scale | Small datasets (<1000) |
| Tiered Storage | Long-term memory | Short-lived sessions |

---

## Development Patterns

### Adding a New Module

1. **Write the module** in `swarm/` or `fleet/`
2. **Write tests** in `tests/test_<module>.py`
3. **Run tests**: `pytest tests/test_<module>.py -v`
4. **Fix failures**: Iterate until green
5. **Commit**: `git add -A && git commit -m "..."`
6. **Push**: `git push origin main`

### Testing Philosophy

Every module has comprehensive tests. Every test uses `tempfile.TemporaryDirectory` for isolation. Every failure is fixed before push.

```python
# Pattern: Test isolation + fallback paths
@pytest.fixture
def base_table():
    return MeshVectorTable(table_id="test")

class TestFeature:
    def test_happy_path(self, base_table):
        # ...

    def test_empty_case(self, base_table):
        # ...

    def test_fallback_without_hnsw(self):
        # ...
```

### Fallback Strategy

```python
try:
    import hnswlib
    _HNSWLIB_AVAILABLE = True
except ImportError:
    _HNSWLIB_AVAILABLE = False

# Use HNSW if available, else brute-force
if _HNSWLIB_AVAILABLE:
    # ... fast path
else:
    # ... O(n) fallback, still correct
```

### Signature Verification

Production fleets use Ed25519 signatures. Tests use `skip_verify=True` or signatures >= 8 characters.

```python
# Production
entry = VectorTableEntry(..., signature=ed25519_sign(...))
table.insert_signed(entry)

# Test
entry = VectorTableEntry(..., signature="test_sig_123")
table.insert(entry, skip_verify=True)
```

---

## Testing Strategy

### Current Test Inventory (160 tests)

| Module | Tests | Key Coverage |
|--------|-------|--------------|
| mesh_vector_tables | 40+ | CRDT, fitness, diversity, sync |
| mesh_wal | 13 | Crash recovery, CRC, checkpoint |
| mesh_grouping | 12 | K-means, DBSCAN, outliers, diversity |
| scene_tracker | 11 | Pattern detection, co-occurrence, cache |
| hnsw_mesh_table | 11 | HNSW build, rebuild, fallback |
| tiered_mesh_storage | 7 | Hot/warm/cold, promotion, demotion |
| fleet_memory | 12 | Time sharding, LRU, temporal queries |
| vector_swarm | 12 | Distributed KNN, consensus, dedup |
| cognitive_cache | 13 | Predictive preloading, accuracy, co-occurrence |
| quanta_vdb_bridge | 12 | CRDT manifest, SQLite shadow, HNSW |
| caslang_executor | 12 | JSONL parse, sandbox, rollback |
| level_runner | 11 | Systems, events, bounds, stats |
| pincher | 10 | Regex, concat, transforms, batch |
| xlang_agent_bridge | 14 | Blueprint, execution, sync, lifecycle |

### Running the Suite

```bash
# Full suite
python3 -m pytest tests/ -v

# Specific module
python3 -m pytest tests/test_fleet_memory.py -v

# With coverage
python3 -m pytest tests/ --cov=swarm --cov=fleet
```

---

## Design Decisions Documented

### Why CRDT over Raft/Paxos?

Raft requires a leader. A leader is a single point of failure and a GIL bottleneck. CRDT lets every node merge independently. The cost is occasional duplicate entries (resolved by vector distance threshold).

### Why HNSW is optional?

C++ compilation is a deployment barrier. `pip install hnswlib` often fails on exotic architectures. The fallback is brute-force O(n) search — slower but correct. This is the "fail gracefully" principle.

### Why three tiers?

- Hot: Speed for active queries
- Warm: Persistence for recovery without full rebuild
- Cold: Cost for historical data

The thresholds are tunable: `hot_threshold_accesses`, `shard_duration`, `max_shards`.

### Why time-shard in FleetMemory?

A single table with 1M entries has O(n) brute-force queries. With 30 daily shards of ~33k entries, queries span only relevant shards. The LRU drops cold shards, preventing unbounded growth.

### Why consensus ranking in VectorSwarm?

Byzantine nodes can return wrong results. Consensus ranking aggregates votes across nodes. An entry returned by 3/4 nodes is more trustworthy than one returned by 1/4.

---

## Glossary

| Term | Definition |
|------|------------|
| **CRDT** | Conflict-free Replicated Data Type. Merges without coordination. |
| **HNSW** | Hierarchical Navigable Small World. Approximate nearest neighbor index. |
| **WAL** | Write-Ahead Log. Crash recovery journal. |
| **Merlke tree** | Hash tree for efficient sync payload verification. |
| **Fitness** | [0.0, 1.0] scalar for evolutionary quality. |
| **Diversity** | Spatial spread metric to avoid convergence. |
| **Scene** | Burst of related queries indicating a use case. |
| **Shard** | Partition of data (time, fitness, or hash-based). |
| **Tier** | Hot (memory), warm (SQLite), cold (compressed). |
| **Bridge** | Python adapter to an external system (xlang, Quanta, etc.). |

---

## Contributing

This is a living system. The architecture diagram is redrawn every time a new module proves useful. The test count is the only metric that matters.

> **"The fleet is vast. Build the bridges."**

---

*Last updated: 2026-06-08*
*Commit: 102fdc0*
*Tests: 147/147 passing*
