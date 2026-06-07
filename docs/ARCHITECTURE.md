# Architecture Guide — Sunset Ecosystem

> **What this is:** A deep dive into how the system works, not just how to use it. For engineers who want to understand the design decisions.

## The Core Idea

The sunset ecosystem is built on a single principle: **local decisions produce global behavior**.

There is no central coordinator that decides what every agent does. Instead, each component has simple rules, and when combined, they produce complex fleet-wide behavior. This is called **emergence** — and it is the only way to build systems that scale beyond your ability to understand them.

## The Three Layers

```
┌─────────────────────────────────────────────┐
│  EMERGENT APPLICATIONS                      │
│  FleetMemory, CognitiveCache, VectorSwarm    │
│  LevelRunner, Pincher, xMind Bridge         │
├─────────────────────────────────────────────┤
│  MESH PRIMITIVES                            │
│  MeshVectorTable, TieredMeshStorage, HNSW  │
│  SceneTracker, MeshGrouping, MeshWAL       │
├─────────────────────────────────────────────┤
│  BRIDGES                                    │
│  Quanta VDB, xlang Agent, caslang, xMind    │
└─────────────────────────────────────────────┘
```

## Layer 1: Bridges

Bridges are adapters. They take external systems (C++ HNSW, event-driven runtime, sandboxed execution) and make them speak the same language as our mesh primitives.

### Quanta VDB Bridge

**Problem:** The C++ HNSW library is fast but has no Python API, no crash recovery, and no fleet synchronization.

**Solution:** A Python wrapper that:
1. Manages the C++ lifecycle (compile, load, teardown)
2. Adds a SQLite-backed manifest for crash recovery
3. Uses CRDT principles for conflict-free fleet sync

**Key insight:** The C++ layer does one thing well (approximate nearest neighbors). Everything else (persistence, sync, error handling) lives in Python where it is easy to reason about.

### xlang Agent Bridge

**Problem:** xlang is a distributed, no-GIL runtime for event-driven systems. But its Python bindings are minimal.

**Solution:** A bridge that translates our AgentFlow graphs into xlang YAML blueprints, then executes them locally and syncs state back to the mesh.

**Key insight:** We do not try to run xlang inside Python. We generate configuration files, spawn xlang processes, and consume their output. This is the Unix philosophy: small tools, connected by data.

### caslang Bridge

**Problem:** We need deterministic, sandboxed execution for agent AI decisions, but Python is not sandboxed and is slow for math-heavy operations.

**Solution:** A JSONL parser that feeds into a simple execution engine. Each line is validated before execution, and the entire batch can be rolled back if any line fails.

**Key insight:** Sandboxing does not require a separate process. It requires validation before execution. The "sandbox" is the parser, not a container.

## Layer 2: Mesh Primitives

These are the building blocks. They are designed to be composed, not used alone.

### MeshVectorTable

**What it is:** A vector store with CRDT merge semantics.

**Why CRDTs?** Because in a distributed system, every node has a different view of the data. CRDTs guarantee that if two nodes merge their views, they will converge to the same state without coordination. This is the difference between a system that works at 10 nodes and one that works at 10,000.

**The merge rule:**
```python
def _crdt_winner(a, b):
    if a.timestamp > b.timestamp:
        return a
    elif b.timestamp > a.timestamp:
        return b
    else:
        # Tie-breaker: deterministic hash
        return a if hash(a.signature) > hash(b.signature) else b
```

**Why this works:**
- Timestamps are monotonic (mostly). If node A has a newer entry than node B, the newer entry wins.
- If timestamps are equal (rare), we use a deterministic hash tie-breaker. Both nodes will make the same decision.
- There is no "conflict resolution" step. The merge is the resolution.

**The vector store:**
- In-memory dict of `agent_id -> VectorTableEntry`
- Each entry is a frozen dataclass (immutable, hashable, fast)
- Vectors are `np.ndarray` (float32 for memory efficiency)
- Fitness scores enable quality-aware queries

**Why not a database?** Because for fleet-scale operations, a dict is faster than SQLite, and the CRDT properties are easier to implement when you control the data structure. The database (SQLite) is used for the warm tier and the WAL, not the hot tier.

### TieredMeshStorage

**What it is:** Hot/warm/cold storage with automatic promotion and demotion.

**Why tiers?** Because memory is finite, and not all data is equally valuable. Hot data is accessed frequently. Warm data is occasionally useful. Cold data is kept for compliance or historical analysis.

**The policy:**
```python
@dataclass
class TierConfig:
    hot_max_entries: int = 10_000
    hot_max_age_seconds: float = 86400.0
    hot_min_fitness: float = 0.5
    warm_max_entries: int = 100_000
    demotion_access_threshold: int = 3
    demotion_thermal_threshold: float = 0.8
```

**How it works:**
1. **Hot tier:** The base `MeshVectorTable`. All reads go here first. Entries are promoted from warm when accessed.
2. **Warm tier:** SQLite-backed. Entries are demoted from hot when they age out or are not accessed enough.
3. **Cold tier:** Compressed JSONL archives. Demoted from warm after a long period of inactivity. Can be loaded back if needed.

**Promotion rule:** When an entry is accessed `demotion_access_threshold` times, it moves to hot.

**Demotion rule:** When an entry's age exceeds `hot_max_age_seconds` or its fitness drops below `hot_min_fitness`, it moves to warm.

**Key insight:** The tiers are not a cache. They are a **storage policy**. The system is designed to make the right data available at the right tier without human intervention.

### HNSW Mesh Table

**What it is:** Approximate nearest neighbor search using HNSW (Hierarchical Navigable Small World).

**Why HNSW?** Because exact nearest neighbor search is O(n), which is too slow for fleets with millions of agents. HNSW is O(log n) with high recall (>95% at reasonable parameters).

**The fallback:** If the C++ HNSW library is not compiled, the system falls back to brute-force O(n) search. This is critical for deployment: the system works immediately, and gets faster when you compile the C++ extension.

**How it works:**
1. Build an HNSW index from the hot tier entries.
2. Auto-rebuild when the index is stale (entries added/removed beyond a threshold).
3. Query returns approximate nearest neighbors with distance scores.
4. Novelty and density scores help identify sparse regions of the vector space.

**Key insight:** HNSW is not a replacement for the mesh table. It is an **overlay** on top of it. The mesh table is the source of truth. The HNSW index is a query accelerator.

### SceneTracker

**What it is:** Query pattern tracking and scene detection.

**Why track queries?** Because if you know what is being queried, you can predict what will be queried next. This is the basis of the CognitiveCache's predictive preloading.

**How it works:**
1. Every query is logged as a `QueryPattern` (type, filter, result size, latency, timestamp).
2. A histogram tracks how often each pattern occurs.
3. Co-occurrence tracking: if agent A is queried after agent B, a link is recorded.
4. Scenes are detected when query patterns change (using a simple change-point detector).

**Key insight:** The scene tracker does not need to understand the meaning of the queries. It only needs to see the patterns. The patterns are the signal. The meaning is irrelevant for prediction.

### MeshGrouping

**What it is:** Clustering for mesh entries.

**Why cluster?** Because a fleet of 10,000 agents is not 10,000 individuals. It is 50 clusters of 200 agents each. Clustering helps you understand the structure of the fleet.

**Algorithms:**
- **K-means:** Lloyd's algorithm with cosine similarity. Fast, but needs k.
- **Hierarchical:** Agglomerative clustering. No k needed, but O(n²).
- **DBSCAN:** Density-based. Finds arbitrary shapes, handles noise.
- **Single-pass:** Online clustering. For streaming data where you cannot see the whole dataset.

**Key insight:** Clustering is not just for visualization. It is for **routing**. If you know which cluster an agent belongs to, you can route queries to the right shard or node.

### MeshWAL

**What it is:** Write-ahead log for crash recovery.

**Why WAL?** Because in-memory state is lost on restart. The WAL records every insert, delete, and merge so the state can be replayed.

**How it works:**
1. Every operation is appended to a WAL file with a CRC32 checksum.
2. Checkpoints are written periodically, truncating old WAL files.
3. On restart, the WAL is replayed from the last checkpoint.
4. Corrupted entries are detected (CRC mismatch) and skipped.

**Key insight:** The WAL is not a database. It is a **journal**. It is optimized for append-only writes, not random access. The SQLite warm tier handles the structured queries. The WAL handles the durability.

## Layer 3: Emergent Applications

These are compositions of the primitives. They are where the magic happens.

### FleetMemory

**Composition:** MeshVectorTable + TieredMeshStorage + time sharding.

**What it does:** Long-term memory for fleets. Each day is a separate shard. Old shards are dropped (LRU) or archived (cold tier).

**Why it emerges:** A single `MeshVectorTable` would grow forever. Time sharding bounds the memory. Tiered storage moves old data to warm/cold. The result is a system that remembers the last 30 days without unbounded growth.

**The shard lifecycle:**
1. Create shard for current day.
2. Write all entries to current shard (O(1) — no cross-shard lookup).
3. Query spans shards from `start_time` to `end_time` (O(shards) — typically 1-30).
4. When `max_shards` exceeded, oldest shard is closed and archived.

**Key insight:** The shard is the unit of garbage collection. Not the entry. Not the table. The shard. This makes memory management predictable and fast.

### CognitiveCache

**Composition:** TieredMeshStorage + SceneTracker + PredictionEngine.

**What it does:** A cache that learns from query patterns and preloads predicted entries before they are needed.

**Why it emerges:** SceneTracker detects patterns. TieredMeshStorage handles hot/warm/cold. CognitiveCache connects them: when it sees a pattern, it promotes the predicted entries to hot tier.

**The prediction engine:**
1. **Co-occurrence predictions:** If agent A is queried, and the tracker shows agent B is often queried next, preload B.
2. **Hot query predictions:** Frequently queried agents are kept hot.
3. **Scene-based predictions:** If the current scene is "breeding", preload breeding-related agents.

**Feedback loop:** Every prediction is tracked (hit/miss). If predictions are wrong, the engine adjusts its confidence thresholds.

**Key insight:** The cache is not just a data structure. It is a **learning system**. It gets better the more you use it.

### VectorSwarm

**Composition:** SwarmRouter + MeshVectorTable + consensus ranking.

**What it does:** Distributed search across multiple nodes. Queries are fanned out, results are deduplicated and ranked by consensus.

**Why it emerges:** A single node cannot hold all agents. Sharding is necessary. But querying across shards requires coordination. VectorSwarm handles the coordination.

**The consensus algorithm:**
1. Query is sent to all relevant nodes (or a quorum subset).
2. Each node returns its local top-k results.
3. Results are deduplicated by agent_id.
4. Each result is scored by how many nodes returned it (consensus).
5. Final ranking is by consensus score, then distance, then fitness.

**Key insight:** Consensus ranking is not just for reliability. It is for **quality**. A result that appears on multiple nodes is more likely to be relevant than a result that only appears on one.

## Design Decisions

### Why CRDT over Raft/Paxos?

Raft and Paxos require a leader and synchronous communication. They are fast but fragile (leader failure = downtime). CRDTs are slower in terms of convergence time but never fail. For a fleet where nodes come and go unpredictably, CRDTs are the right choice.

### Why HNSW as optional?

Because C++ compilation is a deployment barrier. The system must work out of the box, and get faster when compiled. This is the "progressive enhancement" principle.

### Why three-tier storage?

Because two-tier (hot/cold) is not enough. Hot is for active data. Warm is for recent data. Cold is for historical data. The distinction between warm and cold is about **query cost**: warm is SQLite (fast), cold is compressed (slow but compact).

### Why frozen dataclasses?

Because `VectorTableEntry` is the core data structure. It is created, read, and passed around millions of times. Frozen dataclasses are:
- Immutable (no accidental mutation)
- Hashable (can be used in sets and dicts)
- Fast (low overhead compared to dicts)
- Self-documenting (fields are explicit)

### Why numpy arrays?

Because vector math is the bottleneck. `np.dot(a, b)` is 100x faster than a pure Python loop. And float32 is half the memory of float64, which matters at fleet scale.

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Insert (hot) | O(1) | Dict insertion |
| Query by ID | O(1) | Dict lookup |
| Similarity (exact) | O(n) | Linear scan over all entries |
| Similarity (HNSW) | O(log n) | Approximate, >95% recall |
| Tiered insert | O(1) | Hot tier, or SQLite insert |
| WAL append | O(1) | File append |
| WAL replay | O(n) | Read all entries |
| FleetMemory write | O(1) | Shard lookup + insert |
| FleetMemory query | O(s) | s = number of shards in range |
| VectorSwarm KNN | O(n_nodes × log n) | Parallel across nodes |
| CognitiveCache predict | O(q) | q = number of query patterns |

## Failure Modes

### What happens if the HNSW index is corrupt?

The system auto-rebuilds on the next query. The base table is the source of truth. The HNSW index is disposable.

### What happens if SQLite is corrupt?

The warm tier is lost. The hot tier still works. Entries can be re-inserted from the WAL or from other nodes via sync.

### What happens if a node goes offline?

Other nodes continue operating. The offline node's data is stale but not lost. When it comes back, it merges with the fleet via CRDT.

### What happens if two nodes insert the same agent with different data?

The CRDT merge resolves it. Newer timestamp wins. If timestamps are equal, the hash tie-breaker ensures both nodes make the same decision.

### What happens if the WAL is full?

Checkpoints are written automatically. Old WAL files are truncated. The system never runs out of disk space from WAL growth.

## Extending the System

### Adding a new bridge

1. Create a Python wrapper for the external system.
2. Implement `insert()` and `query()` methods that return `VectorTableEntry`.
3. Add a sync method that produces a CRDT-compatible payload.
4. Write tests in `tests/test_your_bridge.py`.

### Adding a new emergent application

1. Identify the primitives you need (e.g., MeshVectorTable + SceneTracker).
2. Compose them in a new class.
3. Add a feedback loop if the application should learn (e.g., CognitiveCache).
4. Write tests in `tests/test_your_app.py`.

### Adding a new mesh primitive

1. Implement the core data structure.
2. Ensure it is immutable or thread-safe.
3. Add CRDT merge semantics if it will be distributed.
4. Write tests. Lots of tests.

## Further Reading

- **DEVELOPER_GUIDE.md** — How to use the system (practical)
- **QUANTA_VDB_DEEP_DIVE.md** — C++ internals and compilation
- **FLUX_OPCODE_ALIGNMENT.md** — VM integration (legacy)
- **FLEET_BFT_QD.md** — Byzantine consensus and quality diversity

---

*This document is living. If you find a design decision that no longer makes sense, update it. The architecture should evolve with the code.*
