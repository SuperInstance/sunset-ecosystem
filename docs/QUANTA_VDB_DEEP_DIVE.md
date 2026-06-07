# Quanta VDB + Mesh Vector Tables — Deep Dive Analysis

## Current State

### Quanta VDB (C++ Implementation)
**Architecture:**
- **HNSW Indexing**: Hierarchical Navigable Small World graphs for O(log n) approximate nearest neighbor search
- **Time Partitioning**: Automatic partition by timestamp (hourly/daily/weekly) + custom tags
- **Async Ingestion**: Background thread with queue for non-blocking adds
- **WAL (Write-Ahead Log)**: Crash recovery with micro-batch merging
- **Scene Tracker**: Query pattern tracking for cache optimization
- **Grouping API**: Centroid-based clustering for semantic grouping
- **TTL Eviction**: Automatic cleanup of old data
- **GIL Bypass**: xlang runtime integration for Python-free C++ execution
- **Binary Save/Load**: Atomic file operations with .new/.old rollback

**Key Classes:**
- `VectorDatabase` — Base class with ID→text mapping, timestamps, parameters
- `PartitionedVdb` — Time+tag partitioned HNSW with WAL, async ingestion, maintenance
- `VdbBucket` — Individual partition storage (HNSW + metadata)
- `SceneTracker` — Query pattern tracking and cache preloading

### Mesh Vector Tables (Python Implementation)
**Architecture:**
- **CRDT Entries**: Immutable, signed, hashable entries with agent_id, vector, timestamp, fitness
- **Fitness Index**: Sorted list for top-k fitness queries (rebuilt on demand)
- **Node Breakdown**: Per-node population counts
- **FleetVectorIndex**: Multi-generation table management with skill views
- **Sync Payloads**: zlib-compressed JSON for gossip transport
- **Signature Verification**: Ed25519 or SHA-256 fallback

**Gaps vs Quanta:**
1. ❌ **No ANN Index**: O(n) brute force search — scales poorly past 10K entries
2. ❌ **No Time Partitioning**: All entries in one table — no temporal queries
3. ❌ **No Async Ingestion**: Every insert blocks the caller
4. ❌ **No TTL/Eviction**: Memory grows unbounded
5. ❌ **No WAL**: No crash recovery
6. ❌ **No Grouping**: No semantic clustering
7. ❌ **No Scene Tracking**: No query pattern optimization
8. ❌ **No Tiered Storage**: All in memory, no spill-to-disk

## Improvement Strategy

### Phase 1: Hybrid Index (Immediate)
Add HNSW-like approximate indexing to mesh_vector_tables.py without requiring C++ compilation. Use `hnswlib` (pure Python available) or implement a simple LSH (Locality Sensitive Hashing) fallback.

### Phase 2: Tiered Storage (High Impact)
Implement hot/warm/cold tiers:
- **Hot**: Recent entries in memory with ANN index
- **Warm**: Older entries in SQLite with B-tree index
- **Cold**: Archived entries in compressed files

### Phase 3: Async Pipeline (Scalability)
Add async ingestion queue with background batch processing, similar to Quanta's approach.

### Phase 4: Emergent Applications
Build new fleet capabilities on top of the improved infrastructure.

## Emergent Application Possibilities

### 1. FleetMemory — Persistent Time-Partitioned Cognition
A cross-node memory system that uses Quanta's time partitioning + mesh CRDT sync. Each fleet node maintains a time-partitioned VDB of agent experiences, decisions, and outcomes. The mesh syncs deltas across nodes. This creates a **collective memory** that grows over time and automatically forgets old, low-value memories via TTL.

**Use case**: After 6 months of fleet operation, query "what did we learn about PyTorch model deployment in March?" and get relevant agent experiences from across the fleet.

### 2. VectorSwarm — Real-Time Fleet-Wide Vector Search
A distributed search layer where every node runs a local ANN index (HNSW) and queries are federated across the mesh. The mesh_vector_tables CRDT sync keeps indices loosely consistent. Query results are merged with conflict resolution (higher fitness wins).

**Use case**: "Find me the 10 most diverse agent configurations that have successfully handled Kubernetes pod eviction scenarios" — search across all nodes simultaneously.

### 3. CognitiveCache — Tiered Agent State Cache
A cache system that uses tiered storage (hot/warm/cold) with automatic promotion/demotion based on access patterns (tracked by SceneTracker). Hot agent states are in memory for fast breeding queries. Warm states are in SQLite. Cold states are archived but can be "thawed" on demand.

**Use case**: A breeding daemon that runs overnight needs access to 10,000 agent configurations, but only 100 are actively used. The cache keeps the 100 hot and the rest in warm/cold tiers.

### 4. PatternMine — Automatic Pattern Discovery
Uses the Grouping API (centroid clustering) to automatically discover patterns in agent vector trajectories over time. Identifies "archetypes" — clusters of agents that share similar capabilities and performance characteristics.

**Use case**: After running 1000 breeding cycles, PatternMine discovers 7 distinct agent archetypes (e.g., "fast-but-fragile", "slow-but-reliable", "generalist", "specialist-A", etc.). These archetypes become named categories in the fleet taxonomy.

### 5. TrajectoryStore — Agent Evolution Tracking
Stores the full trajectory of each agent's vector over time (generation 0 → generation 1 → ...). Enables lineage queries: "Show me the ancestry of this high-performing agent" and "What mutations led to the biggest fitness improvements?"

**Use case**: A researcher wants to understand which breeding operations produced the best agents. TrajectoryStore provides full lineage with vector diffs at each generation.

## Implementation Priority

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| P0 | HNSW ANN Index (hnswlib) | Medium | Critical — enables scaling past 10K |
| P0 | Tiered Storage (hot/warm/cold) | Medium | High — prevents memory exhaustion |
| P1 | Async Ingestion Queue | Low | Medium — improves throughput |
| P1 | TTL / Auto-Eviction | Low | Medium — prevents unbounded growth |
| P1 | WAL for Crash Recovery | Medium | Medium — durability |
| P2 | Scene Tracker (query patterns) | Medium | Low — optimization |
| P2 | Grouping API (clustering) | High | Medium — pattern discovery |
| P3 | FleetMemory application | High | High — emergent capability |
| P3 | PatternMine application | High | High — emergent capability |

## Next Steps
1. Implement HNSW hybrid index in mesh_vector_tables.py
2. Implement tiered storage with SQLite warm tier
3. Build FleetMemory as proof-of-concept emergent application
4. Integrate with Quanta C++ bridge when available

---
*Analysis by kimi1, Fleet Orchestrator | 2026-06-08*
