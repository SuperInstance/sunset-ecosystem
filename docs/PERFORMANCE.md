# Performance Guide — Sunset Ecosystem

> **What this is:** Practical tuning advice for engineers running the system at scale.

## The Golden Rules

1. **Hot tier is everything.** 90% of your queries should hit the hot tier. If they don't, your access patterns are wrong or your thresholds are misconfigured.
2. **HNSW is worth the compile time.** At 10,000+ entries, HNSW is 100x faster than brute force. The 5-minute compile time pays for itself in the first hour.
3. **Shards are cheap, big shards are expensive.** Keep shards under 10,000 entries. Querying across 30 small shards is faster than querying one giant shard.
4. **WAL is not free.** Every insert appends to the WAL. In a high-write workload, the WAL can become the bottleneck. Use batching.

## Tuning the Hot Tier

```python
from swarm.tiered_mesh_storage import TierConfig

# For write-heavy workloads (ingestion, logging)
WRITE_HEAVY = TierConfig(
    hot_max_entries=50_000,  # More entries stay hot
    hot_max_age_seconds=3600,  # Short age: promote quickly, demote quickly
    hot_min_fitness=0.0,  # Don't filter by fitness
    demotion_access_threshold=1,  # Promote on first access
)

# For read-heavy workloads (query serving, search)
READ_HEAVY = TierConfig(
    hot_max_entries=10_000,  # Keep only the most valuable
    hot_max_age_seconds=86400,  # 24-hour retention
    hot_min_fitness=0.7,  # Only high-fitness entries stay hot
    demotion_access_threshold=3,  # Need multiple accesses to promote
)

# For mixed workloads (most fleets)
BALANCED = TierConfig(
    hot_max_entries=25_000,
    hot_max_age_seconds=21600,  # 6 hours
    hot_min_fitness=0.3,
    demotion_access_threshold=2,
)
```

## Tuning HNSW

```python
from swarm.hnsw_mesh_table import HnswMeshTable

# For high recall (99%+): research, breeding
HIGH_RECALL = HnswMeshTable(
    base_table=table,
    ef_construction=200,  # Higher = more accurate index, slower build
    M=32,  # Higher = more connections, more memory
    ef_search=100,  # Higher = more accurate search, slower
)

# For low latency (<5ms): real-time serving
LOW_LATENCY = HnswMeshTable(
    base_table=table,
    ef_construction=50,
    M=16,
    ef_search=20,
)

# For balanced (default): most use cases
BALANCED = HnswMeshTable(
    base_table=table,
    ef_construction=100,
    M=24,
    ef_search=50,
)
```

## Memory Usage

| Component | Per-Entry Overhead | Notes |
|-----------|-------------------|-------|
| MeshVectorTable | 128 bytes + vector | Dict + dataclass overhead |
| HNSW index | 64 bytes + vector | Graph connections + vectors |
| SQLite warm | 32 bytes + vector | Row + index overhead |
| Cold archive | 8 bytes + vector | Compressed JSONL |
| FleetMemory shard | 1 KB fixed | Per-shard metadata |

**Example:** 100,000 agents, 256-dim vectors
- Hot tier (10,000): 10,000 × (128 + 256×4) = **12.4 MB**
- HNSW index (10,000): 10,000 × (64 + 256×4) = **10.9 MB**
- Warm tier (90,000): 90,000 × (32 + 256×4) = **95.2 MB**
- Total: **~118 MB** for 100K agents

## Batch Operations

**Don't:**
```python
for agent in agents:
    table.insert(agent)  # 100,000 individual inserts = 100,000 WAL writes
```

**Do:**
```python
# Batch insert (single WAL write per batch)
for batch in chunked(agents, 1000):
    for agent in batch:
        table.insert(agent)
    wal.checkpoint()  # One checkpoint per batch
```

## Query Patterns

**Fast queries (hot tier):**
- `query(agent_id)` — O(1), always hot
- `query_similarity_sorted()` with HNSW — O(log n), hot tier only

**Slow queries (cross-tier):**
- `query_similarity_sorted()` without HNSW — O(n), scans all tiers
- `FleetMemory.query()` across 30 shards — O(shards), but each shard is fast

**Avoid:**
- Querying cold tier directly. If you need cold data, promote it first.
- Querying without filters. Always use `filter_fn` to reduce the search space.

## Benchmarking

```python
import time
import numpy as np
from swarm.mesh_vector_tables import MeshVectorTable

# Setup
table = MeshVectorTable(table_id="bench")
for i in range(10000):
    entry = VectorTableEntry(
        agent_id=f"agent_{i}",
        vector=np.random.randn(256).astype(np.float32),
        timestamp=time.time(),
        node_id="bench",
        generation=0,
        fitness=0.5,
        signature=f"bench_{i}",
    )
    table.insert(entry, skip_verify=True)

# Benchmark
query_vec = np.random.randn(256).astype(np.float32)

start = time.time()
for _ in range(100):
    table.query_similarity_sorted(query_vec, k=10)
print(f"100 queries: {time.time() - start:.3f}s")
```

## Profiling

```python
from swarm.scene_tracker import SceneTracker

# Enable latency tracking
tracker = SceneTracker(table, strategy=CacheStrategy())

# After queries, check latency histogram
print(tracker._latency_by_type)
```

## Common Bottlenecks

### 1. "Queries are slow"

**Diagnosis:**
```python
print(table.stats)  # Check hot/warm/cold distribution
print(hnsw.stats)  # Check if HNSW is available
```

**Fixes:**
- Enable HNSW (compile C++ extension)
- Increase hot tier size
- Reduce `hot_max_age_seconds`
- Add filters to queries

### 2. "Memory is growing"

**Diagnosis:**
```python
print(memory.get_shard_report())  # Check shard count
print(len(storage.base))  # Check hot tier size
```

**Fixes:**
- Reduce `max_shards` in FleetMemory
- Lower `hot_max_entries`
- Run maintenance: `cache.run_maintenance()`

### 3. "WAL is huge"

**Diagnosis:**
```python
import os

print(os.path.getsize("mesh.wal"))  # Check WAL size
```

**Fixes:**
- Run `wal.checkpoint()` more frequently
- Reduce write frequency (batch inserts)
- Archive old WAL files manually

## Scaling Checklist

- [ ] HNSW compiled and enabled
- [ ] Hot tier sized for 90% of queries
- [ ] WAL checkpointing every 1000 inserts
- [ ] Shards under 10,000 entries
- [ ] SceneTracker enabled for CognitiveCache
- [ ] Maintenance running every 5 minutes
- [ ] Monitoring for latency and memory usage

---

*For more details, see ARCHITECTURE.md (design) and DEVELOPER_GUIDE.md (practical usage).*
