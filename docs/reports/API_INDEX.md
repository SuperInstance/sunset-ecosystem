# 📚 Sunset Ecosystem API Documentation

*Auto-generated from 19 modules*

## Table of Contents

- [BreedOptimizer](#breedoptimizer)
- [CASLangExecutor](#caslangexecutor)
- [CognitiveCache](#cognitivecache)
- [EcosystemHub](#ecosystemhub)
- [FleetAPI](#fleetapi)
- [FleetMemory](#fleetmemory)
- [FleetMonitor](#fleetmonitor)
- [HNSWMeshTable](#hnswmeshtable)
- [LevelRunner](#levelrunner)
- [MeshGrouping](#meshgrouping)
- [MeshWAL](#meshwal)
- [PatternMine](#patternmine)
- [Pincher](#pincher)
- [SceneTracker](#scenetracker)
- [TMinusBridge](#tminusbridge)
- [TernaryTypes](#ternarytypes)
- [TieredMeshStorage](#tieredmeshstorage)
- [VectorSwarm](#vectorswarm)
- [xLangAgentBridge](#xlangagentbridge)

## BreedOptimizer

BreedOptimizer — Intelligent breeding optimization for the sunset-ecosystem fleet.

An emergent application that combines:
- Wasserstein distance for diversity-aware parent selection
- Topological anomaly detection for unusual breeding patterns
- VectorSwarm for distributed KNN search
- CognitiveCache for predicting offspring quality
- TMinusBridge for deadline management
- PatternMine for task templates

The optimizer maintains a breeding archive (MAP-Elites style) and uses
optimal transport theory to measure diversity between agent distributions.

Usage
-----
    optimizer = BreedOptimizer(node_id="alpha", swarm=swarm, cache=cache)

    # Score parent pairs by diversity + predicted quality
    parents = optimizer.select_parents(pool, k=3)

    # Detect anomalies in breeding history
    anomalies = optimizer.detect_anomalies(history)

    # Optimize the breeding archive
    optimizer.optimize_archive(archive, iterations=100)

**Source:** `fleet/breed_optimizer.py`
**Tests:** 39/39

### `class ParentPair`

A pair of parent agents for breeding.

### `class OffspringPrediction`

Prediction for offspring quality.

### `class BreedingArchive`

MAP-Elites style archive for breeding history.

#### `add(behavior, individual)`

Add an individual to the archive.

*Source: `fleet/breed_optimizer.py:84`*

#### `_to_indices(behavior)`

Convert behavior coordinates to archive indices.

*Source: `fleet/breed_optimizer.py:92`*

#### `_update_metrics()`

Update coverage and QD-score.

*Source: `fleet/breed_optimizer.py:99`*

#### `get_best_in_cell(indices)`

Get the best individual in a cell.

*Source: `fleet/breed_optimizer.py:108`*

#### `sample_diverse(k)`

Sample k diverse individuals from different cells.

*Source: `fleet/breed_optimizer.py:119`*

### `class AnomalyResult`

Result of anomaly detection on breeding history.

### `class BreedOptimizer`

Intelligent breeding optimizer for the fleet.

Parameters
----------
node_id : str
    Node identifier.
swarm : VectorSwarm | None
    Distributed search layer.
cache : CognitiveCache | None
    Prediction cache.
tminus : TMinusBridge | None
    Deadline management.

#### `__init__(node_id, swarm, cache, tminus)`

*Source: `fleet/breed_optimizer.py:152`*

#### `wasserstein_distance(distribution_a, distribution_b)`

Compute 1-Wasserstein distance between two distributions.

Uses the Earth Mover's Distance formula:
W_1(A, B) = ∫ |F_A(x) - F_B(x)| dx
where F is the cumulative distribution function.

Parameters
----------
distribution_a : list[float]
    First distribution (sorted or unsorted).
distribution_b : list[float]
    Second distribution.

Returns
-------
float
    Wasserstein distance (0 = identical).

*Source: `fleet/breed_optimizer.py:169`*

#### `diversity_score(agent_a_traits, agent_b_traits)`

Compute diversity score between two agents.

Combines Wasserstein distance with trait overlap.
Higher = more diverse (better for breeding).

*Source: `fleet/breed_optimizer.py:211`*

#### `select_parents(pool, k, diversity_weight)`

Select top-k parent pairs by diversity + predicted quality.

Parameters
----------
pool : list[dict]
    Agents with "id" and "traits" keys.
k : int
    Number of pairs to return.
diversity_weight : float
    Weight for diversity vs predicted quality (0-1).

Returns
-------
list[ParentPair]
    Sorted by composite score (highest first).

*Source: `fleet/breed_optimizer.py:228`*

#### `_predict_offspring(traits_a, traits_b)`

Predict offspring quality from parent traits.

Simple model: offspring fitness = mean(parent fitnesses) + crossover bonus.

*Source: `fleet/breed_optimizer.py:286`*

#### `detect_anomalies(history, threshold)`

Detect anomalies in breeding history.

        Uses simple statistical outlier detection:
        - Z-score > threshold for fitness drop
- Sudden diversity collapse
- Repeated parent pairs (inbreeding)

        Parameters
        ----------
        history : list[dict] | None
            Breeding history entries. If None, uses internal history.
        threshold : float
            Z-score threshold for anomaly detection.

        Returns
        -------
        list[AnomalyResult]
            Detected anomalies.


*Source: `fleet/breed_optimizer.py:335`*

#### `optimize_archive(archive, iterations)`

Optimize a breeding archive using MAP-Elites style selection.

Parameters
----------
archive : BreedingArchive | None
    Archive to optimize. If None, uses internal archive.
iterations : int
    Number of optimization iterations.

Returns
-------
BreedingArchive
    Optimized archive.

*Source: `fleet/breed_optimizer.py:414`*

#### `distributed_select_parents(pool, k)`

Select parents using distributed swarm search.

If VectorSwarm is available, distribute the search across nodes.
Otherwise, falls back to local selection.

*Source: `fleet/breed_optimizer.py:457`*

#### `set_breeding_deadline(parent_deadline, child_budget)`

Set breeding deadline with parent→child inheritance.

Uses TMinusBridge if available, otherwise simple min().

*Source: `fleet/breed_optimizer.py:477`*

#### `record_breeding(parent_a, parent_b, offspring_fitness, diversity, traits)`

Record a breeding event in history.

*Source: `fleet/breed_optimizer.py:492`*

#### `get_history()`

Get breeding history.

*Source: `fleet/breed_optimizer.py:511`*

#### `get_stats()`

Get optimizer statistics.

*Source: `fleet/breed_optimizer.py:515`*

#### `generate_report()`

Generate comprehensive optimizer report.

*Source: `fleet/breed_optimizer.py:541`*

---

## CASLangExecutor

CaslangExecutor — Constrained JSONL execution bridge for the fleet.

Integrates xlang-foundation's caslang (constrained scripting language)
into the sunset-ecosystem as a deterministic, sandboxed execution layer
for LLM-generated plans.

Provides:
- JSON task graph → caslang JSONL conversion
- Schema-validated execution with pre-flight checks
- Sandboxed filesystem / network access
- Rollback on failure with transaction semantics

Architecture
------------
caslang is a JSONL-based execution language where each line is a valid
JSON object representing a single command.  It is designed to be:

- **LLM-friendly**: Single-pass generation in strict format reduces hallucinations
- **Machine-validated**: Pre-execution schema checks guarantee safe host behavior
- **Privacy by default**: Execution happens locally; local data stays local

The bridge maps our `autonomous_repo.py` JSON task graphs into caslang
scripts, executes them in a sandboxed environment, and reports results
back to the fleet conductor.

Reference
---------
- caslang spec: https://github.com/xlang-foundation/caslang

**Source:** `fleet/caslang_executor.py`
**Tests:** 18/18

### `class ValidationError`

Raised when a caslang script fails pre-execution validation.

### `class SandboxViolation`

Raised when a command attempts an operation outside the sandbox.

### `class CaslangScript`

A caslang script is a list of JSONL command objects.

#### `to_jsonl()`

Serialize to caslang JSONL format.

*Source: `fleet/caslang_executor.py:78`*

#### `from_jsonl(text)`

Parse from caslang JSONL format.

*Source: `fleet/caslang_executor.py:86`*

#### `from_task_graph(graph)`

Convert a sunset-ecosystem JSON task graph to caslang.

Task graph format (from autonomous_repo.py):
{
    "tasks": [
        {"id": "t1", "action": "read_file", "params": {"path": "/data/foo"}},
        {"id": "t2", "action": "write_file", "params": {"path": "/data/bar", "data": "..."}},
    ],
    "dependencies": [{"from": "t1", "to": "t2"}]
}

*Source: `fleet/caslang_executor.py:101`*

#### `_convert_action(action, params, tid)`

Map a sunset action to a caslang command.

*Source: `fleet/caslang_executor.py:150`*

### `class ExecutionSandbox`

Sandboxed execution environment for caslang scripts.

Parameters
----------
allowed_paths : list[str]
    Whitelisted filesystem paths (absolute).
allowed_tools : list[str]
    Whitelisted tool names (e.g. "http_get", "semantic_search").
max_file_size : int
    Maximum bytes allowed for file operations.
network_enabled : bool
    Whether HTTP/network tools are allowed.

#### `__init__(allowed_paths, allowed_tools, max_file_size, network_enabled)`

*Source: `fleet/caslang_executor.py:196`*

#### `validate(script)`

Pre-flight validation: check every command against the sandbox.

Returns a list of warnings/errors.  Empty list means clean.

*Source: `fleet/caslang_executor.py:215`*

#### `_is_path_allowed(path)`

Check if a path is within the allowed set.

*Source: `fleet/caslang_executor.py:246`*

### `class CaslangExecutor`

Deterministic, sandboxed executor for caslang scripts.

Parameters
----------
sandbox : ExecutionSandbox
    The sandbox that constrains what scripts can do.
rollback_enabled : bool
    If True, on failure the executor attempts to undo filesystem changes.

#### `__init__(sandbox, rollback_enabled)`

*Source: `fleet/caslang_executor.py:272`*

#### `convert_task_graph(graph)`

Convert a JSON task graph to a caslang script.

*Source: `fleet/caslang_executor.py:286`*

#### `execute(script)`

Execute a caslang script in the sandbox.

Returns a result dict with status, output, and execution log.

*Source: `fleet/caslang_executor.py:292`*

#### `_resolve_value(raw, variables)`

Resolve template references like ${var} or ${data['key']}.

*Source: `fleet/caslang_executor.py:458`*

#### `_check_path(path)`

Verify a path is within the sandbox.

*Source: `fleet/caslang_executor.py:479`*

#### `_snapshot_state()`

Capture filesystem state for rollback.

*Source: `fleet/caslang_executor.py:484`*

#### `_rollback(state)`

Undo filesystem changes from the failed script.

*Source: `fleet/caslang_executor.py:492`*

#### `stats()`

*Source: `fleet/caslang_executor.py:506`*

---

## CognitiveCache

CognitiveCache — Adaptive cache that learns from query patterns.

Combines SceneTracker (pattern detection) with tiered storage (promotion/demotion)
to create an intelligent cache that preloads data before it's needed.

Key concepts:
- **Pattern learning**: SceneTracker identifies hot queries and co-occurrence
- **Predictive preloading**: Pre-load co-occurring entries before they're queried
- **Adaptive tiering**: Promote predicted entries to hot tier, demote cold ones
- **Feedback loop**: Track prediction accuracy and adjust thresholds

Architecture:
  Query → SceneTracker (pattern detection)
            ↓
  Prediction engine (co-occurrence + recency)
            ↓
  TieredMeshStorage (promote/demote)
            ↓
  Feedback loop (accuracy tracking)

Use cases:
- **Fleet breeding**: Preload parent candidates when breeding starts
- **Health monitoring**: Preload related agents when one agent is queried
- **Pattern mining**: Keep cluster centroids hot while demoting outliers
- **Anomaly detection**: Hot normal patterns, cold rare events

Reference: docs/DEVELOPER_GUIDE.md — CognitiveCache

**Source:** `fleet/cognitive_cache.py`
**Tests:** 15/15

### `class CachePrediction`

A single prediction for cache preloading.

### `class PredictionEngine`

Generates predictions from SceneTracker patterns.

Parameters
----------
cooccurrence_threshold : float
    Minimum co-occurrence ratio to trigger preloading.
hot_threshold : int
    Minimum access count to consider an agent "hot".
recency_window_seconds : float
    Time window for recency-based predictions.

#### `predict(tracker, recent_queries)`

Generate predictions based on tracker patterns.

Parameters
----------
tracker : SceneTracker
    Scene tracker with observed patterns.
recent_queries : int
    Number of recent queries to consider.

Returns
-------
list[CachePrediction]
    Predicted agent IDs with confidence scores.

*Source: `fleet/cognitive_cache.py:74`*

### `class CognitiveCache`

Adaptive cache combining pattern detection and tiered storage.

Parameters
----------
storage : TieredMeshStorage
    Tiered storage backend.
tracker : SceneTracker
    Scene tracker for pattern detection.
engine : PredictionEngine
    Prediction engine for preloading.

#### `__init__(storage, tracker, engine)`

*Source: `fleet/cognitive_cache.py:145`*

#### `query(agent_id)`

Query with automatic tracking and predictive preloading.

Parameters
----------
agent_id : str
    Agent ID to query.

Returns
-------
VectorTableEntry | None
    Entry if found.

*Source: `fleet/cognitive_cache.py:160`*

#### `query_similar(vector, k)`

Similarity query with tracking.

Parameters
----------
vector : np.ndarray
    Query vector.
k : int
    Number of results.

Returns
-------
list[VectorTableEntry]
    Similar entries.

*Source: `fleet/cognitive_cache.py:200`*

#### `_preload_predictions()`

Preload predicted entries into hot tier.

*Source: `fleet/cognitive_cache.py:233`*

#### `run_maintenance()`

Run maintenance: demote cold entries, rebuild predictions.

*Source: `fleet/cognitive_cache.py:254`*

#### `stats()`

Cache statistics.

*Source: `fleet/cognitive_cache.py:262`*

---

## EcosystemHub

EcosystemHub — Auto-discovery and integration mapping for the SuperInstance fleet.

An emergent application that:
- Discovers repos via GitHub API
- Maps them to integration opportunities in sunset-ecosystem
- Tracks which repos have Python bridges
- Suggests priority order based on impact × effort
- Generates integration task cards for FleetMonitor

Usage
-----
    hub = EcosystemHub("SuperInstance")
    hub.discover()
    hub.map_integrations()
    for task in hub.suggest_priority_tasks():
        print(task.priority, task.target_repo, task.integration_module)

**Source:** `fleet/ecosystem_hub.py`
**Tests:** 14/14

### `class RepoCard`

A discovered repository with metadata.

### `class IntegrationTask`

A concrete integration task.

### `class IntegrationMap`

A mapping from a SuperInstance repo to a sunset-ecosystem module.

### `class EcosystemHub`

Discover and map the SuperInstance repo collection.

Parameters
----------
org : str
    GitHub organization name (default: "SuperInstance").
cache_path : Path | None
    Where to cache discovered repo list.

#### `__init__(org, cache_path)`

*Source: `fleet/ecosystem_hub.py:90`*

#### `discover(force_refresh)`

Discover all repos in the organization.

Uses cached results if available and fresh (<24h).

*Source: `fleet/ecosystem_hub.py:104`*

#### `_auto_tag(card)`

Auto-tag repos based on name/description.

*Source: `fleet/ecosystem_hub.py:159`*

#### `_save_cache()`

Save discovered repos to cache file.

*Source: `fleet/ecosystem_hub.py:200`*

#### `_repo_to_dict(card)`

*Source: `fleet/ecosystem_hub.py:209`*

#### `map_integrations()`

Map discovered repos to sunset-ecosystem integration opportunities.

Uses hard-coded rules based on repo analysis.

*Source: `fleet/ecosystem_hub.py:226`*

#### `suggest_priority_tasks()`

Generate concrete integration tasks sorted by priority and impact.

*Source: `fleet/ecosystem_hub.py:387`*

#### `generate_report()`

Generate a comprehensive ecosystem report.

*Source: `fleet/ecosystem_hub.py:438`*

#### `write_report(path)`

Write the ecosystem report to a markdown file.

*Source: `fleet/ecosystem_hub.py:505`*

---

## FleetAPI

Fleet API — FastAPI server for sunset-ecosystem operations.

Endpoints:
  GET  /health          Health check
  GET  /status           Fleet status
  POST /agents          Insert agent
  GET  /agents/{id}     Query agent by ID
  POST /agents/similar  Similarity search
  POST /memory/write    Write to FleetMemory
  POST /memory/query    Query FleetMemory
  POST /swarm/knn       Distributed KNN search
  GET  /cache/stats     CognitiveCache stats
  POST /cache/maintenance Run maintenance
  GET  /tests           Test inventory

Usage:
  uvicorn fleet.fleet_api:app --host 0.0.0.0 --port 8000

**Source:** `fleet/fleet_api.py`
**Tests:** 8/8

### `class AgentEntry`

### `class SimilarityQuery`

### `class MemoryWrite`

### `class MemoryQuery`

### `class SwarmKnnQuery`

### `health()`

*Source: `fleet/fleet_api.py:91`*

### `status()`

*Source: `fleet/fleet_api.py:98`*

### `insert_agent(entry)`

*Source: `fleet/fleet_api.py:116`*

### `get_agent(agent_id)`

*Source: `fleet/fleet_api.py:134`*

### `_brute_force_knn(table, vector, k)`

Brute-force KNN search (fallback when HNSW is unavailable).

*Source: `fleet/fleet_api.py:148`*

### `similar_agents(query)`

*Source: `fleet/fleet_api.py:166`*

### `memory_write(req)`

*Source: `fleet/fleet_api.py:186`*

### `memory_query(req)`

*Source: `fleet/fleet_api.py:194`*

### `memory_shards()`

*Source: `fleet/fleet_api.py:225`*

### `cache_stats()`

*Source: `fleet/fleet_api.py:233`*

### `cache_maintenance()`

*Source: `fleet/fleet_api.py:238`*

### `swarm_knn(query)`

*Source: `fleet/fleet_api.py:246`*

### `test_inventory()`

*Source: `fleet/fleet_api.py:271`*

---

## FleetMemory

FleetMemory — Persistent time-partitioned cognition for the fleet.

An emergent application that combines:
- MeshVectorTable CRDT sync for cross-node consistency
- TieredMeshStorage for hot/warm/cold tiered persistence
- HnswMeshTable for fast ANN search
- Time-based partitioning for temporal queries

Use Cases
---------
- **Collective Memory**: Query what the fleet learned last month
- **Experience Replay**: Retrieve similar past decisions for new situations
- **Temporal Analysis**: Track how agent capabilities evolved over time
- **Knowledge Archaeology**: Find forgotten insights from early generations

Architecture
------------
FleetMemory operates on "memory shards" — time-partitioned slices of
the fleet's experience. Each shard is a MeshVectorTable with a specific
time range (e.g., "2026-06-01 to 2026-06-07"). Shards are tiered:
- **Recent shard** (hot): Current week, in memory, fast ANN search
- **Older shards** (warm/cold): Archived, queryable on demand

Memory entries are VectorTableEntry objects with:
- agent_id: Who had the experience
- vector: Semantic embedding of the experience
- timestamp: When it happened
- generation: Which breeding generation
- fitness: How valuable was the outcome
- extra: Context (task description, result, lessons learned)

Reference: docs/QUANTA_VDB_DEEP_DIVE.md — Emergent Application: FleetMemory

**Source:** `fleet/fleet_memory.py`
**Tests:** 12/12

### `class TemporalQuery`

Query parameters for temporal memory search.

### `class MemoryEntry`

Human-readable memory entry with metadata.

### `class MemoryShard`

A single time-partitioned memory shard.

Each shard covers a time range and has its own HNSW index.

#### `__init__(shard_id, start_time, end_time, dim, identity)`

*Source: `fleet/fleet_memory.py:89`*

#### `add_memory(entry)`

Add a memory entry to this shard.

*Source: `fleet/fleet_memory.py:130`*

#### `query_similar(vector, k, min_fitness)`

Find memories similar to the given vector.

*Source: `fleet/fleet_memory.py:136`*

#### `query_by_agent(agent_id)`

Find all memories from a specific agent.

*Source: `fleet/fleet_memory.py:158`*

#### `get_stats()`

*Source: `fleet/fleet_memory.py:167`*

#### `close()`

*Source: `fleet/fleet_memory.py:176`*

### `class FleetMemory`

Fleet-wide persistent memory system with time-partitioned shards.

Parameters
----------
node_id : str
    Identifier for this fleet node.
dim : int
    Vector dimension for memory embeddings.
shard_duration_seconds : float
    Duration of each memory shard (default: 86400 = 1 day).
max_active_shards : int
    Maximum number of shards kept in hot memory.
identity : AgentIdentity | None
    For signing memory entries.

#### `__init__(node_id, dim, shard_duration_seconds, max_active_shards, identity)`

*Source: `fleet/fleet_memory.py:197`*

#### `remember(agent_id, vector, timestamp, generation, fitness, context)`

Store a memory entry in the appropriate time shard.

Parameters
----------
agent_id : str
    Who had this experience.
vector : np.ndarray
    Semantic embedding of the experience.
timestamp : float | None
    Unix timestamp. Defaults to now.
generation : int
    Breeding generation.
fitness : float
    Outcome quality (0.0-1.0).
context : dict
    Additional context (task, result, lessons).

Returns
-------
bool
    True if stored successfully.

*Source: `fleet/fleet_memory.py:222`*

#### `recall(query)`

Retrieve memories matching temporal and similarity criteria.

Parameters
----------
query : TemporalQuery
    Search parameters.

Returns
-------
list[MemoryEntry]
    Matching memories sorted by relevance.

*Source: `fleet/fleet_memory.py:275`*

#### `recall_similar(vector, k, start_time, end_time, min_fitness)`

Quick recall by vector similarity.

*Source: `fleet/fleet_memory.py:335`*

#### `get_memory_stats()`

Return fleet memory statistics.

*Source: `fleet/fleet_memory.py:352`*

#### `get_shard_report()`

Return detailed report of all shards.

*Source: `fleet/fleet_memory.py:365`*

#### `close()`

Close all shards and release resources.

*Source: `fleet/fleet_memory.py:377`*

#### `_timestamp_to_shard_id(timestamp)`

Convert timestamp to shard ID (e.g., '2026-06-08').

*Source: `fleet/fleet_memory.py:387`*

#### `_shard_id_to_time_range(shard_id)`

Convert shard ID to time range.

*Source: `fleet/fleet_memory.py:393`*

#### `_get_or_create_shard(shard_id)`

Get existing shard or create new one.

*Source: `fleet/fleet_memory.py:402`*

#### `_get_shard(shard_id)`

Get shard by ID, loading from disk if needed.

*Source: `fleet/fleet_memory.py:431`*

#### `_select_shards(start_time, end_time)`

Select shard IDs that overlap with the time range.

*Source: `fleet/fleet_memory.py:439`*

---

## FleetMonitor

FleetMonitor — Observability and health tracking for the fleet.

Tracks:
- Node health (CPU, memory, latency)
- Mesh table stats (entries, sync, divergence)
- Tiered storage distribution (hot/warm/cold)
- HNSW index status (build, coverage, rebuilds)
- CognitiveCache predictions (hit rate, accuracy)
- SceneTracker patterns (scenes, transitions, anomalies)
- Alert thresholds and escalation

Usage:
    monitor = FleetMonitor()
    monitor.register_node("node1", table1, storage1, hnsw1)
    monitor.register_node("node2", table2, storage2, hnsw2)

    # Get fleet-wide health report
    report = monitor.health_report()

    # Check alerts
    alerts = monitor.check_alerts()
    for alert in alerts:
        print(alert.level, alert.message)

**Source:** `fleet/fleet_monitor.py`
**Tests:** 10/10

### `class AlertLevel`

### `class Alert`

### `class NodeHealth`

### `class FleetMonitor`

Fleet-wide observability and health tracking.

#### `__init__(cache_hit_threshold, cache_accuracy_threshold, hnsw_coverage_threshold, hot_ratio_threshold, query_rate_threshold)`

*Source: `fleet/fleet_monitor.py:81`*

#### `register_node(node_id, table, storage, hnsw, cache, tracker)`

Register a node for monitoring.

*Source: `fleet/fleet_monitor.py:100`*

#### `health_report()`

Generate fleet-wide health report.

*Source: `fleet/fleet_monitor.py:119`*

#### `_node_health(node_id, node)`

Compute health for a single node.

*Source: `fleet/fleet_monitor.py:144`*

#### `_health_to_dict(health)`

*Source: `fleet/fleet_monitor.py:190`*

#### `check_alerts()`

Check all nodes for alert conditions.

*Source: `fleet/fleet_monitor.py:208`*

#### `_node_alerts(health)`

Generate alerts for a single node.

*Source: `fleet/fleet_monitor.py:216`*

#### `snapshot()`

Take a snapshot of fleet state for persistence.

*Source: `fleet/fleet_monitor.py:290`*

#### `get_history(n)`

Get last n snapshots.

*Source: `fleet/fleet_monitor.py:317`*

#### `get_trends(metric, node_id)`

Get trend for a metric over time.

*Source: `fleet/fleet_monitor.py:321`*

---

## HNSWMeshTable

HnswMeshTable — HNSW-powered approximate nearest neighbor for MeshVectorTables.

Adds O(log n) vector search to the O(n) brute-force MeshVectorTable.
Uses hnswlib (optional dependency) with graceful fallback.

Integration:
- Wraps MeshVectorTable as backing store
- HNSW index is a hot-cache overlay (rebuildable from backing store)
- CRDT sync still operates on the backing store
- ANN queries are served from HNSW with verification from backing store

Reference: docs/QUANTA_VDB_DEEP_DIVE.md — Phase 1: Hybrid Index

**Source:** `swarm/hnsw_mesh_table.py`
**Tests:** 11/11

### `class HnswIndexConfig`

Configuration for HNSW index.

### `class HnswMeshTable`

MeshVectorTable with HNSW approximate nearest neighbor overlay.

Parameters
----------
base_table : MeshVectorTable
    The underlying CRDT table (all writes go here).
config : HnswIndexConfig
    HNSW index parameters.
auto_rebuild_threshold : float
    Rebuild HNSW index when fraction of entries changed exceeds this.

#### `__init__(base_table, config, auto_rebuild_threshold)`

*Source: `swarm/hnsw_mesh_table.py:62`*

#### `insert(entry)`

Insert into base table + HNSW index.

*Source: `swarm/hnsw_mesh_table.py:86`*

#### `knn_search(query_vector, k, filter_fn)`

Approximate k-nearest neighbor search.

Parameters
----------
query_vector : np.ndarray
    The query vector.
k : int
    Number of nearest neighbors to return.
filter_fn : callable | None
    Optional post-filter (entry -> bool).

Returns
-------
list[tuple[VectorTableEntry, float]]
    Results sorted by distance ascending.

*Source: `swarm/hnsw_mesh_table.py:96`*

#### `range_search(query_vector, radius, max_results)`

Find all entries within *radius* of query_vector.

*Source: `swarm/hnsw_mesh_table.py:162`*

#### `get_novelty_neighbors(entry, k)`

Find the k nearest neighbors of *entry* and return distances.
Novelty = average distance to neighbors.

*Source: `swarm/hnsw_mesh_table.py:174`*

#### `compute_local_density(entry, k)`

Local density = 1 / (average distance to k nearest neighbors).
High density = region is crowded. Low density = sparse region.

*Source: `swarm/hnsw_mesh_table.py:186`*

#### `find_sparse_regions(k, n_samples)`

Find entries in sparse (low density) regions.
Returns (entry, density) sorted by density ascending.

*Source: `swarm/hnsw_mesh_table.py:201`*

#### `stats()`

*Source: `swarm/hnsw_mesh_table.py:227`*

#### `_rebuild_index()`

Rebuild HNSW index from base table.

*Source: `swarm/hnsw_mesh_table.py:240`*

#### `_add_to_hnsw(entry)`

Add a single entry to HNSW index.

*Source: `swarm/hnsw_mesh_table.py:283`*

#### `_maybe_trigger_rebuild()`

Auto-rebuild if too many changes.

*Source: `swarm/hnsw_mesh_table.py:312`*

#### `_brute_force_knn(vec, k, filter_fn)`

Fallback brute-force search.

*Source: `swarm/hnsw_mesh_table.py:321`*

---

## LevelRunner

LevelRunner — Simulation / game level execution engine for the fleet.

An emergent application that combines xlang's event-driven runtime,
Quanta's streaming VDB for level state tracking, and caslang's
constrained execution for deterministic level logic.

Use Cases
---------
- **AI Training Scenarios**: Generate procedural levels for agent training
- **Fleet Stress Testing**: Simulate 1000-agent scenarios to find bottlenecks
- **Game Worlds**: Persistent game worlds where agents are NPCs
- **Synthetic Data**: Generate realistic simulation data for ML training

Architecture
------------
The LevelRunner operates on a "Level" abstraction:

1. **Level Definition** — A YAML/JSON spec defining terrain, entities,
   rules, and victory conditions.  Converted to caslang for deterministic
   execution.

2. **State VDB** — Quanta PartitionedVdb stores entity positions, health,
   inventory, and relationships as high-dimensional vectors.  Enables fast
   spatial queries ("find all agents within 50m of point X").

3. **Event Engine** — xlang's event bus handles real-time events
   (collision, combat, trade, discovery).  Agents react via registered
   event handlers.

4. **Tick Loop** — Deterministic simulation ticks at configurable Hz.
   Each tick: (1) process events, (2) update physics, (3) run agent
   AI, (4) commit state to VDB.

Reference
---------
- xlang events: https://github.com/xlang-foundation/xlang/Docs/xlang_spec.md
- Quanta VDB: https://github.com/CantorAI/Quanta

**Source:** `fleet/level_runner.py`
**Tests:** 18/18

### `class Entity`

A single entity in the level (agent, NPC, object, terrain).

#### `__post_init__()`

*Source: `fleet/level_runner.py:80`*

#### `to_vector(dim)`

Serialize entity state to a high-dimensional vector for VDB storage.

*Source: `fleet/level_runner.py:84`*

#### `from_vector(entity_id, entity_type, vector)`

Reconstruct entity from VDB vector (partial, loses fidelity).

*Source: `fleet/level_runner.py:109`*

### `class LevelDefinition`

Specification for a simulation level.

#### `to_caslang()`

Convert level rules to a caslang script for deterministic execution.

*Source: `fleet/level_runner.py:136`*

### `class EventBus`

Lightweight event bus for level simulation (xlang-inspired).

#### `__init__()`

*Source: `fleet/level_runner.py:158`*

#### `on(event_name, handler)`

Register an event handler.

*Source: `fleet/level_runner.py:163`*

#### `emit(event_name, payload)`

Emit an event to all registered handlers.

*Source: `fleet/level_runner.py:168`*

#### `clear()`

Remove all handlers.

*Source: `fleet/level_runner.py:179`*

### `class LevelState`

Mutable state container for a running level.

#### `__init__(definition)`

*Source: `fleet/level_runner.py:191`*

#### `add_entity(entity)`

Add an entity to the level.

*Source: `fleet/level_runner.py:200`*

#### `remove_entity(entity_id)`

Remove an entity from the level.

*Source: `fleet/level_runner.py:213`*

#### `get_entities_near(position, radius, entity_type)`

Spatial query: find entities within radius of position.

*Source: `fleet/level_runner.py:222`*

#### `get_entities_by_faction(faction)`

Return all entities belonging to a faction.

*Source: `fleet/level_runner.py:240`*

#### `check_victory()`

Check if any victory condition is met.

*Source: `fleet/level_runner.py:245`*

#### `stats()`

*Source: `fleet/level_runner.py:266`*

### `class LevelRunner`

Execute simulation levels with deterministic tick loops.

Parameters
----------
quanta_bridge : QuantaVdbBridge | None
    If provided, entity state is persisted to Quanta VDB each tick.
caslang_executor : CaslangExecutor | None
    If provided, entity AI scripts are executed via caslang sandbox.

#### `__init__(quanta_bridge, caslang_executor)`

*Source: `fleet/level_runner.py:290`*

#### `load_level(definition)`

Load a level definition and return a level ID.

*Source: `fleet/level_runner.py:310`*

#### `spawn_entity(level_id, entity_id, entity_type, position, faction, ai_script)`

Spawn an entity into a running level.

*Source: `fleet/level_runner.py:327`*

#### `start_level(level_id)`

Start the tick loop for a level.

*Source: `fleet/level_runner.py:353`*

#### `stop_level(level_id)`

Stop the tick loop for a level.

*Source: `fleet/level_runner.py:378`*

#### `get_level_state(level_id)`

Return the current state of a level.

*Source: `fleet/level_runner.py:390`*

#### `on_tick(level_id, callback)`

Register a callback to run after each tick.

*Source: `fleet/level_runner.py:395`*

#### `_run_tick(level_id)`

Execute one simulation tick.

*Source: `fleet/level_runner.py:402`*

#### `_process_ai(state)`

Run AI scripts for all entities.

*Source: `fleet/level_runner.py:442`*

#### `_update_physics(state)`

Simple Euler physics integration.

*Source: `fleet/level_runner.py:466`*

#### `_check_collisions(state)`

Naive O(n²) collision detection (sufficient for small levels).

*Source: `fleet/level_runner.py:477`*

#### `_handle_collision(payload)`

Default collision handler.

*Source: `fleet/level_runner.py:492`*

#### `_handle_combat(payload)`

Default combat handler.

*Source: `fleet/level_runner.py:496`*

#### `_handle_spawn(payload)`

Default spawn handler.

*Source: `fleet/level_runner.py:503`*

#### `_persist_to_vdb(state)`

Persist entity vectors to Quanta VDB.

*Source: `fleet/level_runner.py:507`*

#### `stats()`

*Source: `fleet/level_runner.py:537`*

---

## MeshGrouping

MeshGrouping — centroid clustering and pattern discovery for MeshVectorTables.

Discovers emergent groups from vector populations using:
- **K-means clustering** for dense population partitioning
- **Centroid tracking** for stable group identities over time
- **Group quality metrics** (cohesion, separation, novelty)
- **Automatic group merging/splitting** based on drift thresholds
- **Pattern labels** for human-readable group descriptions

Use Cases
---------
- **Pattern Discovery**: "What kinds of agents are in the fleet?" → auto-discover clusters
- **Cohort Analysis**: Track agent behavior patterns over time
- **Anomaly Detection**: Identify agents that don't fit any group (outliers)
- **Diversity-aware Breeding**: Select parents from different groups for diversity
- **Fleet Health**: Detect group collapse (agents leaving a cluster)

Architecture
------------
Groups are represented as:
  Group {
    group_id: str
    centroid: np.ndarray
    members: set[agent_id]
    cohesion: float     # avg similarity to centroid
    separation: float   # distance to nearest other centroid
    birth_time: float
    last_update: float
    label: str          # auto-generated description
  }

Clustering algorithms:
- "kmeans": Scikit-learn KMeans (if available) or custom implementation
- "hierarchical": Agglomerative clustering for dendrogram analysis
- "dbscan": Density-based for irregular shapes
- "single_pass": Online incremental clustering (streaming)

Reference: docs/QUANTA_VDB_DEEP_DIVE.md — Remaining Gaps: Grouping

**Source:** `swarm/mesh_grouping.py`
**Tests:** 10/10

### `class ClusterConfig`

Configuration for clustering algorithms.

### `class GroupProfile`

A discovered group of agents.

#### `to_dict()`

*Source: `swarm/mesh_grouping.py:97`*

### `class MeshGrouping`

Clustering and pattern discovery for MeshVectorTables.

Parameters
----------
table : MeshVectorTable
    The table to cluster.
config : ClusterConfig
    Clustering algorithm configuration.

#### `__init__(table, config)`

*Source: `swarm/mesh_grouping.py:123`*

#### `cluster()`

Run clustering on all entries in the table.

Returns
-------
list[GroupProfile]
    Discovered groups.

*Source: `swarm/mesh_grouping.py:137`*

#### `incremental_update(new_entry)`

Update groups incrementally when a new entry arrives.

Uses single-pass clustering for online updates.

Parameters
----------
new_entry : VectorTableEntry
    The new entry to incorporate.

Returns
-------
GroupProfile | None
    The group the entry was assigned to, or None if outlier.

*Source: `swarm/mesh_grouping.py:186`*

#### `find_outliers()`

Find agents that don't fit well in any group (low silhouette).

*Source: `swarm/mesh_grouping.py:261`*

#### `find_dense_regions(k)`

Find groups with highest cohesion (dense clusters).

Returns
-------
list[tuple[str, float]]
    (group_id, cohesion) sorted descending.

*Source: `swarm/mesh_grouping.py:276`*

#### `find_sparse_regions(k)`

Find groups with lowest cohesion (sparse/diffuse clusters).

Returns
-------
list[tuple[str, float]]
    (group_id, cohesion) sorted ascending.

*Source: `swarm/mesh_grouping.py:288`*

#### `get_group_members(group_id)`

Get all entries in a group.

*Source: `swarm/mesh_grouping.py:300`*

#### `get_group_centroid(group_id)`

Get the centroid of a group.

*Source: `swarm/mesh_grouping.py:307`*

#### `compute_diversity_index()`

Compute a diversity score: number of groups × average separation.

Higher = more diverse fleet.

*Source: `swarm/mesh_grouping.py:314`*

#### `compute_cohesion_map()`

Map group_id -> cohesion score.

*Source: `swarm/mesh_grouping.py:325`*

#### `stats()`

*Source: `swarm/mesh_grouping.py:332`*

#### `_kmeans_cluster(vectors)`

K-means clustering using sklearn or custom fallback.

*Source: `swarm/mesh_grouping.py:346`*

#### `_custom_kmeans(vectors)`

Custom K-means implementation (no sklearn).

*Source: `swarm/mesh_grouping.py:359`*

#### `_hierarchical_cluster(vectors)`

Hierarchical clustering using sklearn or custom fallback.

*Source: `swarm/mesh_grouping.py:388`*

#### `_dbscan_cluster(vectors)`

DBSCAN clustering using sklearn or custom fallback.

*Source: `swarm/mesh_grouping.py:397`*

#### `_single_pass_cluster(vectors)`

Single-pass incremental clustering.

*Source: `swarm/mesh_grouping.py:405`*

#### `_single_element_groups(entries)`

Create one group per entry when too few for clustering.

*Source: `swarm/mesh_grouping.py:430`*

#### `_create_group(group_id, members)`

Create a GroupProfile from members.

*Source: `swarm/mesh_grouping.py:448`*

#### `_compute_quality_metrics(vectors, labels)`

Compute cohesion, separation, and silhouette for groups.

*Source: `swarm/mesh_grouping.py:464`*

#### `_get_group_label(label_idx)`

Map label index to group_id.

*Source: `swarm/mesh_grouping.py:495`*

#### `_cosine_similarity(a, b)`

Compute cosine similarity between two vectors.

*Source: `swarm/mesh_grouping.py:500`*

---

## MeshWAL

MeshWAL — Write-Ahead Log crash recovery for MeshVectorTables.

Implements durable, append-only WAL for mesh vector tables with:
- Atomic append of CRDT operations (insert, merge, delete)
- Automatic checkpointing (truncate WAL after successful sync)
- Crash recovery replay on startup
- Batch compression for efficiency
- CRC32 checksums for integrity verification

Use Cases
---------
- **Crash Recovery**: Node crashes and restarts — replay WAL to restore state
- **Durability**: Ensure no data loss even on power failure
- **Replication**: WAL entries can be streamed to replicas for hot standby
- **Audit Trail**: Complete history of all mutations for debugging

Architecture
------------
The WAL is a sequence of append-only files:
  mesh_wal_000001.log  →  mesh_wal_000002.log  →  ...

Each file contains binary records:
  [magic:4][crc32:4][timestamp:8][payload_len:4][payload:N]

Payloads are zlib-compressed JSON:
  {"op": "insert", "entry": {...}}  or  {"op": "merge", "payload": {...}}  or  {"op": "delete", "agent_id": "..."}

Checkpoints are written to a separate metadata file:
  checkpoint.json: { "last_wal_file": "mesh_wal_000005.log", "last_offset": 12345, "timestamp": 1000.0 }

On startup, if checkpoint exists, replay from checkpoint. Otherwise replay all.

Reference: docs/QUANTA_VDB_DEEP_DIVE.md — Remaining Gaps: WAL

**Source:** `swarm/mesh_wal.py`
**Tests:** 13/13

### `class WALCheckpoint`

Checkpoint metadata.

#### `to_dict()`

*Source: `swarm/mesh_wal.py:68`*

#### `from_dict(d)`

*Source: `swarm/mesh_wal.py:77`*

### `class WALEntry`

A single WAL operation record.

#### `to_bytes()`

Serialize to binary WAL record.

*Source: `swarm/mesh_wal.py:94`*

#### `from_bytes(data)`

Deserialize from binary WAL record.

*Source: `swarm/mesh_wal.py:116`*

### `class MeshWAL`

Write-ahead log for MeshVectorTable operations.

Parameters
----------
wal_dir : Path | str
    Directory for WAL files.
max_wal_size : int
    Maximum size of a single WAL file before rotation (bytes).
checkpoint_interval : float
    Seconds between automatic checkpoints.

#### `__init__(wal_dir, max_wal_size, checkpoint_interval)`

*Source: `swarm/mesh_wal.py:165`*

#### `append(op, payload)`

Append an operation to the WAL.

Parameters
----------
op : str
    Operation type: "insert", "merge", "delete", "checkpoint"
payload : dict
    Operation-specific payload.

Returns
-------
bool
    True if appended successfully.

*Source: `swarm/mesh_wal.py:196`*

#### `append_insert(entry_dict)`

Append an insert operation.

*Source: `swarm/mesh_wal.py:236`*

#### `append_merge(payload)`

Append a merge operation.

*Source: `swarm/mesh_wal.py:240`*

#### `append_delete(agent_id)`

Append a delete operation.

*Source: `swarm/mesh_wal.py:244`*

#### `recover(table)`

Replay WAL entries into a MeshVectorTable.

Parameters
----------
table : MeshVectorTable
    The table to recover into.

Returns
-------
dict
    Recovery stats: replayed, errors, last_checkpoint.

*Source: `swarm/mesh_wal.py:250`*

#### `checkpoint(table)`

Write a checkpoint and truncate old WAL files.

Parameters
----------
table : MeshVectorTable
    The table to checkpoint (for verification).

Returns
-------
WALCheckpoint
    The new checkpoint metadata.

*Source: `swarm/mesh_wal.py:361`*

#### `stats()`

*Source: `swarm/mesh_wal.py:420`*

#### `close()`

Close WAL and stop checkpoint thread.

*Source: `swarm/mesh_wal.py:433`*

#### `_load_checkpoint()`

Load last checkpoint from disk.

*Source: `swarm/mesh_wal.py:445`*

#### `_open_current_wal()`

Open or create the current WAL file.

*Source: `swarm/mesh_wal.py:456`*

#### `_rotate_wal()`

Rotate to a new WAL file.

*Source: `swarm/mesh_wal.py:473`*

#### `_checkpoint_loop()`

Background thread for periodic checkpoints.

*Source: `swarm/mesh_wal.py:485`*

---

## PatternMine

PatternMine — Extract operational patterns from agent-operations and apply to fleet monitoring.

Reads pattern files from the agent-operations repo (or local clone) and converts
hard-won rules into FleetMonitor alert conditions and task dispatch templates.

Patterns Mined
-------------
- **Repo Sweep Pattern**: "Batch in groups of 5. Verify output."
- **Task Prompt Pattern**: "Procedural prompts succeed at 90%+. Style guides kill agents."
- **A2A Handoff Pattern**: "HANDOFF.md is the contract. Zero-token output = silent failure."
- **Reliability Pattern**: "5 repos per task max. Separate task from style."

Usage
-----
    miner = PatternMine(repo_path="/path/to/agent-operations")
    miner.load_patterns()

    # Convert to FleetMonitor rules
    rules = miner.to_fleet_monitor_rules()
    for rule in rules:
        monitor.add_rule(rule)

    # Generate task templates
    template = miner.get_task_template("repo_sweep")

**Source:** `fleet/pattern_mine.py`
**Tests:** 23/23

### `class OperationalPattern`

A mined operational pattern with metadata.

### `class AlertRule`

A FleetMonitor-compatible alert rule.

### `class TaskTemplate`

A subagent task template derived from a pattern.

### `class PatternMine`

Mine operational patterns from agent-operations repo.

Parameters
----------
repo_path : Path | str | None
    Path to local clone of agent-operations repo.
    If None, uses default patterns (hard-coded from analysis).

#### `__init__(repo_path)`

*Source: `fleet/pattern_mine.py:176`*

#### `load_patterns()`

Load patterns from repo or use defaults.

*Source: `fleet/pattern_mine.py:184`*

#### `_load_defaults()`

Load built-in patterns.

*Source: `fleet/pattern_mine.py:192`*

#### `_load_from_repo()`

Parse markdown files in agent-operations repo.

*Source: `fleet/pattern_mine.py:198`*

#### `_extract_rules(content)`

Extract rule sentences from markdown content.

*Source: `fleet/pattern_mine.py:230`*

#### `to_fleet_monitor_rules()`

Convert patterns to FleetMonitor alert rules.

*Source: `fleet/pattern_mine.py:245`*

#### `_pattern_to_rule(pattern)`

Convert a single pattern to an alert rule.

*Source: `fleet/pattern_mine.py:258`*

#### `to_task_templates()`

Generate task templates from dispatch patterns.

*Source: `fleet/pattern_mine.py:302`*

#### `_pattern_to_template(pattern)`

Convert a dispatch pattern to a task template.

*Source: `fleet/pattern_mine.py:316`*

#### `get_task_template(name)`

Get a specific task template by name.

*Source: `fleet/pattern_mine.py:342`*

#### `generate_report()`

Generate a comprehensive pattern mining report.

*Source: `fleet/pattern_mine.py:353`*

#### `_categorize_patterns()`

Count patterns by category.

*Source: `fleet/pattern_mine.py:373`*

#### `_top_recommendations()`

Generate top recommendations from patterns.

*Source: `fleet/pattern_mine.py:380`*

#### `write_report(path)`

Write pattern mining report to markdown.

*Source: `fleet/pattern_mine.py:390`*

#### `apply_to_monitor(monitor)`

Apply mined rules to a FleetMonitor instance.

Returns list of rule names that were added.

*Source: `fleet/pattern_mine.py:441`*

---

## Pincher

Pincher — Selective data extraction for the fleet.

An emergent application that combines Quanta's VDB for fast pattern matching
with caslang's constrained queries for deterministic, sandboxed data extraction.

Use Cases
---------
- **Intelligence Gathering**: Extract relevant signals from massive telemetry streams
- **Document Mining**: Pinch specific facts from large document corpora
- **Log Analysis**: Find anomalies and patterns in distributed logs
- **Data Cleaning**: Selectively extract and transform messy data sources

The metaphor: a crab's claw — precise, selective, powerful.  It doesn't
scoop everything; it pinches exactly what matters.

Architecture
------------
The Pincher operates on a "query pipeline" abstraction:

1. **Source Adapter** — Connects to data sources (files, streams, APIs,
   Quanta VDB partitions).  Normalizes data into extractable records.

2. **Pattern Matcher** — Uses Quanta's Hnswlib ANN search or regex/glob
   patterns to identify candidate records.  Fast pre-filter.

3. **Constraint Engine** — caslang sandbox evaluates each candidate against
   precise extraction rules.  Only records that pass all constraints are
   kept.

4. **Transform Pipeline** — Extracted fields are transformed, aggregated,
   and formatted into the target output schema.

5. **Sink Adapter** — Writes results to destination (file, VDB, message bus).

Reference
---------
- Quanta VDB: https://github.com/CantorAI/Quanta
- caslang: https://github.com/xlang-foundation/caslang

**Source:** `fleet/pincher.py`
**Tests:** 14/14

### `class ExtractionResult`

A single extracted record.

#### `to_vector(dim)`

Serialize to a vector for VDB storage.

*Source: `fleet/pincher.py:78`*

### `class DataSource`

Abstract base for data sources.

#### `__iter__()`

*Source: `fleet/pincher.py:94`*

#### `close()`

*Source: `fleet/pincher.py:97`*

### `class FileSource`

Read records from a JSONL or text file.

#### `__init__(path, parser)`

*Source: `fleet/pincher.py:104`*

#### `__iter__()`

*Source: `fleet/pincher.py:108`*

#### `_default_parser(line)`

*Source: `fleet/pincher.py:120`*

### `class QuantaSource`

Query records from Quanta VDB as a data source.

#### `__init__(quanta_bridge, query_vector, k, partition)`

*Source: `fleet/pincher.py:127`*

#### `__iter__()`

*Source: `fleet/pincher.py:140`*

### `class MemorySource`

In-memory data source for testing.

#### `__init__(records)`

*Source: `fleet/pincher.py:157`*

#### `__iter__()`

*Source: `fleet/pincher.py:160`*

### `class ExtractionQuery`

Specification for a pincher extraction job.

- **patterns**: Regex or glob patterns for pre-filtering
- **constraints**: caslang script for precise validation
- **transforms**: Field extraction and transformation rules
- **output_schema**: Target field names and types

#### `compile_patterns()`

Compile regex patterns for matching.

*Source: `fleet/pincher.py:185`*

### `class Pincher`

Selective data extraction engine.

Parameters
----------
quanta_bridge : QuantaVdbBridge | None
    Optional VDB for fast vector-based pre-filtering.
caslang_executor : CaslangExecutor | None
    Optional sandbox for constraint validation.

#### `__init__(quanta_bridge, caslang_executor)`

*Source: `fleet/pincher.py:210`*

#### `extract(query, source)`

Run a pincher extraction query over a data source.

Pipeline:
1. Pre-filter with regex patterns (fast, O(n))
2. Vector similarity filter via Quanta (if available, O(log n))
3. Constraint validation via caslang sandbox (precise, O(m) per candidate)
4. Transform and format output

*Source: `fleet/pincher.py:224`*

#### `extract_to_vdb(query, source, partition_tag)`

Extract and immediately store results in Quanta VDB.

*Source: `fleet/pincher.py:302`*

#### `batch_extract(queries, sources)`

Run multiple extraction queries in parallel over multiple sources.

*Source: `fleet/pincher.py:334`*

#### `_apply_transforms(record, transforms)`

Apply field extraction and transformation rules.

*Source: `fleet/pincher.py:349`*

#### `_compute_confidence(matched_patterns, extracted_fields)`

Compute extraction confidence score.

*Source: `fleet/pincher.py:398`*

#### `stats()`

*Source: `fleet/pincher.py:414`*

---

## SceneTracker

SceneTracker — Query pattern tracking and cache optimization for MeshVectorTables.

Learns from query patterns to pre-load, cache, and optimize future queries:
- **Pattern recognition**: Detects common query patterns (by agent_id, fitness, similarity)
- **Predictive caching**: Pre-loads likely-to-be-queried vectors into hot tier
- **Query histogram**: Tracks frequency of different query types
- **Performance feedback**: Measures query latency and adapts caching strategy
- **Scene detection**: Identifies "scenes" (temporal clusters of related queries)

Use Cases
---------
- **Predictive caching**: "Agents that query agent_A also query agent_B" → pre-load B
- **Performance optimization**: Detect slow queries, promote hot vectors
- **Query routing**: Route queries to appropriate tier (hot/warm/cold) based on pattern
- **Anomaly detection**: Unusual query patterns may indicate probing or errors
- **Fleet health**: Track query volume trends over time

Architecture
------------
Scenes are temporal clusters of queries:
  Scene {
    scene_id: str
    queries: list[QueryPattern]
    start_time: float
    end_time: float
    dominant_pattern: str
    frequency: int
  }

Query patterns are hashed by (query_type, filter_type, result_size):
  QueryPattern {
    pattern_hash: str
    query_type: str  # "by_id", "by_fitness", "similarity", "knn", "range"
    filter_type: str  # "none", "fitness", "agent_id", "keyword"
    result_size: int
    latency_ms: float
    timestamp: float
  }

Cache strategy adapts based on:
- Frequency: high-frequency queries → cache results
- Recency: recently queried → keep in hot tier
- Co-occurrence: "A then B" patterns → pre-load B

Reference: docs/QUANTA_VDB_DEEP_DIVE.md — Remaining Gaps: SceneTracker

**Source:** `swarm/scene_tracker.py`
**Tests:** 10/10

### `class QueryPattern`

A single query event.

#### `pattern_hash()`

Hash of the query pattern for deduplication.

*Source: `swarm/scene_tracker.py:75`*

#### `to_dict()`

*Source: `swarm/scene_tracker.py:80`*

### `class Scene`

A temporal cluster of related queries.

#### `to_dict()`

*Source: `swarm/scene_tracker.py:101`*

### `class CacheStrategy`

Adaptive caching strategy based on query patterns.

### `class SceneTracker`

Query pattern tracking and predictive caching for MeshVectorTables.

Parameters
----------
table : MeshVectorTable
    The table to track queries for.
strategy : CacheStrategy
    Caching strategy configuration.

#### `__init__(table, strategy)`

*Source: `swarm/scene_tracker.py:134`*

#### `track_query(query_type, filter_type, result_size, latency_ms, query_params)`

Record a query event.

Parameters
----------
query_type : str
    Type of query: "by_id", "by_fitness", "similarity", "knn", "range", "all"
filter_type : str
    Filter applied: "none", "fitness", "agent_id", "keyword", "temporal"
result_size : int
    Number of results returned.
latency_ms : float
    Query latency in milliseconds.
query_params : dict
    Additional query parameters for pattern analysis.

*Source: `swarm/scene_tracker.py:165`*

#### `get_cache_recommendations()`

Get agent_ids that should be promoted to hot tier based on patterns.

Returns
-------
list[str]
    Agent IDs to promote.

*Source: `swarm/scene_tracker.py:231`*

#### `apply_cache_recommendations(tiered_storage)`

Apply cache recommendations to promote entries.

Parameters
----------
tiered_storage : TieredMeshStorage | None
    If provided, promote entries to hot tier.

Returns
-------
int
    Number of entries promoted.

*Source: `swarm/scene_tracker.py:256`*

#### `detect_scenes(max_scenes)`

Detect and return recent scenes.

Parameters
----------
max_scenes : int
    Maximum number of scenes to return.

Returns
-------
list[Scene]
    Recent scenes, newest first.

*Source: `swarm/scene_tracker.py:286`*

#### `get_current_scene()`

Get the currently active scene.

*Source: `swarm/scene_tracker.py:312`*

#### `get_latency_stats()`

Get latency statistics by query type.

Returns
-------
dict
    Latency stats per query type and overall.

*Source: `swarm/scene_tracker.py:321`*

#### `get_hot_queries(k)`

Get most frequent query patterns.

Returns
-------
list[tuple[str, int]]
    (pattern_hash, frequency) sorted descending.

*Source: `swarm/scene_tracker.py:349`*

#### `stats()`

*Source: `swarm/scene_tracker.py:363`*

#### `_update_scene(pattern)`

Update the current scene with a new query pattern.

*Source: `swarm/scene_tracker.py:379`*

#### `_update_dominant_pattern(scene)`

Recompute the dominant pattern of a scene.

*Source: `swarm/scene_tracker.py:412`*

#### `_query_rate()`

Return queries per minute in the current window.

*Source: `swarm/scene_tracker.py:423`*

---

## TMinusBridge

TMinusBridge — Python bridge to t-minus-rs Rust crate.

Wraps the t-minus binary via subprocess JSON RPC to provide:
- Cron expression parsing and next-fire computation
- Hierarchical deadline trees with parent→child inheritance
- Token bucket and leaky bucket rate limiters

Integration targets:
- nerve.distributed_metronome_bridge — deadline propagation
- fleet.fleet_conductor — rate limiting for fleet operations
- fleet.fleet_monitor — cron-based scheduled health checks

Usage
-----
    bridge = TMinusBridge()
    next_fire = bridge.cron_next("*/15 * * * *", after=0)
    remaining = bridge.deadline_remaining(parent_secs=60, child_secs=120)
    acquired = bridge.token_bucket(burst=10.0, rate=2.0, acquire=3.0)

**Source:** `fleet/t_minus_bridge.py`
**Tests:** 30/30

### `class DeadlineTree`

A hierarchical deadline tree node.

### `class RateLimiter`

Token bucket rate limiter state.

### `class CronSchedule`

Cron schedule with next fire time.

### `class TMinusBridge`

Bridge to t-minus-rs Rust library.

Parameters
----------
binary_path : Path | str | None
    Path to the t_minus_bridge binary. If None, searches in:
    1. ./bin/t_minus_bridge
    2. ../bin/t_minus_bridge
    3. $PATH

#### `__init__(binary_path)`

*Source: `fleet/t_minus_bridge.py:77`*

#### `_resolve_binary(path)`

Resolve the binary path.

*Source: `fleet/t_minus_bridge.py:81`*

#### `_call(request)`

Call the binary with a JSON request.

*Source: `fleet/t_minus_bridge.py:107`*

#### `cron_next(expr, after)`

Compute next fire time for a cron expression.

Parameters
----------
expr : str
    Cron expression (e.g., "*/15 * * * *").
after : int
    Unix timestamp to compute after. Default 0 (now).

Returns
-------
int
    Unix timestamp of next fire time.

*Source: `fleet/t_minus_bridge.py:127`*

#### `cron_schedule(expr)`

Create a CronSchedule with next fire time.

*Source: `fleet/t_minus_bridge.py:151`*

#### `deadline_remaining(parent_secs, child_secs)`

Compute remaining time in a hierarchical deadline tree.

The child inherits the parent's deadline — if parent expires in 60s
and child has 120s, the child actually has 60s remaining.

Parameters
----------
parent_secs : float
    Parent deadline in seconds.
child_secs : float
    Child deadline in seconds.

Returns
-------
float
    Remaining seconds (min of parent and child).

*Source: `fleet/t_minus_bridge.py:158`*

#### `build_deadline_tree(parent_secs, child_secs)`

Build a deadline tree and compute remaining time.

*Source: `fleet/t_minus_bridge.py:185`*

#### `token_bucket(burst, rate, acquire)`

Create a token bucket and attempt to acquire tokens.

Parameters
----------
burst : float
    Maximum token capacity.
rate : float
    Token refill rate per second.
acquire : float
    Tokens to acquire.

Returns
-------
RateLimiter
    Result with acquired flag and remaining tokens.

*Source: `fleet/t_minus_bridge.py:196`*

#### `check_rate_limit(burst, rate, acquire)`

Quick check if tokens can be acquired.

Returns
-------
bool
    True if tokens were acquired.

*Source: `fleet/t_minus_bridge.py:228`*

#### `schedule_fleet_beat(interval_mins)`

Schedule the next fleet beat using cron.

Parameters
----------
interval_mins : int
    Beat interval in minutes. Default 15.

Returns
-------
int
    Unix timestamp of next beat.

*Source: `fleet/t_minus_bridge.py:241`*

#### `propagate_deadline(parent_deadline, child_budget)`

Propagate a parent deadline to a child task.

Used by MetronomeBridge to enforce that child tasks don't
exceed parent deadlines.

Parameters
----------
parent_deadline : float
    Parent remaining seconds.
child_budget : float
    Child requested seconds.

Returns
-------
float
    Effective child budget (capped by parent).

*Source: `fleet/t_minus_bridge.py:257`*

#### `throttle_fleet_operation(ops_per_sec, burst)`

Throttle a fleet operation using token bucket.

Parameters
----------
ops_per_sec : float
    Target operations per second.
burst : int
    Burst capacity. Default 10.

Returns
-------
bool
    True if operation should proceed.

*Source: `fleet/t_minus_bridge.py:277`*

#### `is_available()`

Check if the binary is available and functional.

*Source: `fleet/t_minus_bridge.py:296`*

#### `__repr__()`

*Source: `fleet/t_minus_bridge.py:304`*

---

## TernaryTypes

TernaryTypes — Ternary logic framework for fleet signal classification.

An emergent application inspired by Market Manifold's ternary insight:
Every continuous signal in the fleet can be classified into three states:
  -1 (Negative / Reduce / Critical)
   0 (Neutral / Hold / Warning)
  +1 (Positive / Accumulate / Healthy)

This module provides ternary logic operations, signal classification,
vector operations, and consensus mechanisms for distributed fleet decisions.

Usage
-----
    from fleet.ternary_types import TernaryValue, TernaryVector, TernaryMap

    # Classify a signal
    val = TernaryMap.classify(0.75, threshold=0.5)  # +1

    # Combine signals
    combined = TernaryValue.majority([+1, +1, 0, -1])  # +1

    # Vector operations
    vec = TernaryVector([+1, 0, -1, +1])
    assert vec.hamming_weight() == 2
    assert vec.density() == 0.5

Integration Points
------------------
- FleetMonitor: health status → ternary classification
- BreedOptimizer: offspring quality → ternary risk level
- VectorSwarm: distributed ternary consensus
- CognitiveCache: query confidence → ternary prediction

**Source:** `fleet/ternary_types.py`
**Tests:** 60/60

### `class TernaryValue`

A single ternary value: -1, 0, or +1.

Ternary logic extends boolean logic with three states:
- NEG (-1): False, negative, critical, reduce
- ZERO (0): Unknown, neutral, warning, hold
- POS (+1): True, positive, healthy, accumulate

#### `from_float(value, threshold)`

Convert a float to ternary value.

Parameters
----------
value : float
    Input value.
threshold : float
    Dead zone around zero. Values in [-threshold, +threshold]
    map to ZERO.

Returns
-------
int
    -1, 0, or +1.

*Source: `fleet/ternary_types.py:64`*

#### `from_bool(value)`

Convert boolean to ternary.

*Source: `fleet/ternary_types.py:87`*

#### `not_(a)`

Ternary NOT: negates the sign, preserves zero.

*Source: `fleet/ternary_types.py:92`*

#### `and_(a, b)`

Ternary AND (minimum).

*Source: `fleet/ternary_types.py:98`*

#### `or_(a, b)`

Ternary OR (maximum).

*Source: `fleet/ternary_types.py:105`*

#### `xor_(a, b)`

Ternary XOR: strict inequality.

0 acts as pass-through (XOR with 0 returns the other value).

*Source: `fleet/ternary_types.py:112`*

#### `majority(values)`

Majority vote among ternary values.

Returns the most common non-zero value, or ZERO if tied.

*Source: `fleet/ternary_types.py:128`*

#### `consensus(values, threshold)`

Consensus vote: requires threshold fraction agreement.

Parameters
----------
values : list[int]
    Ternary values.
threshold : float
    Fraction required for consensus (0.5 = simple majority).

Returns
-------
int
    +1 or -1 if consensus reached, ZERO otherwise.

*Source: `fleet/ternary_types.py:144`*

#### `_validate(value)`

*Source: `fleet/ternary_types.py:173`*

#### `to_string(value)`

Convert ternary value to human-readable string.

*Source: `fleet/ternary_types.py:178`*

#### `to_emoji(value)`

Convert ternary value to emoji.

*Source: `fleet/ternary_types.py:185`*

### `class TernaryVector`

A vector of ternary values.

#### `__post_init__()`

*Source: `fleet/ternary_types.py:198`*

#### `__len__()`

*Source: `fleet/ternary_types.py:202`*

#### `__getitem__(idx)`

*Source: `fleet/ternary_types.py:205`*

#### `__iter__()`

*Source: `fleet/ternary_types.py:208`*

#### `hamming_weight()`

Count non-zero elements.

*Source: `fleet/ternary_types.py:211`*

#### `density()`

Fraction of non-zero elements.

*Source: `fleet/ternary_types.py:215`*

#### `balance()`

Net balance: (pos - neg) / total. Range [-1, 1].

*Source: `fleet/ternary_types.py:221`*

#### `entropy()`

Shannon entropy of the ternary distribution.

*Source: `fleet/ternary_types.py:229`*

#### `and_with(other)`

Element-wise AND with another vector.

*Source: `fleet/ternary_types.py:244`*

#### `or_with(other)`

Element-wise OR with another vector.

*Source: `fleet/ternary_types.py:250`*

#### `not_()`

Element-wise NOT.

*Source: `fleet/ternary_types.py:256`*

#### `majority()`

Majority vote across all elements.

*Source: `fleet/ternary_types.py:260`*

#### `consensus(threshold)`

Consensus vote across all elements.

*Source: `fleet/ternary_types.py:264`*

#### `to_string()`

Convert to string representation.

*Source: `fleet/ternary_types.py:268`*

#### `from_floats(floats, threshold)`

Create a TernaryVector from float values.

*Source: `fleet/ternary_types.py:273`*

#### `from_bools(bools)`

Create a TernaryVector from boolean values.

*Source: `fleet/ternary_types.py:278`*

### `class TernaryMap`

Map continuous signals to ternary classification.

#### `classify(value, threshold)`

Classify a single float value.

*Source: `fleet/ternary_types.py:287`*

#### `classify_with_zscore(value, mean, std, threshold)`

Classify using z-score: value = (x - mean) / std.

Parameters
----------
value : float
    Raw value.
mean : float
    Distribution mean.
std : float
    Distribution standard deviation.
threshold : float
    Z-score threshold for classification.

Returns
-------
int
    +1 if z-score > threshold, -1 if z-score < -threshold, 0 otherwise.

*Source: `fleet/ternary_types.py:292`*

#### `classify_vector(values, threshold)`

Classify a vector of floats.

*Source: `fleet/ternary_types.py:323`*

#### `classify_percentile(value, percentile_25, percentile_75)`

Classify using percentile thresholds.

-1 if value < p25, +1 if value > p75, 0 otherwise.

*Source: `fleet/ternary_types.py:332`*

#### `window_classify(values, window_size, threshold)`

Classify using rolling window averages.

*Source: `fleet/ternary_types.py:349`*

### `class TernaryConsensus`

Distributed consensus using ternary voting.

#### `fleet_vote(votes, threshold)`

Aggregate votes from multiple fleet nodes.

Parameters
----------
votes : dict[str, int]
    Node ID → ternary vote mapping.
threshold : float
    Consensus threshold.

Returns
-------
dict
    Result with consensus, confidence, and dissenters.

*Source: `fleet/ternary_types.py:370`*

#### `weighted_vote(votes, threshold)`

Weighted consensus vote.

Parameters
----------
votes : dict[str, tuple[int, float]]
    Node ID → (vote, weight) mapping.
threshold : float
    Consensus threshold (applied to total weight).

Returns
-------
dict
    Result with weighted consensus.

*Source: `fleet/ternary_types.py:432`*

### `class TernaryOperator`

Higher-order ternary operators.

#### `if_then_else(condition, true_val, false_val)`

Ternary if-then-else.

If condition is POS, return true_val.
If condition is NEG, return false_val.
If condition is ZERO, return the more conservative (min) of the two.

*Source: `fleet/ternary_types.py:490`*

#### `clamp(value, min_val, max_val)`

Clamp a ternary value between min and max.

*Source: `fleet/ternary_types.py:507`*

#### `switch(value, cases)`

Switch on ternary value.

*Source: `fleet/ternary_types.py:515`*

#### `cascade(values, default)`

Cascade: return first non-zero value, or default.

*Source: `fleet/ternary_types.py:521`*

---

## TieredMeshStorage

TieredMeshStorage — Hot / Warm / Cold tiers for MeshVectorTables.

Implements a three-tier storage system:
- **Hot**: In-memory with HNSW ANN index (recent + high-fitness entries)
- **Warm**: SQLite-backed with B-tree index (older, still accessible)
- **Cold**: Compressed file archives (rarely accessed, bulk retrieval)

Promotion/demotion based on:
- Recency (timestamp)
- Fitness (higher = more likely to stay hot)
- Access frequency (tracked per entry)
- Thermal pressure (hot agents get demoted to cool down)

Reference: docs/QUANTA_VDB_DEEP_DIVE.md — Phase 2: Tiered Storage

**Source:** `swarm/tiered_mesh_storage.py`
**Tests:** 7/7

### `class TierConfig`

Configuration for tiered storage thresholds.

### `class PromotionPolicy`

Rules for tier transitions.

### `class TieredMeshStorage`

Tiered storage wrapper for MeshVectorTable.

Parameters
----------
base_table : MeshVectorTable
    The underlying mesh table (becomes the "hot" tier).
db_path : Path | str
    Path to SQLite database for warm tier.
cold_path : Path | str
    Directory for cold archive files.
config : TierConfig
    Tier thresholds.
policy : PromotionPolicy
    Promotion/demotion rules.

#### `__init__(base_table, db_path, cold_path, config, policy)`

*Source: `swarm/tiered_mesh_storage.py:77`*

#### `_init_sqlite()`

Initialize SQLite schema for warm tier.

*Source: `swarm/tiered_mesh_storage.py:111`*

#### `_warm_insert(entry)`

Insert entry into warm SQLite tier.

*Source: `swarm/tiered_mesh_storage.py:139`*

#### `_warm_query(agent_id)`

Query warm tier by agent_id.

*Source: `swarm/tiered_mesh_storage.py:166`*

#### `_warm_query_by_fitness(min_fitness, max_results)`

Query warm tier by fitness threshold.

*Source: `swarm/tiered_mesh_storage.py:180`*

#### `_warm_delete(agent_id)`

Delete from warm tier.

*Source: `swarm/tiered_mesh_storage.py:200`*

#### `_warm_count_entries()`

Count entries in warm tier.

*Source: `swarm/tiered_mesh_storage.py:210`*

#### `_cold_archive(entries)`

Archive entries to a compressed file. Returns filename.

*Source: `swarm/tiered_mesh_storage.py:221`*

#### `_cold_query(agent_id)`

Query cold archives for agent_id. Slow — scans all archives.

*Source: `swarm/tiered_mesh_storage.py:237`*

#### `query(agent_id)`

Query across all tiers: hot -> warm -> cold.

*Source: `swarm/tiered_mesh_storage.py:254`*

#### `insert(entry)`

Insert into appropriate tier based on fitness/age/thermal.

*Source: `swarm/tiered_mesh_storage.py:276`*

#### `query_by_fitness(min_fitness, max_results, include_warm)`

Query across hot and warm tiers by fitness.

*Source: `swarm/tiered_mesh_storage.py:294`*

#### `get_tier_stats()`

Return statistics for each tier.

*Source: `swarm/tiered_mesh_storage.py:315`*

#### `close()`

Stop maintenance thread.

*Source: `swarm/tiered_mesh_storage.py:333`*

#### `_should_be_hot(entry, age)`

Determine if entry should be in hot tier.

*Source: `swarm/tiered_mesh_storage.py:340`*

#### `_maybe_promote(entry)`

Promote warm entry to hot if access threshold met.

*Source: `swarm/tiered_mesh_storage.py:348`*

#### `_demote_oldest_hot()`

Demote oldest/lowest-fitness hot entry to warm.

*Source: `swarm/tiered_mesh_storage.py:360`*

#### `_maintenance_loop()`

Background maintenance: demote old hot entries, archive warm to cold.

*Source: `swarm/tiered_mesh_storage.py:373`*

#### `_run_maintenance()`

Single maintenance pass.

*Source: `swarm/tiered_mesh_storage.py:382`*

#### `_vec_to_b64(vec)`

*Source: `swarm/tiered_mesh_storage.py:422`*

#### `_b64_to_vec(b64, dim)`

*Source: `swarm/tiered_mesh_storage.py:427`*

#### `_row_to_entry(row)`

Convert SQLite row to VectorTableEntry.

*Source: `swarm/tiered_mesh_storage.py:432`*

---

## VectorSwarm

VectorSwarm — Distributed search layer across multiple mesh vector tables.

Enables fleet-wide queries that span multiple nodes, shards, and tables:
- **Fan-out search**: Query dispatched to all relevant nodes/shards in parallel
- **Result aggregation**: Merge, rank, and deduplicate results from multiple sources
- **Shard routing**: Route queries to the most appropriate shards (temporal, fitness, agent_id)
- **Distributed KNN**: Approximate nearest neighbor across the entire fleet
- **Consensus ranking**: Multiple nodes vote on result ranking for resilience

Use Cases
---------
- **Fleet-wide recall**: "Find all agents similar to this vector across the entire fleet"
- **Cross-shard temporal search**: "What happened in the last hour across all nodes?"
- **Distributed breeding pool**: Select parents from all nodes, not just local
- **Pattern mining at scale**: Discover clusters across the entire fleet population
- **Anomaly detection**: Find outliers that don't match any fleet-wide pattern

Architecture
------------
Search layers:
  1. Router → determines which nodes/shards to query
  2. Dispatcher → sends queries in parallel (simulated with threads)
  3. Aggregator → merges results, deduplicates, ranks
  4. Consensus → vote-based ranking for Byzantine resilience

Query plans:
  QueryPlan {
    query_id: str
    target_nodes: list[str]
    target_shards: list[str]
    query_type: str  # "knn", "similarity", "temporal", "fitness"
    params: dict
  }

Result sets:
  SwarmResult {
    query_id: str
    source: str  # "node_id/shard_id"
    entries: list[VectorTableEntry]
    latency_ms: float
    confidence: float
  }

Reference: docs/QUANTA_VDB_DEEP_DIVE.md — Emergent Applications: VectorSwarm

**Source:** `swarm/vector_swarm.py`
**Tests:** 12/12

### `class SwarmQueryPlan`

A query plan for distributed search.

#### `required_responses()`

*Source: `swarm/vector_swarm.py:78`*

### `class SwarmResult`

Result from a single node/shard.

#### `to_dict()`

*Source: `swarm/vector_swarm.py:98`*

### `class SwarmRouter`

Routes queries to appropriate nodes and shards.

#### `__init__()`

*Source: `swarm/vector_swarm.py:112`*

#### `register_node(node_id, shard_ids, node_ref)`

Register a node with its shards.

*Source: `swarm/vector_swarm.py:116`*

#### `route_query(query_type, params)`

Create a query plan for a query.

Parameters
----------
query_type : str
    Type of query.
params : dict
    Query parameters.

Returns
-------
SwarmQueryPlan
    Query plan with target nodes and shards.

*Source: `swarm/vector_swarm.py:125`*

#### `_route_by_hash(key, n)`

Route by consistent hashing of key.

*Source: `swarm/vector_swarm.py:175`*

#### `_route_by_time(time_range)`

Route to shards that might contain the time range.

*Source: `swarm/vector_swarm.py:184`*

#### `_generate_query_id(query_type, params)`

*Source: `swarm/vector_swarm.py:193`*

### `class VectorSwarm`

Distributed search layer across mesh vector tables.

Parameters
----------
router : SwarmRouter
    Query routing component.
max_workers : int
    Max parallel threads for fan-out queries.

#### `__init__(router, max_workers)`

*Source: `swarm/vector_swarm.py:209`*

#### `query_by_id(agent_id, consistency)`

Query an agent by ID across the fleet.

Parameters
----------
agent_id : str
    Agent ID to query.
consistency : str
    "one", "quorum", or "all".

Returns
-------
list[SwarmResult]
    Results from each node.

*Source: `swarm/vector_swarm.py:223`*

#### `query_similar(vector, k, consistency)`

Find similar vectors across the fleet.

Parameters
----------
vector : np.ndarray
    Query vector.
k : int
    Number of results per node.
consistency : str
    "one", "quorum", or "all".

Returns
-------
list[SwarmResult]
    Results from each node.

*Source: `swarm/vector_swarm.py:242`*

#### `query_knn(vector, k, consistency)`

Distributed KNN search with global ranking.

Parameters
----------
vector : np.ndarray
    Query vector.
k : int
    Total number of results to return.
consistency : str
    "one", "quorum", or "all".

Returns
-------
list[tuple[VectorTableEntry, float]]
    Globally ranked results with distances.

*Source: `swarm/vector_swarm.py:268`*

#### `query_fitness_range(min_fitness, max_fitness, consistency)`

Query entries in a fitness range across the fleet.

Parameters
----------
min_fitness : float
    Minimum fitness.
max_fitness : float
    Maximum fitness.
consistency : str
    "one", "quorum", or "all".

Returns
-------
list[SwarmResult]
    Results from each node.

*Source: `swarm/vector_swarm.py:311`*

#### `consensus_rank(results, vector)`

Rank results by consensus voting across nodes.

Each node votes for its top results. Final rank is by
vote count, then by distance to query vector.

Parameters
----------
results : list[SwarmResult]
    Results from multiple nodes.
vector : np.ndarray
    Query vector for tie-breaking.

Returns
-------
list[tuple[VectorTableEntry, float]]
    Consensus-ranked results.

*Source: `swarm/vector_swarm.py:342`*

#### `stats()`

*Source: `swarm/vector_swarm.py:385`*

#### `_execute_plan(plan)`

Execute a query plan across target nodes.

*Source: `swarm/vector_swarm.py:400`*

#### `_query_node(node_id, node_ref, plan)`

Execute a query on a single node.

*Source: `swarm/vector_swarm.py:452`*

#### `_query_shard(node_ref, shard_id, plan)`

Execute a query on a single shard.

*Source: `swarm/vector_swarm.py:486`*

---

## xLangAgentBridge

XlangAgentBridge — Bridge between sunset-ecosystem and xlang/xMind runtime.

Provides:
- Python → xlang module importing (GIL-bypassed C++ interop)
- JSON agent graph → xMind YAML blueprint conversion
- Session memory synchronization between our system and xMind AgentFlow
- Distributed execution via xlang's LRPC IPC layer

Architecture
------------
The bridge has three layers:

1. **Runtime Layer** — lazy-loaded xlang C++ engine via `import xlang`.
   Handles module loading, IPC connections, and tensor marshalling.

2. **AgentFlow Layer** — converts our `json_agent_graph.py` JSON graphs
   into xMind's YAML blueprints (nodes, pins, actions, agents).

3. **Session Layer** — bidirectional session memory sync.  Our fleet
   sessions (agent_id, context, history) map to xMind's session IDs.

Reference
---------
- xlang runtime: https://github.com/xlang-foundation/xlang
- xMind AgentFlow: https://github.com/xlang-foundation/xMind/AgentFlow.md
- xlang IPC: https://github.com/xlang-foundation/xlang/Docs/DISTRIBUTED.md

**Source:** `fleet/xlang_agent_bridge.py`
**Tests:** 16/16

### `class AgentFlowBlueprint`

xMind AgentFlow YAML blueprint, generated from a JSON agent graph.

Nodes: function | action | agent
Pins:  input / output interfaces with X::Value data

#### `to_yaml()`

*Source: `fleet/xlang_agent_bridge.py:66`*

#### `from_json_graph(graph, name)`

Convert a sunset-ecosystem JSON agent graph to xMind YAML.

JSON graph format (from json_agent_graph.py):
{
    "nodes": [{"id": "n1", "type": "agent", "config": {...}}],
    "edges": [{"source": "n1", "target": "n2", "relation": "delegates"}]
}

*Source: `fleet/xlang_agent_bridge.py:81`*

### `class SessionSyncAdapter`

Bidirectional session memory sync between fleet and xMind.

#### `to_xmind_payload()`

Serialize fleet context for xMind session binding.

*Source: `fleet/xlang_agent_bridge.py:137`*

#### `from_xmind_payload(payload)`

Update fleet context from xMind session output.

*Source: `fleet/xlang_agent_bridge.py:150`*

### `class XlangAgentBridge`

Bridge to the xlang runtime and xMind AgentFlow framework.

Parameters
----------
node_id : str
    Fleet node identifier.
xmind_path : Path | str | None
    Path to xMind installation (for local embedding mode).
lrpc_endpoint : str | None
    Remote LRPC endpoint for distributed mode (e.g. "lrpc:9090").

#### `__init__(node_id, xmind_path, lrpc_endpoint)`

*Source: `fleet/xlang_agent_bridge.py:173`*

#### `_load_xlang()`

Lazy-import the xlang C++ runtime.

*Source: `fleet/xlang_agent_bridge.py:196`*

#### `_load_xmind()`

Lazy-import the xMind AgentFlow framework.

*Source: `fleet/xlang_agent_bridge.py:209`*

#### `convert_graph(json_graph, name)`

Convert a JSON agent graph to xMind YAML blueprint.

*Source: `fleet/xlang_agent_bridge.py:229`*

#### `save_blueprint(name, path)`

Save a blueprint to a YAML file.

*Source: `fleet/xlang_agent_bridge.py:236`*

#### `load_blueprint(path)`

Load a blueprint from a YAML file.

*Source: `fleet/xlang_agent_bridge.py:244`*

#### `create_session(session_id, context)`

Create a new session bridge between fleet and xMind.

*Source: `fleet/xlang_agent_bridge.py:259`*

#### `sync_session_to_xmind(session_id)`

Push fleet session state to xMind.

*Source: `fleet/xlang_agent_bridge.py:269`*

#### `sync_session_from_xmind(session_id)`

Pull xMind session state back to fleet.

*Source: `fleet/xlang_agent_bridge.py:291`*

#### `execute_remote(blueprint_name, session_id)`

Execute a blueprint on a remote xlang node via LRPC.

*Source: `fleet/xlang_agent_bridge.py:312`*

#### `execute_local(blueprint_name, session_id, inputs)`

Execute a blueprint locally via Python fallback (no xlang required).

*Source: `fleet/xlang_agent_bridge.py:339`*

#### `stats()`

*Source: `fleet/xlang_agent_bridge.py:399`*

---
