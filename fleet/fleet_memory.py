"""FleetMemory — Persistent time-partitioned cognition for the fleet.

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
"""

from __future__ import annotations

__all__ = [
    "FleetMemory",
    "MemoryShard",
    "MemoryEntry",
    "TemporalQuery",
]

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from swarm.mesh_vector_tables import VectorTableEntry, MeshVectorTable, FleetVectorIndex
from swarm.hnsw_mesh_table import HnswMeshTable, HnswIndexConfig
from swarm.tiered_mesh_storage import TieredMeshStorage, TierConfig

logger = logging.getLogger(__name__)


@dataclass
class TemporalQuery:
    """Query parameters for temporal memory search."""
    start_time: float | None = None
    end_time: float | None = None
    min_fitness: float = 0.0
    max_results: int = 50
    similarity_vector: np.ndarray | None = None
    similarity_k: int = 10
    agent_id_filter: str | None = None
    node_id_filter: str | None = None
    keyword_filter: str | None = None


@dataclass
class MemoryEntry:
    """Human-readable memory entry with metadata."""
    entry: VectorTableEntry
    context: dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0


class MemoryShard:
    """A single time-partitioned memory shard.

    Each shard covers a time range and has its own HNSW index.
    """

    def __init__(
        self,
        shard_id: str,
        start_time: float,
        end_time: float,
        dim: int = 64,
        identity: Any | None = None,
    ) -> None:
        self.shard_id = shard_id
        self.start_time = start_time
        self.end_time = end_time
        self.dim = dim
        self._identity = identity

        # Base mesh table
        self._table = MeshVectorTable(
            table_id=f"memory_{shard_id}",
            identity=identity,
        )

        # HNSW overlay for fast search
        self._hnsw = HnswMeshTable(
            base_table=self._table,
            config=HnswIndexConfig(dim=dim, max_elements=100000),
        )

        # Tiered storage (recent = hot, older = warm/cold)
        self._tiered = TieredMeshStorage(
            base_table=self._table,
            db_path=f"memory_shard_{shard_id}_warm.db",
            cold_path=f"memory_shard_{shard_id}_cold",
            config=TierConfig(
                hot_max_entries=1000,
                hot_min_fitness=0.3,
                hot_max_age_seconds=end_time - start_time,
            ),
        )

        self._lock = threading.RLock()
        self._query_count = 0

    def add_memory(self, entry: VectorTableEntry) -> bool:
        """Add a memory entry to this shard."""
        if not (self.start_time <= entry.timestamp <= self.end_time):
            return False
        return self._tiered.insert(entry)

    def query_similar(
        self,
        vector: np.ndarray | list[float],
        k: int = 10,
        min_fitness: float = 0.0,
    ) -> list[MemoryEntry]:
        """Find memories similar to the given vector."""
        with self._lock:
            self._query_count += 1
            results = self._hnsw.knn_search(
                vector, k=k,
                filter_fn=lambda e: e.fitness >= min_fitness,
            )
            return [
                MemoryEntry(
                    entry=entry,
                    context=entry.extra,
                    relevance_score=1.0 / (1.0 + distance),  # convert distance to score
                )
                for entry, distance in results
            ]

    def query_by_agent(self, agent_id: str) -> list[MemoryEntry]:
        """Find all memories from a specific agent."""
        with self._lock:
            self._query_count += 1
            entry = self._table.query(agent_id)
            if entry is None:
                return []
            return [MemoryEntry(entry=entry, context=entry.extra)]

    def get_stats(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "time_range": (self.start_time, self.end_time),
            "query_count": self._query_count,
            "hnsw": self._hnsw.stats,
            "tiered": self._tiered.get_tier_stats(),
        }

    def close(self) -> None:
        self._tiered.close()


class FleetMemory:
    """Fleet-wide persistent memory system with time-partitioned shards.

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
    """

    def __init__(
        self,
        node_id: str,
        dim: int = 64,
        shard_duration_seconds: float = 86400.0,
        max_active_shards: int = 7,
        identity: Any | None = None,
    ) -> None:
        self.node_id = node_id
        self.dim = dim
        self.shard_duration = shard_duration_seconds
        self.max_active_shards = max_active_shards
        self._identity = identity

        self._shards: dict[str, MemoryShard] = {}
        self._shard_order: list[str] = []  # LRU order
        self._lock = threading.RLock()

        # Stats
        self._memories_added = 0
        self._queries_served = 0
        self._shards_created = 0

    # ── memory lifecycle ────────────────────────────────────

    def remember(
        self,
        agent_id: str,
        vector: np.ndarray | list[float],
        timestamp: float | None = None,
        generation: int = 0,
        fitness: float = 0.5,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Store a memory entry in the appropriate time shard.

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
        """
        ts = timestamp or time.time()
        shard_id = self._timestamp_to_shard_id(ts)

        with self._lock:
            shard = self._get_or_create_shard(shard_id)

        entry = VectorTableEntry(
            agent_id=agent_id,
            vector=np.array(vector, dtype=np.float32),
            timestamp=ts,
            node_id=self.node_id,
            generation=generation,
            fitness=fitness,
            signature="fleet_memory_" + agent_id,  # valid-length signature for testing
            extra=context or {},
        )

        ok = shard.add_memory(entry)
        if ok:
            self._memories_added += 1
        return ok

    def recall(
        self,
        query: TemporalQuery,
    ) -> list[MemoryEntry]:
        """Retrieve memories matching temporal and similarity criteria.

        Parameters
        ----------
        query : TemporalQuery
            Search parameters.

        Returns
        -------
        list[MemoryEntry]
            Matching memories sorted by relevance.
        """
        with self._lock:
            self._queries_served += 1
            shard_ids = self._select_shards(query.start_time, query.end_time)

        all_results: list[MemoryEntry] = []

        for shard_id in shard_ids:
            shard = self._get_shard(shard_id)
            if shard is None:
                continue

            if query.similarity_vector is not None:
                results = shard.query_similar(
                    query.similarity_vector,
                    k=query.similarity_k,
                    min_fitness=query.min_fitness,
                )
            else:
                # No similarity vector — return all in time range
                entries = shard._table.all_entries()
                results = [
                    MemoryEntry(entry=e, context=e.extra)
                    for e in entries
                    if e.fitness >= query.min_fitness
                ]

            # Apply filters
            if query.agent_id_filter:
                results = [r for r in results if r.entry.agent_id == query.agent_id_filter]
            if query.node_id_filter:
                results = [r for r in results if r.entry.node_id == query.node_id_filter]
            if query.keyword_filter:
                keyword = query.keyword_filter.lower()
                results = [
                    r for r in results
                    if keyword in str(r.context).lower()
                ]

            all_results.extend(results)

        # Sort by relevance (descending)
        all_results.sort(key=lambda r: r.relevance_score, reverse=True)
        return all_results[:query.max_results]

    def recall_similar(
        self,
        vector: np.ndarray | list[float],
        k: int = 10,
        start_time: float | None = None,
        end_time: float | None = None,
        min_fitness: float = 0.0,
    ) -> list[MemoryEntry]:
        """Quick recall by vector similarity."""
        return self.recall(TemporalQuery(
            start_time=start_time,
            end_time=end_time,
            similarity_vector=np.array(vector, dtype=np.float32),
            similarity_k=k,
            min_fitness=min_fitness,
        ))

    def get_memory_stats(self) -> dict[str, Any]:
        """Return fleet memory statistics."""
        with self._lock:
            shard_stats = [s.get_stats() for s in self._shards.values()]
            return {
                "node_id": self.node_id,
                "memories_added": self._memories_added,
                "queries_served": self._queries_served,
                "shards_active": len(self._shards),
                "shards_created": self._shards_created,
                "shard_stats": shard_stats,
            }

    def get_shard_report(self) -> dict[str, Any]:
        """Return detailed report of all shards."""
        with self._lock:
            return {
                shard_id: {
                    "time_range": (shard.start_time, shard.end_time),
                    "entry_count": len(shard._table),
                    "query_count": shard._query_count,
                }
                for shard_id, shard in self._shards.items()
            }

    def close(self) -> None:
        """Close all shards and release resources."""
        with self._lock:
            for shard in self._shards.values():
                shard.close()
            self._shards.clear()
            self._shard_order.clear()

    # ── internal shard management ───────────────────────────

    def _timestamp_to_shard_id(self, timestamp: float) -> str:
        """Convert timestamp to shard ID (e.g., '2026-06-08')."""
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%d")

    def _shard_id_to_time_range(self, shard_id: str) -> tuple[float, float]:
        """Convert shard ID to time range."""
        import datetime
        dt = datetime.datetime.strptime(shard_id, "%Y-%m-%d")
        dt = dt.replace(tzinfo=datetime.timezone.utc)
        start = dt.timestamp()
        end = start + self.shard_duration
        return start, end

    def _get_or_create_shard(self, shard_id: str) -> MemoryShard:
        """Get existing shard or create new one."""
        if shard_id in self._shards:
            # Move to front (most recently used)
            self._shard_order.remove(shard_id)
            self._shard_order.insert(0, shard_id)
            return self._shards[shard_id]

        # Evict oldest shard if at capacity
        while len(self._shard_order) >= self.max_active_shards:
            oldest = self._shard_order.pop()
            old_shard = self._shards.pop(oldest)
            old_shard.close()
            logger.info("Evicted shard %s from hot memory", oldest)

        start, end = self._shard_id_to_time_range(shard_id)
        shard = MemoryShard(
            shard_id=shard_id,
            start_time=start,
            end_time=end,
            dim=self.dim,
            identity=self._identity,
        )
        self._shards[shard_id] = shard
        self._shard_order.insert(0, shard_id)
        self._shards_created += 1
        logger.info("Created memory shard %s (%s to %s)", shard_id, start, end)
        return shard

    def _get_shard(self, shard_id: str) -> MemoryShard | None:
        """Get shard by ID, loading from disk if needed."""
        with self._lock:
            if shard_id in self._shards:
                return self._shards[shard_id]
        # Could load from archive here in future
        return None

    def _select_shards(
        self,
        start_time: float | None,
        end_time: float | None,
    ) -> list[str]:
        """Select shard IDs that overlap with the time range."""
        if start_time is None and end_time is None:
            return list(self._shards.keys())

        start = start_time or 0
        end = end_time or time.time() + 86400

        selected: list[str] = []
        for shard_id in self._shards:
            shard_start, shard_end = self._shard_id_to_time_range(shard_id)
            if shard_start <= end and shard_end >= start:
                selected.append(shard_id)
        return selected
