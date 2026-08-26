"""VectorSwarm — Distributed search layer across multiple mesh vector tables.

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
"""

from __future__ import annotations

__all__ = ["VectorSwarm", "SwarmQueryPlan", "SwarmResult", "SwarmRouter"]

import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from swarm.mesh_vector_tables import VectorTableEntry

logger = logging.getLogger(__name__)


@dataclass
class SwarmQueryPlan:
    """A query plan for distributed search."""

    query_id: str
    query_type: str  # "knn", "similarity", "temporal", "fitness", "id"
    target_nodes: list[str] = field(default_factory=list)
    target_shards: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    timeout_ms: float = 5000.0
    consistency: str = "quorum"  # "one", "quorum", "all"

    @property
    def required_responses(self) -> int:
        n = len(self.target_nodes)
        if self.consistency == "one":
            return 1
        elif self.consistency == "quorum":
            return max(1, n // 2 + 1)
        else:  # all
            return n


@dataclass
class SwarmResult:
    """Result from a single node/shard."""

    query_id: str
    source: str
    entries: list[VectorTableEntry]
    latency_ms: float
    confidence: float = 1.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "source": self.source,
            "entry_count": len(self.entries),
            "latency_ms": self.latency_ms,
            "confidence": self.confidence,
            "error": self.error,
        }


class SwarmRouter:
    """Routes queries to appropriate nodes and shards."""

    def __init__(self) -> None:
        self._node_index: dict[str, Any] = {}  # node_id -> node info
        self._shard_index: dict[str, list[str]] = {}  # shard_id -> node_ids

    def register_node(self, node_id: str, shard_ids: list[str], node_ref: Any) -> None:
        """Register a node with its shards."""
        self._node_index[node_id] = {
            "node_ref": node_ref,
            "shard_ids": shard_ids,
        }
        for shard_id in shard_ids:
            self._shard_index.setdefault(shard_id, []).append(node_id)

    def route_query(self, query_type: str, params: dict[str, Any]) -> SwarmQueryPlan:
        """Create a query plan for a query.

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
        """
        query_id = self._generate_query_id(query_type, params)

        # Determine target nodes based on query type
        if query_type == "id":
            # For ID lookups, we don't know which node has the agent
            # Broadcast to all nodes (or hash-based for writes)
            target_nodes = list(self._node_index.keys())
        elif query_type == "temporal":
            # Route by time range → all nodes (fleet-wide temporal)
            target_nodes = list(self._node_index.keys())
        else:
            # Default: all nodes
            target_nodes = list(self._node_index.keys())

        # Determine target shards
        target_shards = []
        if "shard_id" in params:
            target_shards = [params["shard_id"]]
        elif "time_range" in params:
            # Route to time-relevant shards
            target_shards = self._route_by_time(params["time_range"])
        else:
            # All shards on target nodes
            for node_id in target_nodes:
                target_shards.extend(self._node_index[node_id].get("shard_ids", []))
            target_shards = list(set(target_shards))

        return SwarmQueryPlan(
            query_id=query_id,
            query_type=query_type,
            target_nodes=target_nodes,
            target_shards=target_shards,
            params=params,
        )

    def _route_by_hash(self, key: str, n: int = 1) -> list[str]:
        """Route by consistent hashing of key."""
        nodes = list(self._node_index.keys())
        if not nodes:
            return []
        hash_val = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        idx = hash_val % len(nodes)
        return (
            [nodes[idx]]
            if n == 1
            else [nodes[(idx + i) % len(nodes)] for i in range(min(n, len(nodes)))]
        )

    def _route_by_time(self, time_range: tuple[float, float]) -> list[str]:
        """Route to shards that might contain the time range."""
        # Simplified: all shards (would use shard metadata in production)
        all_shards = []
        for node_info in self._node_index.values():
            all_shards.extend(node_info.get("shard_ids", []))
        return list(set(all_shards))

    @staticmethod
    def _generate_query_id(query_type: str, params: dict[str, Any]) -> str:
        key = f"{query_type}:{sorted(params.items())}:{time.time()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


class VectorSwarm:
    """Distributed search layer across mesh vector tables.

    Parameters
    ----------
    router : SwarmRouter
        Query routing component.
    max_workers : int
        Max parallel threads for fan-out queries.
    """

    def __init__(
        self,
        router: SwarmRouter,
        max_workers: int = 4,
    ) -> None:
        self.router = router
        self.max_workers = max_workers
        self._query_count = 0
        self._total_latency_ms = 0.0
        self._success_count = 0
        self._failure_count = 0

    # ── distributed queries ─────────────────────────────────────

    def query_by_id(
        self, agent_id: str, consistency: str = "quorum"
    ) -> list[SwarmResult]:
        """Query an agent by ID across the fleet.

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
        """
        plan = self.router.route_query("id", {"agent_id": agent_id})
        plan.consistency = consistency
        return self._execute_plan(plan)

    def query_similar(
        self,
        vector: np.ndarray,
        k: int = 5,
        consistency: str = "quorum",
    ) -> list[SwarmResult]:
        """Find similar vectors across the fleet.

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
        """
        plan = self.router.route_query(
            "similarity", {"vector": vector.tolist(), "k": k}
        )
        plan.consistency = consistency
        return self._execute_plan(plan)

    def query_knn(
        self,
        vector: np.ndarray,
        k: int = 5,
        consistency: str = "all",
    ) -> list[tuple[VectorTableEntry, float]]:
        """Distributed KNN search with global ranking.

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
        """
        plan = self.router.route_query("knn", {"vector": vector.tolist(), "k": k})
        plan.consistency = consistency
        results = self._execute_plan(plan)

        # Aggregate and rank globally
        all_entries: list[tuple[VectorTableEntry, float]] = []
        for result in results:
            for entry in result.entries:
                dist = float(np.linalg.norm(entry.vector - vector))
                all_entries.append((entry, dist))

        # Sort by distance, deduplicate by agent_id
        seen: set[str] = set()
        ranked: list[tuple[VectorTableEntry, float]] = []
        for entry, dist in sorted(all_entries, key=lambda x: x[1]):
            if entry.agent_id not in seen:
                seen.add(entry.agent_id)
                ranked.append((entry, dist))

        return ranked[:k]

    def query_fitness_range(
        self,
        min_fitness: float,
        max_fitness: float = 1.0,
        consistency: str = "all",
    ) -> list[SwarmResult]:
        """Query entries in a fitness range across the fleet.

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
        """
        plan = self.router.route_query(
            "fitness",
            {"min_fitness": min_fitness, "max_fitness": max_fitness},
        )
        plan.consistency = consistency
        return self._execute_plan(plan)

    # ── consensus ranking ───────────────────────────────────────

    def consensus_rank(
        self,
        results: list[SwarmResult],
        vector: np.ndarray,
    ) -> list[tuple[VectorTableEntry, float]]:
        """Rank results by consensus voting across nodes.

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
        """
        vote_counts: dict[str, int] = {}
        entry_map: dict[str, VectorTableEntry] = {}

        for result in results:
            for entry in result.entries:
                vote_counts[entry.agent_id] = vote_counts.get(entry.agent_id, 0) + 1
                entry_map[entry.agent_id] = entry

        # Rank by vote count, then by distance
        ranked = []
        for agent_id, votes in sorted(
            vote_counts.items(), key=lambda x: x[1], reverse=True
        ):
            entry = entry_map[agent_id]
            dist = float(np.linalg.norm(entry.vector - vector))
            ranked.append((entry, dist, votes))

        # Return (entry, distance) tuples
        return [(entry, dist) for entry, dist, _ in ranked]

    # ── stats ─────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "query_count": self._query_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "mean_latency_ms": (
                self._total_latency_ms / self._query_count
                if self._query_count > 0
                else 0.0
            ),
            "node_count": len(self.router._node_index),
            "shard_count": len(self.router._shard_index),
        }

    # ── internal ────────────────────────────────────────────

    def _execute_plan(self, plan: SwarmQueryPlan) -> list[SwarmResult]:
        """Execute a query plan across target nodes."""
        start_time = time.time()
        self._query_count += 1

        results: list[SwarmResult] = []
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for node_id in plan.target_nodes:
                node_info = self.router._node_index.get(node_id)
                if node_info and "node_ref" in node_info:
                    future = executor.submit(
                        self._query_node,
                        node_id,
                        node_info["node_ref"],
                        plan,
                    )
                    futures[future] = node_id

            for future in as_completed(futures, timeout=plan.timeout_ms / 1000.0):
                node_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    self._success_count += 1
                except Exception as exc:
                    logger.warning("Query failed for node %s: %s", node_id, exc)
                    self._failure_count += 1
                    results.append(
                        SwarmResult(
                            query_id=plan.query_id,
                            source=node_id,
                            entries=[],
                            latency_ms=(time.time() - start_time) * 1000,
                            error=str(exc),
                        )
                    )

                # Early exit if we have enough responses
                if completed >= plan.required_responses:
                    # Cancel remaining futures
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break

        latency_ms = (time.time() - start_time) * 1000
        self._total_latency_ms += latency_ms

        return results

    def _query_node(
        self, node_id: str, node_ref: Any, plan: SwarmQueryPlan
    ) -> SwarmResult:
        """Execute a query on a single node."""
        start_time = time.time()

        entries: list[VectorTableEntry] = []
        try:
            # Determine which shards to query on this node
            node_shards = self.router._node_index.get(node_id, {}).get("shard_ids", [])
            target_shards = [s for s in plan.target_shards if s in node_shards]
            if not target_shards:
                target_shards = node_shards  # fallback to all node shards

            # Execute query on each shard
            for shard_id in target_shards:
                shard_entries = self._query_shard(node_ref, shard_id, plan)
                entries.extend(shard_entries)

            latency_ms = (time.time() - start_time) * 1000
            return SwarmResult(
                query_id=plan.query_id,
                source=f"{node_id}/{','.join(target_shards)}",
                entries=entries,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.time() - start_time) * 1000
            return SwarmResult(
                query_id=plan.query_id,
                source=node_id,
                entries=[],
                latency_ms=latency_ms,
                error=str(exc),
            )

    def _query_shard(
        self, node_ref: Any, shard_id: str, plan: SwarmQueryPlan
    ) -> list[VectorTableEntry]:
        """Execute a query on a single shard."""
        # In production, this would call the node's shard API
        # For simulation, we check if node_ref has the shard method
        if hasattr(node_ref, "query_shard"):
            return node_ref.query_shard(shard_id, plan)

        # Fallback: try common query patterns on the node_ref
        if plan.query_type == "id":
            agent_id = plan.params.get("agent_id")
            if agent_id and hasattr(node_ref, "query"):
                entry = node_ref.query(agent_id)
                return [entry] if entry else []

        elif plan.query_type == "similarity":
            vector = np.array(plan.params.get("vector", []), dtype=np.float32)
            k = plan.params.get("k", 5)
            if hasattr(node_ref, "query_similar"):
                return node_ref.query_similar(vector, k)
            elif hasattr(node_ref, "all_entries"):
                entries = list(node_ref.all_entries())
                # Sort by distance
                distances = [
                    (float(np.linalg.norm(e.vector - vector)), e) for e in entries
                ]
                sorted_entries = [e for _, e in sorted(distances, key=lambda x: x[0])]
                return sorted_entries[:k]

        elif plan.query_type == "fitness":
            min_f = plan.params.get("min_fitness", 0.0)
            max_f = plan.params.get("max_fitness", 1.0)
            if hasattr(node_ref, "query_by_fitness"):
                return node_ref.query_by_fitness(min_f, max_f)
            elif hasattr(node_ref, "all_entries"):
                return [
                    e for e in node_ref.all_entries() if min_f <= e.fitness <= max_f
                ]

        elif plan.query_type == "knn":
            vector = np.array(plan.params.get("vector", []), dtype=np.float32)
            k = plan.params.get("k", 5)
            if hasattr(node_ref, "query_similar"):
                return node_ref.query_similar(vector, k)
            elif hasattr(node_ref, "all_entries"):
                entries = list(node_ref.all_entries())
                distances = [
                    (float(np.linalg.norm(e.vector - vector)), e) for e in entries
                ]
                sorted_entries = [e for _, e in sorted(distances, key=lambda x: x[0])]
                return sorted_entries[:k]

        return []
