"""SceneTracker — Query pattern tracking and cache optimization for MeshVectorTables.

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
"""

from __future__ import annotations

__all__ = ["SceneTracker", "QueryPattern", "Scene", "CacheStrategy"]

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QueryPattern:
    """A single query event."""
    query_type: str  # "by_id", "by_fitness", "similarity", "knn", "range", "all"
    filter_type: str  # "none", "fitness", "agent_id", "keyword", "temporal"
    result_size: int
    latency_ms: float
    timestamp: float
    query_params: dict[str, Any] = field(default_factory=dict)

    @property
    def pattern_hash(self) -> str:
        """Hash of the query pattern for deduplication."""
        key = f"{self.query_type}:{self.filter_type}:{self.result_size}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "filter_type": self.filter_type,
            "result_size": self.result_size,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "pattern_hash": self.pattern_hash,
        }


@dataclass
class Scene:
    """A temporal cluster of related queries."""
    scene_id: str
    queries: list[QueryPattern] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    dominant_pattern: str = ""
    frequency: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "query_count": len(self.queries),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time,
            "dominant_pattern": self.dominant_pattern,
            "frequency": self.frequency,
        }


@dataclass
class CacheStrategy:
    """Adaptive caching strategy based on query patterns."""
    hot_threshold_accesses: int = 3
    hot_threshold_latency_ms: float = 50.0
    preload_cooccurrence_count: int = 2
    scene_timeout_seconds: float = 60.0
    max_cache_entries: int = 100


class SceneTracker:
    """Query pattern tracking and predictive caching for MeshVectorTables.

    Parameters
    ----------
    table : MeshVectorTable
        The table to track queries for.
    strategy : CacheStrategy
        Caching strategy configuration.
    """

    def __init__(
        self,
        table: Any,
        strategy: CacheStrategy | None = None,
    ) -> None:
        self.table = table
        self.strategy = strategy or CacheStrategy()

        # Query history
        self._queries: list[QueryPattern] = []
        self._pattern_histogram: dict[str, int] = defaultdict(int)
        self._latency_by_type: dict[str, list[float]] = defaultdict(list)
        self._query_count = 0
        self._total_latency_ms = 0.0

        # Scene detection
        self._scenes: list[Scene] = []
        self._current_scene: Scene | None = None
        self._scene_counter = 0

        # Predictive caching
        self._cooccurrence_map: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._access_counts: dict[str, int] = defaultdict(int)
        self._cached_entries: set[str] = set()

        # Performance tracking
        self._last_query_time: float = 0.0
        self._query_rate_window: list[float] = []

    # ── query tracking ────────────────────────────────────────

    def track_query(
        self,
        query_type: str,
        filter_type: str = "none",
        result_size: int = 0,
        latency_ms: float = 0.0,
        query_params: dict[str, Any] | None = None,
    ) -> None:
        """Record a query event.

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
        """
        pattern = QueryPattern(
            query_type=query_type,
            filter_type=filter_type,
            result_size=result_size,
            latency_ms=latency_ms,
            timestamp=time.time(),
            query_params=query_params or {},
        )

        self._queries.append(pattern)
        self._pattern_histogram[pattern.pattern_hash] += 1
        self._latency_by_type[query_type].append(latency_ms)
        self._query_count += 1
        self._total_latency_ms += latency_ms
        self._last_query_time = pattern.timestamp

        # Track access counts for predictive caching
        if query_type == "by_id" and query_params:
            agent_id = query_params.get("agent_id")
            if agent_id:
                self._access_counts[agent_id] += 1

        # Track co-occurrence for preloading
        if len(self._queries) >= 2:
            prev = self._queries[-2]
            curr = self._queries[-1]
            if prev.query_type == "by_id" and curr.query_type == "by_id":
                prev_id = prev.query_params.get("agent_id", "")
                curr_id = curr.query_params.get("agent_id", "")
                if prev_id and curr_id and prev_id != curr_id:
                    self._cooccurrence_map[prev_id][curr_id] += 1

        # Scene detection
        self._update_scene(pattern)

        # Query rate tracking
        self._query_rate_window.append(pattern.timestamp)
        # Prune old entries (keep last 60 seconds)
        cutoff = pattern.timestamp - 60.0
        self._query_rate_window = [t for t in self._query_rate_window if t > cutoff]

    # ── predictive caching ───────────────────────────────────

    def get_cache_recommendations(self) -> list[str]:
        """Get agent_ids that should be promoted to hot tier based on patterns.

        Returns
        -------
        list[str]
            Agent IDs to promote.
        """
        recommendations: list[str] = []

        # High-access entries
        for agent_id, count in self._access_counts.items():
            if count >= self.strategy.hot_threshold_accesses:
                recommendations.append(agent_id)

        # Co-occurrence preloading
        for agent_id, cooccurs in self._cooccurrence_map.items():
            if self._access_counts.get(agent_id, 0) >= self.strategy.hot_threshold_accesses:
                for co_id, count in cooccurs.items():
                    if count >= self.strategy.preload_cooccurrence_count:
                        recommendations.append(co_id)

        # Deduplicate
        return list(dict.fromkeys(recommendations))[:self.strategy.max_cache_entries]

    def apply_cache_recommendations(self, tiered_storage: Any | None = None) -> int:
        """Apply cache recommendations to promote entries.

        Parameters
        ----------
        tiered_storage : TieredMeshStorage | None
            If provided, promote entries to hot tier.

        Returns
        -------
        int
            Number of entries promoted.
        """
        recommendations = self.get_cache_recommendations()
        promoted = 0

        for agent_id in recommendations:
            if tiered_storage is not None:
                # Query to trigger promotion
                tiered_storage.query(agent_id)
                promoted += 1
            else:
                # Just mark as cached
                self._cached_entries.add(agent_id)
                promoted += 1

        return promoted

    # ── scene detection ───────────────────────────────────────

    def detect_scenes(self, max_scenes: int = 10) -> list[Scene]:
        """Detect and return recent scenes.

        Parameters
        ----------
        max_scenes : int
            Maximum number of scenes to return.

        Returns
        -------
        list[Scene]
            Recent scenes, newest first.
        """
        # Close current scene if old
        if self._current_scene is not None:
            if time.time() - self._current_scene.end_time > self.strategy.scene_timeout_seconds:
                self._scenes.append(self._current_scene)
                self._current_scene = None

        # Return recent scenes
        all_scenes = self._scenes.copy()
        if self._current_scene is not None:
            all_scenes.append(self._current_scene)

        return sorted(all_scenes, key=lambda s: s.end_time, reverse=True)[:max_scenes]

    def get_current_scene(self) -> Scene | None:
        """Get the currently active scene."""
        if self._current_scene is not None:
            if time.time() - self._current_scene.end_time <= self.strategy.scene_timeout_seconds:
                return self._current_scene
        return None

    # ── performance metrics ───────────────────────────────────

    def get_latency_stats(self) -> dict[str, Any]:
        """Get latency statistics by query type.

        Returns
        -------
        dict
            Latency stats per query type and overall.
        """
        stats: dict[str, Any] = {}
        for query_type, latencies in self._latency_by_type.items():
            if latencies:
                stats[query_type] = {
                    "count": len(latencies),
                    "mean_ms": float(np.mean(latencies)),
                    "median_ms": float(np.median(latencies)),
                    "p95_ms": float(np.percentile(latencies, 95)),
                    "max_ms": float(np.max(latencies)),
                }

        if self._query_count > 0:
            stats["overall"] = {
                "total_queries": self._query_count,
                "mean_latency_ms": self._total_latency_ms / self._query_count,
                "query_rate_per_minute": len(self._query_rate_window),
            }

        return stats

    def get_hot_queries(self, k: int = 5) -> list[tuple[str, int]]:
        """Get most frequent query patterns.

        Returns
        -------
        list[tuple[str, int]]
            (pattern_hash, frequency) sorted descending.
        """
        patterns = sorted(self._pattern_histogram.items(), key=lambda x: x[1], reverse=True)
        return patterns[:k]

    # ── stats ─────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "table_id": self.table.table_id if hasattr(self.table, "table_id") else "unknown",
            "total_queries": self._query_count,
            "unique_patterns": len(self._pattern_histogram),
            "scene_count": len(self._scenes) + (1 if self._current_scene else 0),
            "cached_entries": len(self._cached_entries),
            "mean_latency_ms": self._total_latency_ms / self._query_count if self._query_count > 0 else 0.0,
            "query_rate_per_minute": len(self._query_rate_window),
            "latency_stats": self.get_latency_stats(),
            "hot_queries": self.get_hot_queries(5),
            "scenes": [s.to_dict() for s in self.detect_scenes(5)],
        }

    # ── internal ────────────────────────────────────────────

    def _update_scene(self, pattern: QueryPattern) -> None:
        """Update the current scene with a new query pattern."""
        if self._current_scene is None:
            self._scene_counter += 1
            self._current_scene = Scene(
                scene_id=f"scene_{self._scene_counter:04d}",
                queries=[pattern],
                start_time=pattern.timestamp,
                end_time=pattern.timestamp,
                dominant_pattern=pattern.pattern_hash,
                frequency=1,
            )
        else:
            # Check if pattern fits current scene (same type within timeout)
            if pattern.timestamp - self._current_scene.end_time <= self.strategy.scene_timeout_seconds:
                self._current_scene.queries.append(pattern)
                self._current_scene.end_time = pattern.timestamp
                self._current_scene.frequency += 1
                # Update dominant pattern
                self._update_dominant_pattern(self._current_scene)
            else:
                # Close current scene, start new
                self._scenes.append(self._current_scene)
                self._scene_counter += 1
                self._current_scene = Scene(
                    scene_id=f"scene_{self._scene_counter:04d}",
                    queries=[pattern],
                    start_time=pattern.timestamp,
                    end_time=pattern.timestamp,
                    dominant_pattern=pattern.pattern_hash,
                    frequency=1,
                )

    def _update_dominant_pattern(self, scene: Scene) -> None:
        """Recompute the dominant pattern of a scene."""
        pattern_counts: dict[str, int] = defaultdict(int)
        for q in scene.queries:
            pattern_counts[q.pattern_hash] += 1

        if pattern_counts:
            dominant = max(pattern_counts.items(), key=lambda x: x[1])
            scene.dominant_pattern = dominant[0]
            scene.frequency = dominant[1]
