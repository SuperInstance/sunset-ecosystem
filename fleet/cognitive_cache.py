"""CognitiveCache — Adaptive cache that learns from query patterns.

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
"""

from __future__ import annotations

__all__ = ["CognitiveCache", "CachePrediction", "PredictionEngine"]

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from swarm.mesh_vector_tables import MeshVectorTable, VectorTableEntry
from swarm.scene_tracker import SceneTracker, CacheStrategy
from swarm.tiered_mesh_storage import TieredMeshStorage, TierConfig

logger = logging.getLogger(__name__)


@dataclass
class CachePrediction:
    """A single prediction for cache preloading."""
    agent_id: str
    confidence: float          # 0.0-1.0 based on co-occurrence frequency
    reason: str                # "cooccurrence", "hot", "scene", "recent"
    predicted_queries: int = 1  # Expected number of queries


@dataclass
class PredictionEngine:
    """Generates predictions from SceneTracker patterns.

    Parameters
    ----------
    cooccurrence_threshold : float
        Minimum co-occurrence ratio to trigger preloading.
    hot_threshold : int
        Minimum access count to consider an agent "hot".
    recency_window_seconds : float
        Time window for recency-based predictions.
    """
    cooccurrence_threshold: float = 0.3
    hot_threshold: int = 3
    recency_window_seconds: float = 60.0

    def predict(self, tracker: SceneTracker, recent_queries: int = 10) -> list[CachePrediction]:
        """Generate predictions based on tracker patterns.

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
        """
        predictions: dict[str, CachePrediction] = {}

        # 1. Co-occurrence predictions
        cooccurrence = tracker._cooccurrence_map
        for agent_id, neighbors in cooccurrence.items():
            total_queries = tracker._access_counts.get(agent_id, 0)
            if total_queries == 0:
                continue

            for neighbor_id, count in neighbors.items():
                ratio = count / total_queries
                if ratio >= self.cooccurrence_threshold:
                    if neighbor_id not in predictions:
                        predictions[neighbor_id] = CachePrediction(
                            agent_id=neighbor_id,
                            confidence=ratio,
                            reason="cooccurrence",
                        )
                    else:
                        predictions[neighbor_id].confidence = max(
                            predictions[neighbor_id].confidence, ratio
                        )

        # 2. Hot query predictions — skip since get_hot_queries returns hashes
        # We already get co-occurrence predictions from step 1
        # Hot entries are handled by the storage tier itself

        # 3. Scene-based predictions
        current_scene = tracker.get_current_scene()
        if current_scene:
            scene_queries = len(current_scene.queries)
            for query in current_scene.queries:
                agent_id = query.query_params.get("agent_id")
                if agent_id and agent_id not in predictions:
                    predictions[agent_id] = CachePrediction(
                        agent_id=agent_id,
                        confidence=min(scene_queries / 5.0, 1.0),
                        reason="scene",
                    )

        return list(predictions.values())


class CognitiveCache:
    """Adaptive cache combining pattern detection and tiered storage.

    Parameters
    ----------
    storage : TieredMeshStorage
        Tiered storage backend.
    tracker : SceneTracker
        Scene tracker for pattern detection.
    engine : PredictionEngine
        Prediction engine for preloading.
    """

    def __init__(
        self,
        storage: TieredMeshStorage,
        tracker: SceneTracker,
        engine: PredictionEngine | None = None,
    ) -> None:
        self.storage = storage
        self.tracker = tracker
        self.engine = engine or PredictionEngine()
        self._prediction_hits = 0
        self._prediction_misses = 0
        self._preload_count = 0

    # ── query interface ─────────────────────────────────────────

    def query(self, agent_id: str) -> VectorTableEntry | None:
        """Query with automatic tracking and predictive preloading.

        Parameters
        ----------
        agent_id : str
            Agent ID to query.

        Returns
        -------
        VectorTableEntry | None
            Entry if found.
        """
        # Query storage
        entry = self.storage.query(agent_id)

        # Track query
        self.tracker.track_query(
            "by_id", "none",
            result_size=1 if entry else 0,
            latency_ms=0.0,
            query_params={"agent_id": agent_id},
        )

        # Update prediction accuracy
        if entry:
            self._prediction_hits += 1
        else:
            self._prediction_misses += 1

        # Maybe promote if found
        if entry:
            self.storage._maybe_promote(entry)

        # Run predictive preloading periodically
        if (self._prediction_hits + self._prediction_misses) % 5 == 0:
            self._preload_predictions()

        return entry

    def query_similar(self, vector: np.ndarray, k: int = 5) -> list[VectorTableEntry]:
        """Similarity query with tracking.

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
        """
        # Query all entries from base table, sort by distance
        all_entries = list(self.storage.base.all_entries())
        # Brute-force distance sort
        distances = [(float(np.linalg.norm(e.vector - vector)), e) for e in all_entries]
        distances.sort(key=lambda x: x[0])
        entries = [e for _, e in distances[:k]]

        self.tracker.track_query(
            "similarity", "fitness",
            result_size=len(entries),
            latency_ms=0.0,
            query_params={"k": k},
        )

        return entries

    # ── predictive preloading ─────────────────────────────────

    def _preload_predictions(self) -> None:
        """Preload predicted entries into hot tier."""
        predictions = self.engine.predict(self.tracker)

        for pred in predictions:
            # Check if already in hot table (base)
            if self.storage.base.query(pred.agent_id):
                continue

            # Try to find in warm tier and promote
            warm_entry = self.storage._warm_query(pred.agent_id)
            if warm_entry:
                self.storage._maybe_promote(warm_entry)
                self._preload_count += 1
                logger.debug(
                    "Preloaded %s (confidence=%.2f, reason=%s)",
                    pred.agent_id, pred.confidence, pred.reason,
                )

    # ── maintenance ─────────────────────────────────────────────

    def run_maintenance(self) -> None:
        """Run maintenance: demote cold entries, rebuild predictions."""
        self.storage._run_maintenance()
        self._preload_predictions()

    # ── stats ─────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics."""
        total = self._prediction_hits + self._prediction_misses
        return {
            "prediction_hits": self._prediction_hits,
            "prediction_misses": self._prediction_misses,
            "prediction_accuracy": (
                self._prediction_hits / total if total > 0 else 0.0
            ),
            "preload_count": self._preload_count,
            "tracker_stats": self.tracker.stats,
            "storage_stats": self.storage.get_tier_stats(),
        }
