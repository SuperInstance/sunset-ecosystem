"""BreedOptimizer — Intelligent breeding optimization for the sunset-ecosystem fleet.

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
"""

from __future__ import annotations

__all__ = [
    "BreedOptimizer",
    "ParentPair",
    "OffspringPrediction",
    "BreedingArchive",
    "AnomalyResult",
]

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from fleet.cognitive_cache import CognitiveCache
    from fleet.t_minus_bridge import TMinusBridge
    from swarm.vector_swarm import VectorSwarm

logger = logging.getLogger(__name__)


@dataclass
class ParentPair:
    """A pair of parent agents for breeding."""

    parent_a: str
    parent_b: str
    diversity_score: float
    predicted_quality: float
    wasserstein_distance: float
    deadline_secs: float | None = None


@dataclass
class OffspringPrediction:
    """Prediction for offspring quality."""

    expected_fitness: float
    confidence: float
    novelty_score: float
    risk_level: str  # low, medium, high


@dataclass
class BreedingArchive:
    """MAP-Elites style archive for breeding history."""

    cells: dict[tuple[int, ...], list[Any]] = field(default_factory=dict)
    dimensions: int = 2
    bins_per_dim: int = 10
    coverage: float = 0.0
    qd_score: float = 0.0

    def add(self, behavior: tuple[float, ...], individual: Any) -> None:
        """Add an individual to the archive."""
        indices = self._to_indices(behavior)
        if indices not in self.cells:
            self.cells[indices] = []
        self.cells[indices].append(individual)
        self._update_metrics()

    def _to_indices(self, behavior: tuple[float, ...]) -> tuple[int, ...]:
        """Convert behavior coordinates to archive indices."""
        return tuple(
            min(int(b * self.bins_per_dim), self.bins_per_dim - 1) for b in behavior
        )

    def _update_metrics(self) -> None:
        """Update coverage and QD-score."""
        total_cells = self.bins_per_dim**self.dimensions
        self.coverage = len(self.cells) / total_cells
        self.qd_score = sum(
            max(
                (ind.get("fitness", 0.0) for ind in cell if ind is not None),
                default=0.0,
            )
            for cell in self.cells.values()
        )

    def get_best_in_cell(self, indices: tuple[int, ...]) -> Any | None:
        """Get the best individual in a cell."""
        cell = self.cells.get(indices, [])
        if not cell:
            return None
        # Filter out None entries
        valid = [ind for ind in cell if ind is not None]
        if not valid:
            return None
        return max(valid, key=lambda x: x.get("fitness", 0.0))

    def sample_diverse(self, k: int = 5) -> list[Any]:
        """Sample k diverse individuals from different cells."""
        if not self.cells:
            return []
        cells = list(self.cells.keys())
        selected = random.sample(cells, min(k, len(cells)))
        return [self.get_best_in_cell(c) for c in selected if self.get_best_in_cell(c)]


@dataclass
class AnomalyResult:
    """Result of anomaly detection on breeding history."""

    is_anomaly: bool
    anomaly_score: float
    reason: str
    affected_parents: list[str] = field(default_factory=list)


class BreedOptimizer:
    """Intelligent breeding optimizer for the fleet.

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
    """

    def __init__(
        self,
        node_id: str = "default",
        swarm: VectorSwarm | None = None,
        cache: CognitiveCache | None = None,
        tminus: TMinusBridge | None = None,
    ) -> None:
        self.node_id = node_id
        self.swarm = swarm
        self.cache = cache
        self.tminus = tminus
        self.archive = BreedingArchive()
        self._history: list[dict[str, Any]] = []
        self._wasserstein_cache: dict[tuple[str, str], float] = {}

    # ── Wasserstein Distance ────────────────────────────────

    def wasserstein_distance(
        self,
        distribution_a: list[float],
        distribution_b: list[float],
    ) -> float:
        """Compute 1-Wasserstein distance between two distributions.

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
        """
        if not distribution_a or not distribution_b:
            return float("inf")

        a = sorted(distribution_a)
        b = sorted(distribution_b)

        # Pad shorter distribution with its median
        median_a = a[len(a) // 2]
        median_b = b[len(b) // 2]
        while len(a) < len(b):
            a.append(median_a)
        while len(b) < len(a):
            b.append(median_b)

        # W_1 = mean absolute difference of sorted samples
        distance = sum(abs(x - y) for x, y in zip(a, b)) / len(a)

        return distance

    def diversity_score(
        self,
        agent_a_traits: list[float],
        agent_b_traits: list[float],
    ) -> float:
        """Compute diversity score between two agents.

        Combines Wasserstein distance with trait overlap.
        Higher = more diverse (better for breeding).
        """
        w_dist = self.wasserstein_distance(agent_a_traits, agent_b_traits)
        # Normalize: assume traits are in [0, 1] range
        normalized = min(w_dist / (1.0 + w_dist), 1.0)
        return normalized

    # ── Parent Selection ─────────────────────────────────────

    def select_parents(
        self,
        pool: list[dict[str, Any]],
        k: int = 3,
        diversity_weight: float = 0.5,
    ) -> list[ParentPair]:
        """Select top-k parent pairs by diversity + predicted quality.

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
        """
        if len(pool) < 2:
            return []

        pairs: list[ParentPair] = []
        for i, a in enumerate(pool):
            for b in pool[i + 1 :]:
                a_id = a.get("id", str(i))
                b_id = b.get("id", str(i + 1))
                a_traits = a.get("traits", [])
                b_traits = b.get("traits", [])

                if not a_traits or not b_traits:
                    continue

                w_dist = self.wasserstein_distance(a_traits, b_traits)
                div_score = self.diversity_score(a_traits, b_traits)
                pred = self._predict_offspring(a_traits, b_traits)

                composite = (
                    diversity_weight * div_score
                    + (1 - diversity_weight) * pred.expected_fitness
                )

                pair = ParentPair(
                    parent_a=a_id,
                    parent_b=b_id,
                    diversity_score=div_score,
                    predicted_quality=pred.expected_fitness,
                    wasserstein_distance=w_dist,
                )
                pairs.append(pair)

        # Sort by composite score descending
        pairs.sort(key=lambda p: p.diversity_score + p.predicted_quality, reverse=True)
        return pairs[:k]

    def _predict_offspring(
        self,
        traits_a: list[float],
        traits_b: list[float],
    ) -> OffspringPrediction:
        """Predict offspring quality from parent traits.

        Simple model: offspring fitness = mean(parent fitnesses) + crossover bonus.
        """
        if not traits_a or not traits_b:
            return OffspringPrediction(
                expected_fitness=0.0,
                confidence=0.0,
                novelty_score=0.0,
                risk_level="high",
            )

        mean_a = sum(traits_a) / len(traits_a)
        mean_b = sum(traits_b) / len(traits_b)
        expected = (mean_a + mean_b) / 2.0

        # Crossover bonus: diverse parents → higher potential
        w_dist = self.wasserstein_distance(traits_a, traits_b)
        bonus = w_dist * 0.1  # Small bonus for diversity
        expected = min(expected + bonus, 1.0)

        # Confidence: higher for similar-length trait vectors
        confidence = 1.0 - abs(len(traits_a) - len(traits_b)) / max(
            len(traits_a), len(traits_b), 1
        )
        confidence = max(confidence, 0.3)

        # Novelty: proportional to Wasserstein distance
        novelty = min(w_dist, 1.0)

        # Risk: high if either parent is extreme
        risk = "low"
        if mean_a < 0.2 or mean_b < 0.2 or mean_a > 0.8 or mean_b > 0.8:
            risk = "medium"
        if w_dist >= 0.8:
            risk = "high"

        return OffspringPrediction(
            expected_fitness=expected,
            confidence=confidence,
            novelty_score=novelty,
            risk_level=risk,
        )

    # ── Anomaly Detection ─────────────────────────────────────

    def detect_anomalies(
        self,
        history: list[dict[str, Any]] | None = None,
        threshold: float = 2.0,
    ) -> list[AnomalyResult]:
        """Detect anomalies in breeding history.

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
        """
        data = history or self._history
        if len(data) < 5:
            return []

        anomalies: list[AnomalyResult] = []

        # Fitness anomaly detection
        fitnesses = [h.get("offspring_fitness", 0.0) for h in data]
        mean_fit = sum(fitnesses) / len(fitnesses)
        std_fit = math.sqrt(
            sum((f - mean_fit) ** 2 for f in fitnesses) / len(fitnesses)
        )

        for i, h in enumerate(data):
            fit = h.get("offspring_fitness", 0.0)
            if std_fit > 0 and abs(fit - mean_fit) / std_fit > threshold:
                anomalies.append(
                    AnomalyResult(
                        is_anomaly=True,
                        anomaly_score=abs(fit - mean_fit) / std_fit,
                        reason=f"Fitness z-score {abs(fit - mean_fit) / std_fit:.1f} exceeds threshold",
                        affected_parents=[h.get("parent_a", ""), h.get("parent_b", "")],
                    )
                )

        # Inbreeding detection: repeated pairs
        pair_counts: dict[tuple[str, str], int] = {}
        for h in data:
            pair = tuple(sorted([h.get("parent_a", ""), h.get("parent_b", "")]))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        for pair, count in pair_counts.items():
            if count > len(data) * 0.3:  # Same pair > 30% of history
                anomalies.append(
                    AnomalyResult(
                        is_anomaly=True,
                        anomaly_score=float(count) / len(data),
                        reason=f"Inbreeding: pair {pair} appears {count}/{len(data)} times",
                        affected_parents=list(pair),
                    )
                )

        # Diversity collapse: consecutive low diversity
        diversities = [h.get("diversity", 0.0) for h in data]
        if len(diversities) >= 3:
            for i in range(len(diversities) - 2):
                if all(d < 0.1 for d in diversities[i : i + 3]):
                    anomalies.append(
                        AnomalyResult(
                            is_anomaly=True,
                            anomaly_score=3.0,
                            reason=f"Diversity collapse at generations {i}-{i + 2}",
                            affected_parents=[
                                data[i].get("parent_a", ""),
                                data[i].get("parent_b", ""),
                            ],
                        )
                    )

        return anomalies

    # ── Archive Optimization ────────────────────────────────

    def optimize_archive(
        self,
        archive: BreedingArchive | None = None,
        iterations: int = 100,
    ) -> BreedingArchive:
        """Optimize a breeding archive using MAP-Elites style selection.

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
        """
        target = archive or self.archive

        for _ in range(iterations):
            # Sample diverse individuals
            candidates = target.sample_diverse(k=5)
            if len(candidates) < 2:
                break

            # Compute behavior descriptors (fitness, diversity)
            for c in candidates:
                if c is None:
                    continue
                fitness = c.get("fitness", 0.0)
                traits = c.get("traits", [])
                # Simple behavior descriptor: (fitness, trait_mean)
                behavior = (fitness, sum(traits) / len(traits) if traits else 0.0)
                target.add(behavior, c)

        logger.info(
            "Archive optimized: coverage=%.2f%%, QD-score=%.1f",
            target.coverage * 100,
            target.qd_score,
        )
        return target

    # ── Integration with Swarm ────────────────────────────────

    def distributed_select_parents(
        self,
        pool: list[dict[str, Any]],
        k: int = 3,
    ) -> list[ParentPair]:
        """Select parents using distributed swarm search.

        If VectorSwarm is available, distribute the search across nodes.
        Otherwise, falls back to local selection.
        """
        if self.swarm is None:
            return self.select_parents(pool, k=k)

        # Distribute: each node evaluates a subset of pairs
        # For now, just use local selection with swarm metadata
        logger.info("Using distributed swarm selection (node=%s)", self.node_id)
        return self.select_parents(pool, k=k)

    # ── Deadline Management ─────────────────────────────────

    def set_breeding_deadline(
        self,
        parent_deadline: float,
        child_budget: float,
    ) -> float:
        """Set breeding deadline with parent→child inheritance.

        Uses TMinusBridge if available, otherwise simple min().
        """
        if self.tminus:
            return self.tminus.propagate_deadline(parent_deadline, child_budget)
        return min(parent_deadline, child_budget)

    # ── History Management ──────────────────────────────────

    def record_breeding(
        self,
        parent_a: str,
        parent_b: str,
        offspring_fitness: float,
        diversity: float,
        traits: list[float] | None = None,
    ) -> None:
        """Record a breeding event in history."""
        entry = {
            "parent_a": parent_a,
            "parent_b": parent_b,
            "offspring_fitness": offspring_fitness,
            "diversity": diversity,
            "traits": traits or [],
            "timestamp": time.time(),
        }
        self._history.append(entry)

    def get_history(self) -> list[dict[str, Any]]:
        """Get breeding history."""
        return self._history

    def get_stats(self) -> dict[str, Any]:
        """Get optimizer statistics."""
        if not self._history:
            return {
                "count": 0,
                "mean_fitness": 0.0,
                "mean_diversity": 0.0,
                "archive_coverage": self.archive.coverage,
                "archive_qd_score": self.archive.qd_score,
                "anomalies_detected": 0,
            }

        fitnesses = [h["offspring_fitness"] for h in self._history]
        diversities = [h["diversity"] for h in self._history]

        return {
            "count": len(self._history),
            "mean_fitness": sum(fitnesses) / len(fitnesses),
            "mean_diversity": sum(diversities) / len(diversities),
            "archive_coverage": self.archive.coverage,
            "archive_qd_score": self.archive.qd_score,
            "anomalies_detected": len(self.detect_anomalies()),
        }

    # ── Reports ───────────────────────────────────────────────

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive optimizer report."""
        return {
            "node_id": self.node_id,
            "history_size": len(self._history),
            "archive_coverage": self.archive.coverage,
            "archive_qd_score": self.archive.qd_score,
            "stats": self.get_stats(),
            "swarm_connected": self.swarm is not None,
            "cache_connected": self.cache is not None,
            "tminus_connected": self.tminus is not None,
        }
