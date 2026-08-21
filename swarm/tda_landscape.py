"""swarm/tda_landscape.py — Topological Data Analysis for fitness landscape guidance.

Uses persistent homology to characterize the topology of the fitness landscape:
- Detect holes (local optima traps, Betti-1)
- Detect loops (cyclic fitness patterns, Betti-1 in 2D)
- Detect voids (Betti-2)
- Guide breeding: avoid holes, exploit loops, navigate ridges

This is genuinely novel: no existing evolutionary algorithm uses persistent
homology to understand the topological structure of the fitness landscape.

Usage
-----
    from swarm.tda_landscape import TDALandscape, LandscapeGuide

    # Sample the landscape
    tda = TDALandscape(dimension=2, max_filtration=1.0)
    for genome in population:
        tda.add_sample(genome.to_vector(), fitness=genome.fitness)

    # Get topological features
    features = tda.compute_homology()
    print(f"Holes: {features['betti_1']}, Voids: {features['betti_2']}")

    # Guide breeding
    guide = LandscapeGuide(tda)
    recommendation = guide.recommend_direction(current_position)
    # recommendation: 'avoid', 'exploit', 'explore', 'ridge'
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform


try:
    from ripser import ripser

    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False


try:
    from sklearn.cluster import DBSCAN

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class PersistencePair:
    """A birth-death pair in a persistence diagram."""

    birth: float
    death: float
    dimension: int

    @property
    def persistence(self) -> float:
        """Persistence = death - birth. Longer = more significant feature."""
        return self.death - self.birth if self.death != float("inf") else 0.0

    def is_significant(self, threshold: float = 0.1) -> bool:
        """Check if feature is significant (persistent)."""
        return self.persistence > threshold


@dataclass
class TDALandscape:
    """Fitness landscape sampler with topological analysis."""

    dimension: int = 2
    max_filtration: float = 1.0
    persistence_threshold: float = 0.1

    _samples: List[np.ndarray] = field(default_factory=list, repr=False)
    _fitnesses: List[float] = field(default_factory=list, repr=False)

    def add_sample(self, position: np.ndarray, fitness: float) -> None:
        """Add a sample point to the landscape."""
        self._samples.append(np.array(position).flatten())
        self._fitnesses.append(fitness)

    def get_samples(self) -> np.ndarray:
        """Get all samples as an array."""
        if not self._samples:
            return np.array([])
        return np.array(self._samples)

    def get_fitnesses(self) -> np.ndarray:
        """Get all fitnesses as an array."""
        return np.array(self._fitnesses)

    def compute_homology(self) -> Dict:
        """Compute persistent homology of the fitness landscape.

        Returns Betti numbers, persistence diagrams, and significant features.
        """
        samples = self.get_samples()
        if len(samples) < 3:
            return {
                "betti_0": 0,
                "betti_1": 0,
                "betti_2": 0,
                "significant_holes": 0,
                "significant_loops": 0,
                "significant_voids": 0,
                "diagrams": {},
                "persistence_entropy": 0.0,
            }

        if RIPSER_AVAILABLE:
            return self._compute_ripser(samples)
        else:
            return self._compute_fallback(samples)

    def _compute_ripser(self, samples: np.ndarray) -> Dict:
        """Compute homology using Ripser."""
        # Weight distances by fitness (high fitness = closer together)
        distances = squareform(pdist(samples))

        # Normalize fitnesses to [0, 1] and use as weights
        fitnesses = self.get_fitnesses()
        if len(fitnesses) > 0:
            f_min, f_max = fitnesses.min(), fitnesses.max()
            if f_max > f_min:
                weights = (fitnesses - f_min) / (f_max - f_min)
                # High fitness = closer (smaller distance)
                for i in range(len(distances)):
                    for j in range(len(distances)):
                        if i != j:
                            distances[i, j] *= (2.0 - weights[i] - weights[j]) / 2.0

        # Compute persistent homology
        result = ripser(
            distances, maxdim=min(2, self.dimension - 1), distance_matrix=True
        )
        diagrams = result["dgms"]

        # Analyze diagrams
        significant_features = {0: 0, 1: 0, 2: 0}
        persistence_entropies = {0: 0.0, 1: 0.0, 2: 0.0}

        for dim, diagram in enumerate(diagrams):
            if dim > 2:
                continue
            pairs = []
            for birth, death in diagram:
                if death == float("inf"):
                    death = self.max_filtration
                pair = PersistencePair(
                    birth=float(birth), death=float(death), dimension=dim
                )
                pairs.append(pair)
                if pair.is_significant(self.persistence_threshold):
                    significant_features[dim] += 1

            # Persistence entropy
            persistences = [p.persistence for p in pairs if p.persistence > 0]
            if persistences:
                total = sum(persistences)
                probs = [p / total for p in persistences]
                entropy = -sum(p * math.log(p) for p in probs if p > 0)
                persistence_entropies[dim] = entropy

        return {
            "betti_0": len(diagrams[0]) if len(diagrams) > 0 else 0,
            "betti_1": len(diagrams[1]) if len(diagrams) > 1 else 0,
            "betti_2": len(diagrams[2]) if len(diagrams) > 2 else 0,
            "significant_components": significant_features[0],
            "significant_holes": significant_features[1],
            "significant_voids": significant_features[2],
            "diagrams": {
                dim: [
                    (p.birth, p.death, p.persistence)
                    for p in [
                        PersistencePair(
                            float(b),
                            float(d) if d != float("inf") else self.max_filtration,
                            dim,
                        )
                        for b, d in diagram
                    ]
                ]
                for dim, diagram in enumerate(diagrams)
            },
            "persistence_entropy": persistence_entropies,
        }

    def _compute_fallback(self, samples: np.ndarray) -> Dict:
        """Fallback topological analysis without Ripser."""
        # Use DBSCAN to find clusters (components)
        if SKLEARN_AVAILABLE and len(samples) >= 3:
            dbscan = DBSCAN(eps=self.max_filtration, min_samples=2)
            labels = dbscan.fit_predict(samples)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
        else:
            n_clusters = 1
            n_noise = 0

        # Estimate "holes" using local fitness curvature
        fitnesses = self.get_fitnesses()
        if len(fitnesses) >= 5:
            # Sort by fitness and look for saddle points
            sorted_indices = np.argsort(fitnesses)
            # A "hole" might be a region surrounded by high fitness
            # Simple heuristic: count of local maxima
            local_maxima = 0
            for i in range(1, len(fitnesses) - 1):
                if fitnesses[i] > fitnesses[i - 1] and fitnesses[i] > fitnesses[i + 1]:
                    local_maxima += 1
            estimated_holes = max(0, local_maxima - 1)
        else:
            estimated_holes = 0

        return {
            "betti_0": n_clusters + n_noise,
            "betti_1": estimated_holes,
            "betti_2": 0,
            "significant_components": n_clusters,
            "significant_holes": estimated_holes,
            "significant_voids": 0,
            "diagrams": {},
            "persistence_entropy": {"estimated": True},
        }

    def get_landscape_features(self) -> Dict:
        """Get features useful for breeding guidance."""
        homology = self.compute_homology()
        fitnesses = self.get_fitnesses()

        features = {
            "num_samples": len(self._samples),
            "holes": homology.get("significant_holes", 0),
            "loops": homology.get("significant_holes", 0),  # Betti-1 in 2D
            "voids": homology.get("significant_voids", 0),
            "components": homology.get("significant_components", 1),
            "fitness_range": float(fitnesses.max() - fitnesses.min())
            if len(fitnesses) > 0
            else 0.0,
            "fitness_std": float(fitnesses.std()) if len(fitnesses) > 0 else 0.0,
        }

        # Entropy measures landscape ruggedness
        if "persistence_entropy" in homology:
            pe = homology["persistence_entropy"]
            if isinstance(pe, dict):
                features["entropy_dim0"] = pe.get(0, 0.0)
                features["entropy_dim1"] = pe.get(1, 0.0)
                features["entropy_dim2"] = pe.get(2, 0.0)
            else:
                features["entropy_dim0"] = float(pe)

        return features


class LandscapeGuide:
    """Breeding guidance based on topological landscape analysis."""

    def __init__(self, tda: TDALandscape):
        self.tda = tda

    def recommend_direction(self, current_position: np.ndarray) -> Dict:
        """Recommend a breeding direction based on landscape topology.

        Returns a dict with:
        - strategy: 'avoid', 'exploit', 'explore', 'ridge', 'unknown'
        - confidence: float [0, 1]
        - rationale: str
        """
        features = self.tda.get_landscape_features()
        samples = self.tda.get_samples()
        fitnesses = self.tda.get_fitnesses()

        if len(samples) < 3:
            return {
                "strategy": "explore",
                "confidence": 1.0,
                "rationale": "Insufficient samples, explore widely",
            }

        # Find nearest samples
        distances = np.linalg.norm(samples - current_position, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_fitness = fitnesses[nearest_idx]

        # Check if we're near a hole (low fitness surrounded by high fitness)
        if features["holes"] > 0:
            # Look for local minimum
            local_window = 5
            start = max(0, nearest_idx - local_window)
            end = min(len(fitnesses), nearest_idx + local_window + 1)
            local_fitnesses = fitnesses[start:end]

            if nearest_fitness <= np.percentile(local_fitnesses, 25):
                return {
                    "strategy": "avoid",
                    "confidence": min(0.5 + features["holes"] * 0.1, 0.9),
                    "rationale": f"Near local minimum, {features['holes']} holes detected",
                }

        # Check if we're on a ridge (high fitness, surrounded by lower)
        if nearest_fitness >= np.percentile(fitnesses, 90):
            return {
                "strategy": "ridge",
                "confidence": 0.8,
                "rationale": "On high-fitness ridge, follow gradient",
            }

        # Check if we're in an exploitable region (high fitness, flat landscape)
        if features["fitness_std"] < 0.1 * features.get("fitness_range", 1.0):
            return {
                "strategy": "exploit",
                "confidence": 0.7,
                "rationale": "Flat landscape region, exploit locally",
            }

        # Default: explore
        return {
            "strategy": "explore",
            "confidence": 0.6,
            "rationale": "Rugged landscape, explore for better regions",
        }

    def get_avoidance_zones(self, threshold: float = 0.2) -> List[np.ndarray]:
        """Get regions to avoid (near holes/local minima)."""
        samples = self.tda.get_samples()
        fitnesses = self.tda.get_fitnesses()

        if len(samples) < 5:
            return []

        # Find local minima
        avoidance = []
        for i in range(1, len(fitnesses) - 1):
            if fitnesses[i] < fitnesses[i - 1] and fitnesses[i] < fitnesses[i + 1]:
                if fitnesses[i] < np.percentile(fitnesses, threshold * 100):
                    avoidance.append(samples[i])

        return avoidance

    def get_exploitation_zones(self, top_percentile: float = 90.0) -> List[np.ndarray]:
        """Get high-fitness regions to exploit."""
        samples = self.tda.get_samples()
        fitnesses = self.tda.get_fitnesses()

        if len(samples) == 0:
            return []

        threshold = np.percentile(fitnesses, top_percentile)
        return [samples[i] for i in range(len(fitnesses)) if fitnesses[i] >= threshold]
