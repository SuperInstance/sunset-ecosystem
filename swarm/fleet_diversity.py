"""
FleetDiversity — Pyversity-Powered Diversity Selection for Fleet Breeding

Wraps `Pringled/pyversity` (491⭐) to provide DPP/MMR/MSD/COVER/SSD
diversification strategies for the Cocapn Fleet breeding pipeline.

Research-backed selection:
- DPP (Determinantal Point Process): probabilistic repulsion — default
- MMR (Maximal Marginal Relevance): relevance-first with similarity penalty
- MSD (Max Sum of Distances): maximum variety, may sacrifice relevance
- COVER (Facility-Location): topic coverage, clustering scenarios
- SSD (Sliding Spectrum Decomposition): sequence-aware novelty for feeds

Integration Points
--------------------
- FleetBreederConsensus: diversity-aware parent selection before BFT proposal
- QDArchive: diversify elite archive for cross-node breeding pools
- FleetVectorIndex: diversity re-ranking for nearest-neighbor retrieval
- CMAESEmitter: diverse seed selection for emitter initialization

References
----------
- Pyversity: https://github.com/Pringled/pyversity (MIT)
- DPP: Kulesza & Taskar (2012), Chen et al (2018) — fast greedy MAP
- MMR: Carbonell & Goldstein (1998)
- MSD: Borodin, Lee & Ye (2012)
- COVER: Puthiya Parambath, Usunier & Grandvalet (2016)
- SSD: Huang, Wang, Zhang & Xu (2021)
"""

from __future__ import annotations

__all__ = [
    "FleetDiversitySelector",
    "DiversityStrategy",
    "PopulationItem",
    "DiversityStats",
]

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ── Re-export pyversity strategy names (fallback to string if not installed) ──


class DiversityStrategy(Enum):
    """Diversification strategies backed by pyversity research."""

    DPP = "dpp"  # Determinantal Point Process — probabilistic repulsion
    MMR = "mmr"  # Maximal Marginal Relevance
    MSD = "msd"  # Max Sum of Distances
    COVER = "cover"  # Facility-Location coverage
    SSD = "ssd"  # Sliding Spectrum Decomposition (sequence-aware)


@dataclass
class PopulationItem:
    """A single item in the breeding population."""

    id: str
    embedding: np.ndarray
    fitness: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiversityStats:
    """Diversity metrics for a population."""

    n_items: int
    mean_fitness: float
    mean_pairwise_distance: float
    min_pairwise_distance: float
    max_pairwise_distance: float
    ilad: float  # Intra-List Average Distance
    ilmd: float  # Intra-List Minimum Distance
    selected_indices: List[int]
    selected_fitness_mean: float
    selected_diversity_mean: float


# ── Core Adapter ────────────────────────────────────────────────────────────


class FleetDiversitySelector:
    """Fleet-grade diversity selection powered by pyversity.

    Provides research-backed diversification for:
    - Parent selection in breeding (DPP default)
    - Archive elite diversification (COVER for topic coverage)
    - Nearest-neighbor re-ranking (MMR for relevance+diversity balance)
    - Feed/sequence diversification (SSD for novelty over time)

    All strategies are pure NumPy, no heavy dependencies.
    """

    def __init__(
        self,
        strategy: DiversityStrategy = DiversityStrategy.DPP,
        diversity: float = 0.5,
        default_k: int = 10,
    ) -> None:
        self.strategy = strategy
        self.diversity = diversity
        self.default_k = default_k
        self._selection_history: List[Tuple[str, List[int]]] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def select_parents(
        self,
        population: List[PopulationItem],
        k: Optional[int] = None,
        strategy: Optional[DiversityStrategy] = None,
        diversity: Optional[float] = None,
    ) -> List[PopulationItem]:
        """Select ``k`` diverse parents from a population.

        Uses the configured strategy (DPP by default) to balance
        fitness (relevance) and embedding-space diversity.

        :param population: List of population items with embeddings.
        :param k: Number of parents to select (default: ``default_k``).
        :param strategy: Override diversification strategy.
        :param diversity: Override diversity parameter [0, 1].
        :return: Selected parents in diversity order.
        """
        if not population:
            return []

        k = k or self.default_k
        k = min(k, len(population))
        strat = strategy or self.strategy
        div = diversity if diversity is not None else self.diversity

        embeddings = np.stack([p.embedding for p in population])
        scores = np.array([p.fitness for p in population])

        indices = self._diversify(embeddings, scores, k, strat, div)

        self._selection_history.append((strat.value, indices.tolist()))
        return [population[i] for i in indices]

    def diversify_archive_elites(
        self,
        elites: List[Tuple[Any, float]],
        embeddings: np.ndarray,
        k: Optional[int] = None,
        strategy: Optional[DiversityStrategy] = None,
    ) -> List[Tuple[Any, float]]:
        """Select a diverse subset of archive elites.

        Ideal for cross-node breeding pool synchronization where
        you want maximum behavioral coverage, not just top fitness.

        :param elites: List of (individual, fitness) pairs from QDArchive.
        :param embeddings: Embeddings of elites (must match ``len(elites)``).
        :param k: Number to select.
        :param strategy: Override strategy (COVER recommended for coverage).
        :return: Diverse elite subset.
        """
        if not elites:
            return []

        k = k or self.default_k
        k = min(k, len(elites))
        strat = strategy or DiversityStrategy.COVER

        scores = np.array([fit for _, fit in elites])
        indices = self._diversify(embeddings, scores, k, strat, diversity=0.5)

        return [elites[i] for i in indices]

    def rerank_nearest_neighbors(
        self,
        candidates: List[PopulationItem],
        query_embedding: np.ndarray,
        k: Optional[int] = None,
        diversity: Optional[float] = None,
    ) -> List[PopulationItem]:
        """Re-rank nearest neighbors with diversity.

        First computes cosine similarity as relevance score, then
        applies DPP/MMR diversification to avoid redundant results.

        :param candidates: Candidate items to re-rank.
        :param query_embedding: Query vector for similarity scoring.
        :param k: Number to return.
        :param diversity: Override diversity parameter.
        :return: Diverse top-k candidates.
        """
        if not candidates:
            return []

        k = k or self.default_k
        k = min(k, len(candidates))
        div = diversity if diversity is not None else self.diversity

        embeddings = np.stack([c.embedding for c in candidates])
        # Cosine similarity as relevance
        norms = np.linalg.norm(embeddings, axis=1)
        q_norm = np.linalg.norm(query_embedding)
        scores = (
            embeddings @ query_embedding / (norms * q_norm + 1e-10)
            if q_norm > 0
            else np.zeros(len(candidates))
        )

        indices = self._diversify(embeddings, scores, k, DiversityStrategy.DPP, div)
        return [candidates[i] for i in indices]

    def compute_diversity_stats(
        self,
        population: List[PopulationItem],
        selected_indices: Optional[List[int]] = None,
    ) -> DiversityStats:
        """Compute diversity statistics for a population.

        :param population: Items to analyze.
        :param selected_indices: Indices of selected subset (for ILAD/ILMD).
        :return: Diversity metrics.
        """
        if not population:
            return DiversityStats(
                n_items=0,
                mean_fitness=0.0,
                mean_pairwise_distance=0.0,
                min_pairwise_distance=0.0,
                max_pairwise_distance=0.0,
                ilad=0.0,
                ilmd=0.0,
                selected_indices=[],
                selected_fitness_mean=0.0,
                selected_diversity_mean=0.0,
            )

        embeddings = np.stack([p.embedding for p in population])
        n = len(population)
        fitnesses = [p.fitness for p in population]

        # Pairwise cosine distances (1 - similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = embeddings @ embeddings.T / (norms @ norms.T + 1e-10)
        distances = 1.0 - similarities
        np.fill_diagonal(distances, 0.0)

        # Only upper triangle for unique pairs
        upper = distances[np.triu_indices(n, k=1)]
        mean_dist = float(upper.mean()) if len(upper) else 0.0
        min_dist = float(upper.min()) if len(upper) else 0.0
        max_dist = float(upper.max()) if len(upper) else 0.0

        # ILAD / ILMD on selected subset
        ilad = ilmd = selected_fitness_mean = selected_div_mean = 0.0
        if selected_indices:
            sel_emb = embeddings[selected_indices]
            sel_norms = np.linalg.norm(sel_emb, axis=1, keepdims=True)
            sel_sim = sel_emb @ sel_emb.T / (sel_norms @ sel_norms.T + 1e-10)
            sel_dist = 1.0 - sel_sim
            np.fill_diagonal(sel_dist, 0.0)
            m = len(selected_indices)
            if m > 1:
                sel_upper = sel_dist[np.triu_indices(m, k=1)]
                ilad = float(sel_upper.mean())
                ilmd = float(sel_upper.min())
            selected_fitness_mean = np.mean(
                [population[i].fitness for i in selected_indices]
            )
            selected_div_mean = ilad

        return DiversityStats(
            n_items=n,
            mean_fitness=float(np.mean(fitnesses)),
            mean_pairwise_distance=mean_dist,
            min_pairwise_distance=min_dist,
            max_pairwise_distance=max_dist,
            ilad=ilad,
            ilmd=ilmd,
            selected_indices=selected_indices or [],
            selected_fitness_mean=selected_fitness_mean,
            selected_diversity_mean=selected_div_mean,
        )

    def get_history(self) -> List[Tuple[str, List[int]]]:
        """Return selection history as (strategy_name, indices) pairs."""
        return self._selection_history.copy()

    def clear_history(self) -> None:
        """Clear selection history."""
        self._selection_history.clear()

    # ── Internal ───────────────────────────────────────────────────────────

    def _diversify(
        self,
        embeddings: np.ndarray,
        scores: np.ndarray,
        k: int,
        strategy: DiversityStrategy,
        diversity: float,
    ) -> np.ndarray:
        """Call pyversity.diversify with fallback pure-NumPy implementation."""
        try:
            from pyversity import diversify, Strategy

            strat_map = {
                DiversityStrategy.DPP: Strategy.DPP,
                DiversityStrategy.MMR: Strategy.MMR,
                DiversityStrategy.MSD: Strategy.MSD,
                DiversityStrategy.COVER: Strategy.COVER,
                DiversityStrategy.SSD: Strategy.SSD,
            }
            result = diversify(
                embeddings=embeddings,
                scores=scores,
                k=k,
                strategy=strat_map[strategy],
                diversity=diversity,
            )
            return result.indices
        except ImportError:
            log.warning("pyversity not installed; using fallback %s", strategy.value)
            return self._fallback_diversify(embeddings, scores, k, strategy, diversity)

    def _fallback_diversify(
        self,
        embeddings: np.ndarray,
        scores: np.ndarray,
        k: int,
        strategy: DiversityStrategy,
        diversity: float,
    ) -> np.ndarray:
        """Pure-NumPy fallback when pyversity is not available."""
        n, d = embeddings.shape
        if k >= n:
            return np.arange(n)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / (norms + 1e-10)

        if strategy == DiversityStrategy.MMR:
            return self._fallback_mmr(normed, scores, k, diversity)
        elif strategy == DiversityStrategy.MSD:
            return self._fallback_msd(normed, scores, k, diversity)
        elif strategy == DiversityStrategy.DPP:
            return self._fallback_dpp(normed, scores, k, diversity)
        elif strategy == DiversityStrategy.COVER:
            return self._fallback_cover(normed, scores, k, diversity)
        elif strategy == DiversityStrategy.SSD:
            return self._fallback_ssd(normed, scores, k, diversity)
        else:
            # Default to top-k by score
            return np.argsort(scores)[::-1][:k]

    def _fallback_mmr(
        self, normed: np.ndarray, scores: np.ndarray, k: int, diversity: float
    ) -> np.ndarray:
        """Maximal Marginal Relevance: relevance - λ * max_sim(selected)."""
        selected: List[int] = []
        remaining = set(range(len(scores)))
        for _ in range(k):
            if not remaining:
                break
            best_idx = -1
            best_score = -np.inf
            for idx in remaining:
                rel = scores[idx]
                if selected:
                    sims = normed[idx] @ normed[selected].T
                    max_sim = sims.max()
                else:
                    max_sim = 0.0
                mmr_score = (1 - diversity) * rel - diversity * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            selected.append(best_idx)
            remaining.discard(best_idx)
        return np.array(selected)

    def _fallback_msd(
        self, normed: np.ndarray, scores: np.ndarray, k: int, diversity: float
    ) -> np.ndarray:
        """Max Sum of Distances: relevance + λ * sum_dist(selected)."""
        selected: List[int] = []
        remaining = set(range(len(scores)))
        for _ in range(k):
            if not remaining:
                break
            best_idx = -1
            best_score = -np.inf
            for idx in remaining:
                rel = scores[idx]
                if selected:
                    dists = 1.0 - normed[idx] @ normed[selected].T
                    sum_dist = dists.sum()
                else:
                    sum_dist = 0.0
                msd_score = (1 - diversity) * rel + diversity * sum_dist / max(
                    len(selected), 1
                )
                if msd_score > best_score:
                    best_score = msd_score
                    best_idx = idx
            selected.append(best_idx)
            remaining.discard(best_idx)
        return np.array(selected)

    def _fallback_dpp(
        self, normed: np.ndarray, scores: np.ndarray, k: int, diversity: float
    ) -> np.ndarray:
        """Greedy DPP: argmax det(L_S) where L = diag(scores) * S * S^T."""
        # Similarity matrix
        S = normed @ normed.T
        # DPP kernel: L = diag(scores) * S * diag(scores)
        L = np.outer(scores, scores) * S
        selected: List[int] = []
        remaining = list(range(len(scores)))

        for _ in range(k):
            if not remaining:
                break
            best_idx = -1
            best_gain = -np.inf
            for idx in remaining:
                # Marginal gain: L[idx, idx] - L[idx, selected] @ L[selected, selected]^-1 @ L[selected, idx]
                if selected:
                    L_ss = L[np.ix_(selected, selected)]
                    try:
                        L_ss_inv = np.linalg.inv(L_ss + 1e-6 * np.eye(len(selected)))
                    except np.linalg.LinAlgError:
                        L_ss_inv = np.linalg.pinv(L_ss)
                    gain = L[idx, idx] - L[idx, selected] @ L_ss_inv @ L[selected, idx]
                else:
                    gain = L[idx, idx]
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
        return np.array(selected)

    def _fallback_cover(
        self, normed: np.ndarray, scores: np.ndarray, k: int, diversity: float
    ) -> np.ndarray:
        """Facility-Location: maximize coverage of the dataset."""
        # Greedy facility location: pick points that maximize minimum distance to selected
        selected: List[int] = []
        remaining = set(range(len(scores)))
        for _ in range(k):
            if not remaining:
                break
            best_idx = -1
            best_score = -np.inf
            for idx in remaining:
                rel = scores[idx]
                if selected:
                    dists = 1.0 - normed[idx] @ normed[selected].T
                    min_dist = dists.min()
                else:
                    min_dist = 1.0  # maximum possible
                # Balance relevance and coverage
                cover_score = (1 - diversity) * rel + diversity * min_dist
                if cover_score > best_score:
                    best_score = cover_score
                    best_idx = idx
            selected.append(best_idx)
            remaining.discard(best_idx)
        return np.array(selected)

    def _fallback_ssd(
        self, normed: np.ndarray, scores: np.ndarray, k: int, diversity: float
    ) -> np.ndarray:
        """SSD: sequence-aware novelty (simplified without recent_embeddings)."""
        # Fallback to MMR-like selection since we don't have sequence context
        return self._fallback_mmr(normed, scores, k, diversity)

    # ── Integration Helpers ─────────────────────────────────────────────────

    @classmethod
    def from_breeder_consensus(
        cls,
        consensus: Any,  # FleetBreederConsensus
        strategy: DiversityStrategy = DiversityStrategy.DPP,
        diversity: float = 0.5,
    ) -> "FleetDiversitySelector":
        """Create a selector wired to a FleetBreederConsensus instance."""
        return cls(strategy=strategy, diversity=diversity)

    def wire_to_qd_archive(
        self,
        archive: Any,  # QDArchive
        k: int = 10,
        strategy: Optional[DiversityStrategy] = None,
    ) -> List[Any]:
        """Select diverse elites from a QDArchive for breeding.

        :param archive: QDArchive instance.
        :param k: Number of diverse elites to select.
        :param strategy: Override strategy (COVER for topic coverage).
        :return: Selected elite individuals.
        """
        elites = archive.get_all_elites()
        if not elites:
            return []
        individuals = [ind for ind, _ in elites]
        fitnesses = [fit for _, fit in elites]
        # Use behavior grid indices as simple embeddings for diversification
        grid_indices = list(archive._grid.keys())
        embeddings = np.array(
            [
                np.array([idx / g for idx, g in zip(pos, archive.grid_shape)])
                for pos in grid_indices
            ]
        )
        if embeddings.shape[0] == 0:
            return individuals[:k]

        selected = self.diversify_archive_elites(
            list(zip(individuals, fitnesses)),
            embeddings,
            k,
            strategy,
        )
        return [ind for ind, _ in selected]

    def __repr__(self) -> str:
        return (
            f"FleetDiversitySelector("
            f"strategy={self.strategy.value}, "
            f"diversity={self.diversity}, "
            f"history={len(self._selection_history)} selections)"
        )
