"""SpectralMeshRouting — Optimize mesh gossip topology via spectral graph theory.

Uses the graph Laplacian eigenvalues and eigenvector centrality to:
1. Compute optimal gossip paths (minimize convergence time)
2. Detect network bottlenecks (algebraic connectivity λ₂)
3. Dynamically rewire mesh edges for faster anti-entropy

Mathematical foundation:
- Graph Laplacian L = D - A (degree matrix - adjacency)
- Fiedler value λ₂ = algebraic connectivity
- Effective resistance for pairwise gossip efficiency
- Spectral clustering for community detection

Reference: Spielman (2007) "Spectral Graph Theory and its Applications"
"""

from __future__ import annotations

__all__ = [
    "SpectralMeshRouter",
    "GraphLaplacian",
    "effective_resistance",
    "fiedler_vector",
]

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Graph Laplacian ───────────────────────────────────────────

@dataclass(frozen=True)
class GraphLaplacian:
    """Immutable spectral representation of a mesh topology."""

    adjacency: np.ndarray  # (n, n) binary adjacency matrix
    laplacian: np.ndarray  # (n, n) combinatorial Laplacian L = D - A
    eigenvalues: np.ndarray  # sorted ascending: λ₁=0 ≤ λ₂ ≤ ... ≤ λₙ
    eigenvectors: np.ndarray  # (n, n) orthonormal eigenvectors
    fiedler_value: float  # λ₂ — algebraic connectivity
    fiedler_vector: np.ndarray  # eigenvector for λ₂

    @classmethod
    def from_adjacency(cls, adj: np.ndarray) -> "GraphLaplacian":
        """Build from adjacency matrix (symmetric, zero diagonal)."""
        adj = np.asarray(adj, dtype=np.float64)
        n = adj.shape[0]
        assert adj.shape == (n, n), "Adjacency must be square"
        # Symmetrize
        adj = (adj + adj.T) / 2.0
        np.fill_diagonal(adj, 0.0)

        degree = np.diag(adj.sum(axis=1))
        lap = degree - adj

        # Eigendecomposition (Hermitian, real eigenvalues)
        w, v = np.linalg.eigh(lap)
        # Sort ascending
        idx = np.argsort(w)
        w = w[idx]
        v = v[:, idx]

        fiedler_val = float(w[1]) if n > 1 else 0.0
        fiedler_vec = v[:, 1].copy() if n > 1 else np.zeros(n)

        return cls(
            adjacency=adj,
            laplacian=lap,
            eigenvalues=w,
            eigenvectors=v,
            fiedler_value=fiedler_val,
            fiedler_vector=fiedler_vec,
        )

    def spectral_gap(self) -> float:
        """λ₂ / λ_max — normalized connectivity (0 = disconnected, 1 = complete)."""
        lmax = self.eigenvalues[-1]
        return self.fiedler_value / lmax if lmax > 0 else 0.0

    def effective_resistance(self, i: int, j: int) -> float:
        """Effective resistance between nodes i and j via Moore-Penrose pseudoinverse."""
        n = self.laplacian.shape[0]
        # L⁺ = pseudoinverse
        pinv = np.linalg.pinv(self.laplacian)
        return float(pinv[i, i] + pinv[j, j] - 2 * pinv[i, j])

    def cheeger_bound(self) -> float:
        """Cheeger inequality lower bound: λ₂/2 ≤ h(G) ≤ √(2λ₂)."""
        return self.fiedler_value / 2.0

    def spectral_clustering(self, k: int = 2) -> list[list[int]]:
        """k-way spectral clustering using first k eigenvectors."""
        n = self.adjacency.shape[0]
        if k < 2 or k > n:
            raise ValueError(f"k must be in [2, {n}]")
        # Use first k eigenvectors (skip λ₁=0)
        X = self.eigenvectors[:, 1:k+1]  # (n, k)
        # Simple k-means on rows
        centroids = X[:k, :].copy()
        labels = np.argmin(
            np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2),
            axis=1,
        )
        clusters: list[list[int]] = [[] for _ in range(k)]
        for idx, lab in enumerate(labels):
            clusters[int(lab)].append(idx)
        return clusters


# ── Spectral Mesh Router ──────────────────────────────────────

class SpectralMeshRouter:
    """Dynamic mesh topology optimizer using spectral graph theory.

    Recomputes Laplacian periodically and suggests edge rewirings
    that maximize algebraic connectivity (λ₂).
    """

    def __init__(self, node_ids: list[str]) -> None:
        self.node_ids = list(node_ids)
        self.n = len(node_ids)
        self._adj = np.eye(self.n, dtype=np.float64)  # start with self-loops only
        self._lap: GraphLaplacian | None = None
        self._history: list[tuple[float, float]] = []  # (time, λ₂)

    # ── topology management ───────────────────────────────

    def add_edge(self, i: int, j: int, weight: float = 1.0) -> None:
        """Add undirected edge (or update weight)."""
        self._adj[i, j] = max(self._adj[i, j], weight)
        self._adj[j, i] = self._adj[i, j]
        self._recompute()

    def remove_edge(self, i: int, j: int) -> None:
        """Remove edge."""
        self._adj[i, j] = 0.0
        self._adj[j, i] = 0.0
        self._recompute()

    def _recompute(self) -> None:
        """Rebuild spectral representation."""
        self._lap = GraphLaplacian.from_adjacency(self._adj)
        self._history.append(
            (__import__("time").time(), self._lap.fiedler_value)
        )

    # ── optimization suggestions ────────────────────────────

    def suggest_rewiring(self, num_edges: int = 1) -> list[tuple[int, int, float]]:
        """Suggest edges to add that maximally increase λ₂.

        Uses greedy spectral augmentation: for each candidate edge,
        compute new λ₂, pick the best.
        """
        if self._lap is None:
            self._recompute()
        assert self._lap is not None

        current_l2 = self._lap.fiedler_value
        suggestions: list[tuple[int, int, float]] = []
        adj_work = self._adj.copy()

        for _ in range(num_edges):
            best_gain = -1.0
            best_pair: tuple[int, int] | None = None

            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if adj_work[i, j] > 0:
                        continue  # already connected
                    # Try adding this edge
                    adj_work[i, j] = 1.0
                    adj_work[j, i] = 1.0
                    lap_tmp = GraphLaplacian.from_adjacency(adj_work)
                    gain = lap_tmp.fiedler_value - current_l2
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (i, j)
                    # Revert
                    adj_work[i, j] = 0.0
                    adj_work[j, i] = 0.0

            if best_pair is not None:
                i, j = best_pair
                suggestions.append((i, j, best_gain))
                adj_work[i, j] = 1.0
                adj_work[j, i] = 1.0
                current_l2 += best_gain

        return suggestions

    def optimal_gossip_order(self) -> list[str]:
        """Return node order for gossip: sort by Fiedler vector magnitude.

        Nodes with similar Fiedler values should gossip together
        (they're in the same spectral partition).
        """
        if self._lap is None:
            self._recompute()
        assert self._lap is not None

        # Sort by Fiedler vector value — this puts nodes in the
        # same community near each other
        order = np.argsort(self._lap.fiedler_vector)
        return [self.node_ids[int(i)] for i in order]

    # ── properties ──────────────────────────────────────────

    @property
    def algebraic_connectivity(self) -> float:
        if self._lap is None:
            self._recompute()
        assert self._lap is not None
        return self._lap.fiedler_value

    @property
    def spectral_gap_ratio(self) -> float:
        if self._lap is None:
            self._recompute()
        assert self._lap is not None
        return self._lap.spectral_gap()

    @property
    def is_connected(self) -> bool:
        return self.algebraic_connectivity > 1e-10

    # ── diagnostics ───────────────────────────────────────────

    def bottleneck_nodes(self, threshold_quantile: float = 0.9) -> list[str]:
        """Nodes with highest effective resistance to others — bottlenecks."""
        if self._lap is None:
            self._recompute()
        assert self._lap is not None

        resistances = []
        for i in range(self.n):
            total_r = sum(
                self._lap.effective_resistance(i, j)
                for j in range(self.n) if i != j
            )
            resistances.append(total_r)

        thresh = np.quantile(resistances, threshold_quantile)
        return [
            self.node_ids[i]
            for i, r in enumerate(resistances)
            if r >= thresh
        ]

    def cluster_report(self, k: int = 2) -> dict[str, Any]:
        """Spectral clustering report for mesh communities."""
        if self._lap is None:
            self._recompute()
        assert self._lap is not None

        clusters = self._lap.spectral_clustering(k)
        return {
            "k": k,
            "clusters": [
                {"id": ci, "nodes": [self.node_ids[i] for i in c]}
                for ci, c in enumerate(clusters)
            ],
            "fiedler_value": self._lap.fiedler_value,
            "spectral_gap": self._lap.spectral_gap(),
        }


# ── standalone helpers ─────────────────────────────────────────

def effective_resistance(adj: np.ndarray, i: int, j: int) -> float:
    """One-shot effective resistance between nodes i and j."""
    lap = GraphLaplacian.from_adjacency(adj)
    return lap.effective_resistance(i, j)


def fiedler_vector(adj: np.ndarray) -> np.ndarray:
    """One-shot Fiedler vector computation."""
    lap = GraphLaplacian.from_adjacency(adj)
    return lap.fiedler_vector
