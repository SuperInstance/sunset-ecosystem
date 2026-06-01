"""fleet/conservation_spectral_bridge.py — Conservation Spectral Framework integration.

Connects the SuperInstance Conservation Spectral Framework to the sunset-ecosystem
breeding loop. Agents get spectral fingerprints (Laplacian eigenvalue spectra).
Breeding diversity is computed via spectral alignment (cosine similarity of
eigenvalue spectra). Conservation ratios detect anomalies in trinity alignment.

The bridge works with or without the conservation-spectral-python package:
- If installed: uses the optimized Rust-backed engine
- If not installed: uses a pure-Python fallback with exact math

Usage:
    from fleet.conservation_spectral_bridge import SpectralFingerprint,
                                                    SpectralAlignmentScorer,
                                                    ConservationRatioMonitor

    # Fingerprint an agent from its capability graph
    fp = SpectralFingerprint.from_agent(agent_id="abc", capabilities=[...])

    # Score alignment between two agents (higher = more diverse)
    alignment = SpectralAlignmentScorer.score(fp_a, fp_b)

    # Monitor conservation ratio for trinity anomaly detection
    monitor = ConservationRatioMonitor()
    ratio = monitor.compute_ratio(agent_graph)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Try to import the real conservation-spectral package
try:
    from conservation_spectral import ConservationEngine
    _HAS_CONSERVATION_SPECTRAL = True
except ImportError:
    _HAS_CONSERVATION_SPECTRAL = False
    ConservationEngine = None

# ---------------------------------------------------------------------------
# Pure-Python fallback for the core spectral operations

def _laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Compute the combinatorial Laplacian L = D - A."""
    degrees = np.sum(adjacency, axis=1)
    D = np.diag(degrees)
    return D - adjacency

def _eigendecompose(laplacian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (eigenvalues, eigenvectors) sorted ascending."""
    vals, vecs = np.linalg.eigh(laplacian)
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]

def _conservation_ratio(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    adjacency: np.ndarray,
) -> float:
    """
    Conservation ratio: how much structural information survives
    projection from graph space to eigenvalue space.

    CR = trace(A @ V @ diag(λ) @ V^T) / trace(A @ A)
    where A is adjacency, V is eigenvector matrix, λ is eigenvalues.
    In practice: CR = sum(λ_i * ||v_i||²_A) / sum(A_ij²)
    """
    n = len(eigenvalues)
    # Reconstruct from spectral components
    spectral_reconstruct = np.zeros_like(adjacency, dtype=float)
    for i in range(n):
        v = eigenvectors[:, i]
        spectral_reconstruct += eigenvalues[i] * np.outer(v, v)

    # Frobenius norm of original adjacency
    norm_adj = np.linalg.norm(adjacency, "fro") ** 2
    if norm_adj == 0:
        return 0.0

    # Frobenius norm of spectral reconstruction
    norm_spectral = np.linalg.norm(spectral_reconstruct, "fro") ** 2

    return float(norm_spectral / norm_adj)

def _spectral_alignment(
    eigenvalues_a: np.ndarray,
    eigenvalues_b: np.ndarray,
) -> float:
    """
    Cosine similarity of two eigenvalue spectra.
    Higher = more aligned (less diverse = not good for breeding).
    """
    # Pad to same length
    max_len = max(len(eigenvalues_a), len(eigenvalues_b))
    a = np.zeros(max_len, dtype=float)
    b = np.zeros(max_len, dtype=float)
    a[:len(eigenvalues_a)] = eigenvalues_a
    b[:len(eigenvalues_b)] = eigenvalues_b

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))

def _fiedler_vector(eigenvectors: np.ndarray) -> np.ndarray:
    """Return the Fiedler vector (eigenvector of λ₂)."""
    # eigenvectors are sorted by eigenvalue ascending
    # λ₂ is the second-smallest eigenvalue (index 1)
    if eigenvectors.shape[1] < 2:
        return np.zeros(eigenvectors.shape[0])
    return eigenvectors[:, 1]

# ---------------------------------------------------------------------------
# Data classes

@dataclass
class SpectralFingerprint:
    """An agent's spectral identity — its Laplacian eigenvalue spectrum."""
    agent_id: str
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    fiedler_vector: np.ndarray
    conservation_ratio: float
    spectral_gap: float
    adjacency_matrix: np.ndarray = field(repr=False)
    metadata: dict = field(default_factory=dict)

    @property
    def alignment_coefficient(self) -> float:
        """α = λ₂ / CR(a) — the alignment coefficient."""
        if self.conservation_ratio == 0:
            return float("inf")
        return float(self.eigenvalues[1] / self.conservation_ratio)

    @classmethod
    def from_adjacency(
        cls,
        agent_id: str,
        adjacency: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> "SpectralFingerprint":
        """Build fingerprint from an adjacency matrix (capability graph)."""
        adjacency = np.asarray(adjacency, dtype=float)
        L = _laplacian(adjacency)
        vals, vecs = _eigendecompose(L)
        cr = _conservation_ratio(vals, vecs, adjacency)
        gap = float(vals[1] - vals[0]) if len(vals) > 1 else 0.0
        fiedler = _fiedler_vector(vecs)

        return cls(
            agent_id=agent_id,
            eigenvalues=vals,
            eigenvectors=vecs,
            fiedler_vector=fiedler,
            conservation_ratio=cr,
            spectral_gap=gap,
            adjacency_matrix=adjacency,
            metadata=metadata or {},
        )

    @classmethod
    def from_agent(
        cls,
        agent_id: str,
        capabilities: list[str],
        capability_links: Optional[list[tuple[str, str, float]]] = None,
        metadata: Optional[dict] = None,
    ) -> "SpectralFingerprint":
        """
        Build fingerprint from an agent's capability list and optional links.

        capabilities: list of capability names (nodes in graph)
        capability_links: (cap_a, cap_b, weight) edges — optional
        """
        n = len(capabilities)
        adj = np.zeros((n, n), dtype=float)

        # Build adjacency from links
        if capability_links:
            for a, b, w in capability_links:
                if a in capabilities and b in capabilities:
                    i = capabilities.index(a)
                    j = capabilities.index(b)
                    adj[i, j] = w
                    adj[j, i] = w  # undirected
        else:
            # Fully connected with equal weights if no links specified
            adj = np.ones((n, n), dtype=float) - np.eye(n)

        return cls.from_adjacency(
            agent_id=agent_id,
            adjacency=adj,
            metadata={
                "capabilities": capabilities,
                "capability_links": capability_links,
                **(metadata or {}),
            },
        )

    def to_dict(self) -> dict:
        """Serialize for JSON / WAL / git storage."""
        return {
            "agent_id": self.agent_id,
            "eigenvalues": self.eigenvalues.tolist(),
            "conservation_ratio": self.conservation_ratio,
            "spectral_gap": self.spectral_gap,
            "alignment_coefficient": self.alignment_coefficient,
            "fiedler_vector": self.fiedler_vector.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpectralFingerprint":
        """Deserialize from dict."""
        eigenvalues = np.array(d["eigenvalues"], dtype=float)
        n = len(eigenvalues)
        # Reconstruct minimal eigenvectors (not stored in dict to save space)
        vecs = np.eye(n)  # identity as placeholder — sufficient for alignment
        fiedler = vecs[:, 1] if n > 1 else np.zeros(n)
        return cls(
            agent_id=d["agent_id"],
            eigenvalues=eigenvalues,
            eigenvectors=vecs,
            fiedler_vector=fiedler,
            conservation_ratio=d["conservation_ratio"],
            spectral_gap=d["spectral_gap"],
            adjacency_matrix=np.zeros((n, n)),  # not stored
            metadata=d.get("metadata", {}),
        )


class SpectralAlignmentScorer:
    """
    Score breeding diversity via spectral alignment.

    High alignment = similar spectra = NOT diverse (bad for breeding).
    Low alignment = different spectra = diverse (good for breeding).

    The score is 1 - cosine_similarity, so higher = more diverse.
    """

    @staticmethod
    def score(a: SpectralFingerprint, b: SpectralFingerprint) -> float:
        """Diversity score: 0.0 (identical) to 1.0 (orthogonal)."""
        alignment = _spectral_alignment(a.eigenvalues, b.eigenvalues)
        return 1.0 - alignment

    @staticmethod
    def score_batch(
        fingerprints: list[SpectralFingerprint],
    ) -> np.ndarray:
        """Pairwise diversity matrix."""
        n = len(fingerprints)
        mat = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                s = SpectralAlignmentScorer.score(fingerprints[i], fingerprints[j])
                mat[i, j] = s
                mat[j, i] = s
        return mat

    @staticmethod
    def select_diverse_parents(
        fingerprints: list[SpectralFingerprint],
        n_parents: int = 2,
        min_diversity: float = 0.3,
    ) -> list[str]:
        """
        Select diverse parents for breeding.

        Greedy max-min selection: pick the pair with maximum
        minimum pairwise diversity.
        """
        if len(fingerprints) < n_parents:
            return [fp.agent_id for fp in fingerprints]

        mat = SpectralAlignmentScorer.score_batch(fingerprints)
        idx = list(range(len(fingerprints)))

        # Greedy: start with the pair that has maximum diversity
        best = None
        best_score = -1.0
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if mat[i, j] >= min_diversity and mat[i, j] > best_score:
                    best_score = mat[i, j]
                    best = [i, j]

        if best is None:
            # No pair meets threshold — just pick the most diverse pair
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    if mat[i, j] > best_score:
                        best_score = mat[i, j]
                        best = [i, j]

        selected = best[:n_parents] if len(best) >= n_parents else best
        return [fingerprints[i].agent_id for i in selected]


class ConservationRatioMonitor:
    """
    Monitor conservation ratios for trinity anomaly detection.

    When an agent's conservation ratio drops suddenly, it means the
    agent's structural coherence is breaking — a trinity violation
    is likely (ethos, pathos, or logos out of alignment).
    """

    def __init__(self, window_size: int = 10, threshold: float = 0.5):
        self.window_size = window_size
        self.threshold = threshold
        self._history: dict[str, list[float]] = {}

    def record(self, agent_id: str, ratio: float) -> None:
        """Record a conservation ratio for an agent."""
        if agent_id not in self._history:
            self._history[agent_id] = []
        self._history[agent_id].append(ratio)
        if len(self._history[agent_id]) > self.window_size:
            self._history[agent_id].pop(0)

    def is_anomaly(self, agent_id: str) -> tuple[bool, float]:
        """Return (is_anomaly, severity) for an agent."""
        history = self._history.get(agent_id, [])
        if len(history) < 2:
            return False, 0.0

        # Sudden drop in conservation ratio = anomaly
        recent = history[-1]
        baseline = np.mean(history[:-1])
        if baseline == 0:
            return False, 0.0

        drop = (baseline - recent) / baseline
        severity = max(0.0, drop)
        is_anom = severity > self.threshold
        return is_anom, severity

    def record_from_fingerprint(self, fp: SpectralFingerprint) -> None:
        """Record from a fingerprint."""
        self.record(fp.agent_id, fp.conservation_ratio)

    def get_history(self, agent_id: str) -> list[float]:
        return list(self._history.get(agent_id, []))


class SpectralBreederDiversity:
    """
    Integration layer: use spectral fingerprints as the diversity
    metric for the breeding loop.

    Replaces or augments HDC novelty / cosine similarity with
    spectral alignment scores.
    """

    def __init__(self):
        self.fingerprints: dict[str, SpectralFingerprint] = {}
        self.monitor = ConservationRatioMonitor()

    def register_agent(
        self,
        agent_id: str,
        capabilities: list[str],
        capability_links: Optional[list[tuple[str, str, float]]] = None,
        metadata: Optional[dict] = None,
    ) -> SpectralFingerprint:
        """Register an agent and compute its spectral fingerprint."""
        fp = SpectralFingerprint.from_agent(
            agent_id=agent_id,
            capabilities=capabilities,
            capability_links=capability_links,
            metadata=metadata,
        )
        self.fingerprints[agent_id] = fp
        self.monitor.record_from_fingerprint(fp)
        return fp

    def score_diversity(self, agent_a: str, agent_b: str) -> float:
        """Diversity score between two registered agents."""
        if agent_a not in self.fingerprints or agent_b not in self.fingerprints:
            return 0.0
        return SpectralAlignmentScorer.score(
            self.fingerprints[agent_a], self.fingerprints[agent_b]
        )

    def select_parents(self, n: int = 2, min_diversity: float = 0.3) -> list[str]:
        """Select diverse parents from all registered agents."""
        fps = list(self.fingerprints.values())
        return SpectralAlignmentScorer.select_diverse_parents(
            fps, n_parents=n, min_diversity=min_diversity
        )

    def detect_anomalies(self) -> list[tuple[str, float]]:
        """Return list of (agent_id, severity) for anomalous agents."""
        anomalies = []
        for agent_id in self.fingerprints:
            is_anom, severity = self.monitor.is_anomaly(agent_id)
            if is_anom:
                anomalies.append((agent_id, severity))
        return anomalies

    def to_archive(self) -> list[dict]:
        """Export all fingerprints for git storage / WAL."""
        return [fp.to_dict() for fp in self.fingerprints.values()]

    @classmethod
    def from_archive(cls, archive: list[dict]) -> "SpectralBreederDiversity":
        """Restore from archive."""
        sbd = cls()
        for d in archive:
            fp = SpectralFingerprint.from_dict(d)
            sbd.fingerprints[fp.agent_id] = fp
            sbd.monitor.record_from_fingerprint(fp)
        return sbd

# ---------------------------------------------------------------------------
# Conservation Spectral Engine (optional, uses real package if available)

class ConservationSpectralEngine:
    """
    Wrapper around the conservation-spectral package (if installed).
    Falls back to pure-Python implementation.
    """

    def __init__(self):
        self._engine = None
        if _HAS_CONSERVATION_SPECTRAL and ConservationEngine:
            self._engine = ConservationEngine()

    def analyze(self, adjacency: np.ndarray) -> dict:
        """Analyze a graph and return spectral metrics."""
        adjacency = np.asarray(adjacency, dtype=float)

        if self._engine:
            try:
                result = self._engine.analyze(adjacency.tolist())
                return {
                    "conservation_ratio": result.conservation_ratio,
                    "spectral_gap": result.spectral_gap,
                    "fiedler_vector": np.array(result.fiedler_vector),
                }
            except Exception:
                pass  # fall through to pure Python

        # Pure-Python fallback
        L = _laplacian(adjacency)
        vals, vecs = _eigendecompose(L)
        cr = _conservation_ratio(vals, vecs, adjacency)
        gap = float(vals[1] - vals[0]) if len(vals) > 1 else 0.0
        fiedler = _fiedler_vector(vecs)

        return {
            "conservation_ratio": cr,
            "spectral_gap": gap,
            "fiedler_vector": fiedler,
        }

    def is_available(self) -> bool:
        return self._engine is not None
