"""fleet/conservation_spectral_bridge.py — Agent-Native Language via Laplacians.

Implements SuperInstance's Conservation Spectral framework as a Python bridge
for sunset-ecosystem fleet coordination. Agents communicate through spectral
properties of their state graphs rather than raw data.

Core concepts:
  Spectral fingerprint   — Laplacian of agent's interaction graph = identity
  Eigenvalue similarity   — cosine alignment between agent spectra
  Fiedler vector          — routing decisions based on graph connectivity
  Conservation ratio      — confidence metric from eigenvalue gaps
  FLUX(A, B)             — L_composed − L_A − L_B = collaborative intelligence

References
----------
- SuperInstance/SuperInstance README — Agent-Native Language section
- SuperInstance/conservation-spectral-* (20+ languages, 204+ tests)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Spectral primitives ─────────────────────────────────────────────────

@dataclass
class SpectralFingerprint:
    """Agent identity encoded as spectral properties of an interaction graph."""
    agent_name: str
    laplacian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    @property
    def dimension(self) -> int:
        return self.laplacian.shape[0]

    @property
    def fiedler_value(self) -> float:
        """Second-smallest eigenvalue — graph connectivity."""
        return float(self.eigenvalues[1]) if len(self.eigenvalues) > 1 else 0.0

    @property
    def fiedler_vector(self) -> np.ndarray:
        """Eigenvector corresponding to Fiedler value — used for routing."""
        return self.eigenvectors[:, 1] if self.eigenvectors.shape[1] > 1 else np.zeros(self.dimension)

    def __repr__(self) -> str:
        return f"SpectralFingerprint({self.agent_name}, dim={self.dimension}, fiedler={self.fiedler_value:.4f})"


def build_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Build graph Laplacian L = D − A from adjacency matrix."""
    degrees = np.sum(adjacency, axis=1)
    d = np.diag(degrees)
    return d - adjacency


def spectral_decompose(laplacian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted eigenvalues and eigenvectors of Laplacian."""
    vals, vecs = np.linalg.eigh(laplacian)
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]


def compute_fingerprint(agent_name: str, adjacency: np.ndarray) -> SpectralFingerprint:
    """Compute spectral fingerprint from an agent's interaction adjacency matrix."""
    l = build_laplacian(adjacency)
    vals, vecs = spectral_decompose(l)
    return SpectralFingerprint(agent_name=agent_name, laplacian=l, eigenvalues=vals, eigenvectors=vecs)


# ── Alignment & confidence ──────────────────────────────────────────────

def eigenvalue_similarity(a: SpectralFingerprint, b: SpectralFingerprint) -> float:
    """Cosine similarity between eigenvalue spectra."""
    min_dim = min(len(a.eigenvalues), len(b.eigenvalues))
    va = a.eigenvalues[:min_dim]
    vb = b.eigenvalues[:min_dim]
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(np.dot(va, vb) / norm)


def conservation_ratio(fp: SpectralFingerprint) -> float:
    """Confidence metric from eigenvalue gap: λ₂ / λ₃."""
    if len(fp.eigenvalues) < 3 or fp.eigenvalues[2] == 0:
        return 0.0
    return float(fp.eigenvalues[1] / fp.eigenvalues[2])


def fiedler_alignment(a: SpectralFingerprint, b: SpectralFingerprint) -> float:
    """Cosine similarity between Fiedler vectors — routing alignment."""
    fa = a.fiedler_vector
    fb = b.fiedler_vector
    min_dim = min(len(fa), len(fb))
    fa = fa[:min_dim]
    fb = fb[:min_dim]
    norm = np.linalg.norm(fa) * np.linalg.norm(fb)
    if norm == 0:
        return 0.0
    return float(np.dot(fa, fb) / norm)


# ── Collaborative intelligence ───────────────────────────────────────────

def flux_collaborative_intelligence(a: SpectralFingerprint, b: SpectralFingerprint) -> np.ndarray:
    """FLUX(A, B) = L_composed − L_A − L_B.

    The residual Laplacian represents the collaborative intelligence
    that exists only in the space between agents.
    """
    max_dim = max(a.dimension, b.dimension)
    la = _pad_matrix(a.laplacian, max_dim)
    lb = _pad_matrix(b.laplacian, max_dim)
    # Simple composition: union of edges (OR)
    adj_a = _laplacian_to_adjacency(la)
    adj_b = _laplacian_to_adjacency(lb)
    adj_composed = np.clip(adj_a + adj_b, 0, 1)
    l_composed = build_laplacian(adj_composed)
    return l_composed - la - lb


def collaborative_score(a: SpectralFingerprint, b: SpectralFingerprint) -> float:
    """Scalar score: Frobenius norm of FLUX(A, B) residual."""
    flux = flux_collaborative_intelligence(a, b)
    return float(np.linalg.norm(flux, "fro"))


# ── Helpers ─────────────────────────────────────────────────────────────

def _pad_matrix(m: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad a square matrix to target dimension with zeros."""
    if m.shape[0] >= target_dim:
        return m[:target_dim, :target_dim]
    padded = np.zeros((target_dim, target_dim), dtype=m.dtype)
    padded[: m.shape[0], : m.shape[1]] = m
    return padded


def _laplacian_to_adjacency(laplacian: np.ndarray) -> np.ndarray:
    """Recover adjacency from Laplacian (A = D − L)."""
    degrees = np.diag(laplacian)
    d = np.diag(degrees)
    adj = d - laplacian
    # Zero out diagonal (no self-loops)
    np.fill_diagonal(adj, 0)
    return adj


# ── Fleet registry ──────────────────────────────────────────────────────

class SpectralFleetRegistry:
    """Registry of spectral fingerprints for fleet-wide alignment queries."""

    def __init__(self):
        self._fingerprints: Dict[str, SpectralFingerprint] = {}

    def register(self, fp: SpectralFingerprint) -> None:
        self._fingerprints[fp.agent_name] = fp

    def get(self, agent_name: str) -> Optional[SpectralFingerprint]:
        return self._fingerprints.get(agent_name)

    def alignment_matrix(self) -> Dict[Tuple[str, str], float]:
        """Pairwise eigenvalue similarity for all registered agents."""
        agents = list(self._fingerprints.keys())
        matrix = {}
        for i, a_name in enumerate(agents):
            for b_name in agents[i:]:
                a = self._fingerprints[a_name]
                b = self._fingerprints[b_name]
                matrix[(a_name, b_name)] = eigenvalue_similarity(a, b)
        return matrix

    def best_collaborator(self, agent_name: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Find agents with highest collaborative score."""
        source = self._fingerprints.get(agent_name)
        if source is None:
            return []
        scores = []
        for name, fp in self._fingerprints.items():
            if name == agent_name:
                continue
            scores.append((name, collaborative_score(source, fp)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def fiedler_route(self, source: str, target: str) -> List[str]:
        """Find intermediate agents via Fiedler vector alignment."""
        # Naive: find agents whose Fiedler vector aligns with both source and target
        s = self._fingerprints.get(source)
        t = self._fingerprints.get(target)
        if s is None or t is None:
            return []
        candidates = []
        for name, fp in self._fingerprints.items():
            if name in (source, target):
                continue
            align_s = fiedler_alignment(s, fp)
            align_t = fiedler_alignment(fp, t)
            candidates.append((name, align_s * align_t))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in candidates[:3]]

    def __len__(self) -> int:
        return len(self._fingerprints)


__all__ = [
    "SpectralFingerprint",
    "build_laplacian",
    "spectral_decompose",
    "compute_fingerprint",
    "eigenvalue_similarity",
    "conservation_ratio",
    "fiedler_alignment",
    "flux_collaborative_intelligence",
    "collaborative_score",
    "SpectralFleetRegistry",
]
