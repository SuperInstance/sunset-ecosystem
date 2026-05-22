"""FluxVectorTable — compressed, hardware-accelerated agent memory.

Wraps turbovec (Google TurboQuant) to store and search agent vectors
with fleet-specific metadata: fitness, thermal budget, generation,
and capability masks.

Reference: docs/TURBOVEC-REFACTOR-ANALYSIS.md
"""

from __future__ import annotations

__all__ = ["FluxVectorTable", "AgentVector", "AgentMeta"]

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from turbovec import IdMapIndex
except ImportError as exc:
    raise ImportError(
        "turbovec not installed. Run: pip install turbovec\n"
        "Or install with: pip install -e '.[vecsearch]'"
    ) from exc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentVector:
    """A single agent's compressed latent vector + metadata.

    Attributes:
        agent_id: Unique agent identifier (uint64-compatible).
        vector: Flat float32 array of dimension ``dim``.
        fitness: Trinity product (ethos × pathos × logos) [0, 1].
        generation: Which breeding generation this agent belongs to.
        capability_mask: 16-bit R15 capability mask.
        thermal_pressure: Current thermal load on the agent [0, 1].
    """

    agent_id: int
    vector: list[float]
    fitness: float = 0.0
    generation: int = 0
    capability_mask: int = 0xFFFF
    thermal_pressure: float = 0.0

    @property
    def dim(self) -> int:
        return len(self.vector)


@dataclass
class AgentMeta:
    """Metadata stored alongside the vector index for fast lookups."""

    fitness: float = 0.0
    generation: int = 0
    capability_mask: int = 0xFFFF
    thermal_pressure: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class FluxVectorTable:
    """Compressed vector store for fleet agent DNA and state.

    Uses turbovec's ``IdMapIndex`` for stable uint64 IDs and O(1) deletes.
    All vectors are quantized to 2–4 bits per coordinate with zero training.

    Args:
        dim: Vector dimensionality. Must be a multiple of 8.
        bit_width: Quantization bits per coordinate (2, 3, or 4).
        capability_filter: If set, search only returns agents whose
            capability_mask intersects this mask.

    Example::

        table = FluxVectorTable(dim=256, bit_width=4)
        table.add(AgentVector(agent_id=42, vector=[0.1, -0.2, ...]))
        neighbours = table.search(query=[0.0, 0.1, ...], k=5)
    """

    def __init__(
        self,
        dim: int,
        bit_width: int = 4,
        capability_filter: int | None = None,
    ) -> None:
        if dim % 8 != 0:
            raise ValueError(f"dim must be multiple of 8, got {dim}")
        if bit_width not in (2, 3, 4):
            raise ValueError(f"bit_width must be 2, 3, or 4, got {bit_width}")

        self.dim = dim
        self.bit_width = bit_width
        self._capability_filter = capability_filter

        self._index = IdMapIndex(dim=dim, bit_width=bit_width)
        self._meta: dict[int, AgentMeta] = {}

    # ── public API ──────────────────────────────────────────

    def add(self, av: AgentVector) -> None:
        """Add or overwrite an agent vector."""
        if av.dim != self.dim:
            raise ValueError(
                f"AgentVector dim {av.dim} != table dim {self.dim}"
            )

        import numpy as np

        vec_arr = np.array(av.vector, dtype=np.float32).reshape(1, self.dim)
        id_arr = np.array([av.agent_id], dtype=np.uint64)

        if av.agent_id in self._meta:
            # turbovec IdMapIndex does not support in-place update;
            # remove then re-add.
            self._index.remove(av.agent_id)

        self._index.add_with_ids(vec_arr, id_arr)
        self._meta[av.agent_id] = AgentMeta(
            fitness=av.fitness,
            generation=av.generation,
            capability_mask=av.capability_mask,
            thermal_pressure=av.thermal_pressure,
        )
        logger.debug("Added agent %d to vector table", av.agent_id)

    def search(
        self,
        query: list[float],
        k: int = 10,
        min_fitness: float | None = None,
        max_thermal: float | None = None,
        allowlist: list[int] | None = None,
    ) -> list[tuple[int, float, AgentMeta]]:
        """Search for the k nearest agents to *query*.

        Args:
            query: Flat float32 query vector of length ``dim``.
            k: Number of results.
            min_fitness: Drop results below this fitness threshold.
            max_thermal: Drop results above this thermal pressure.
            allowlist: Restrict search to these agent IDs.

        Returns:
            List of (agent_id, score, metadata) sorted best-first.
        """
        import numpy as np

        q_arr = np.array(query, dtype=np.float32).reshape(1, self.dim)

        # Build capability-filtered allowlist if requested
        candidates = self._build_allowlist(
            explicit=allowlist,
            min_fitness=min_fitness,
            max_thermal=max_thermal,
        )

        if candidates is not None and len(candidates) == 0:
            return []

        if candidates is not None:
            id_arr = np.array(candidates, dtype=np.uint64)
            scores_arr, ids_arr = self._index.search(q_arr, k=k, allowlist=id_arr)
        else:
            scores_arr, ids_arr = self._index.search(q_arr, k=k)

        results: list[tuple[int, float, AgentMeta]] = []
        for score, aid in zip(scores_arr[0], ids_arr[0]):
            meta = self._meta.get(int(aid))
            if meta is None:
                continue
            results.append((int(aid), float(score), meta))

        return results

    def remove(self, agent_id: int) -> bool:
        """Remove an agent vector. Returns True if it existed."""
        if agent_id not in self._meta:
            return False
        ok = self._index.remove(agent_id)
        if ok:
            del self._meta[agent_id]
            logger.debug("Removed agent %d from vector table", agent_id)
        return ok

    def contains(self, agent_id: int) -> bool:
        """Check whether an agent is in the table."""
        return self._index.contains(agent_id)

    def write(self, path: str | Path) -> None:
        """Serialize index + metadata to disk."""
        path = Path(path)
        self._index.write(str(path.with_suffix(".tvim")))
        # Metadata is small — store as JSON sidecar
        import json

        meta_path = path.with_suffix(".meta.json")
        serialisable = {
            str(aid): {
                "fitness": m.fitness,
                "generation": m.generation,
                "capability_mask": m.capability_mask,
                "thermal_pressure": m.thermal_pressure,
                "extra": m.extra,
            }
            for aid, m in self._meta.items()
        }
        meta_path.write_text(json.dumps(serialisable, indent=2))
        logger.info("Wrote vector table to %s", path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        dim: int,
        bit_width: int = 4,
    ) -> "FluxVectorTable":
        """Load a previously saved vector table."""
        path = Path(path)
        import json

        instance = cls(dim=dim, bit_width=bit_width)
        instance._index = IdMapIndex.load(str(path.with_suffix(".tvim")))
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            raw = json.loads(meta_path.read_text())
            for aid_str, m in raw.items():
                instance._meta[int(aid_str)] = AgentMeta(
                    fitness=m["fitness"],
                    generation=m["generation"],
                    capability_mask=m["capability_mask"],
                    thermal_pressure=m["thermal_pressure"],
                    extra=m.get("extra", {}),
                )
        logger.info("Loaded vector table from %s (%d agents)", path, len(instance._meta))
        return instance

    def __len__(self) -> int:
        return len(self._meta)

    def compute_novelty(
        self,
        agent_id: int,
        vector: list[float],
        population_vectors: list[list[float]],
    ) -> float:
        """Compute novelty as cosine distance from population centroid.

        Args:
            agent_id: Agent ID (included for API symmetry; not used in calc).
            vector: The agent's vector.
            population_vectors: Vectors of all other agents in the population.

        Returns:
            Cosine distance in [0, 2]. 0 = identical to population,
            higher = more divergent.
        """
        import numpy as np

        if not population_vectors:
            return 0.0

        vec = np.array(vector, dtype=np.float32)
        pop = np.array(population_vectors, dtype=np.float32)
        centroid = np.mean(pop, axis=0)

        vn = np.linalg.norm(vec)
        cn = np.linalg.norm(centroid)
        if vn == 0 or cn == 0:
            return 1.0

        sim = float(np.dot(vec, centroid) / (vn * cn))
        # Clamp for numerical safety
        sim = max(-1.0, min(1.0, sim))
        return 1.0 - sim  # distance = 1 - similarity

    def _get_vector(self, agent_id: int) -> np.ndarray | None:
        """Retrieve the raw float32 vector for an agent."""
        if agent_id not in self._meta:
            return None
        if hasattr(self._index, "_vectors"):
            vec = self._index._vectors.get(agent_id)
            if vec is not None:
                return np.array(vec, dtype=np.float32)
        return None

    def compute_diversity_matrix(self) -> tuple[np.ndarray, list[int]]:
        """Compute pairwise diversity (cosine distance) for all agents.

        Returns:
            (diversity_matrix, agent_ids) where diversity_matrix[i, j]
            is the cosine distance between agent_ids[i] and agent_ids[j].
            Range [0, 2] — 0 = identical, 2 = opposite.
        """
        agent_ids = sorted(self._meta.keys())
        n = len(agent_ids)
        if n < 2:
            return np.zeros((n, n), dtype=np.float32), agent_ids

        vectors: list[np.ndarray] = []
        for aid in agent_ids:
            vec = self._get_vector(aid)
            if vec is not None:
                vectors.append(vec)
            else:
                vectors.append(np.zeros(self.dim, dtype=np.float32))

        vecs = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vecs / norms

        sims = normalized @ normalized.T
        sims = np.clip(sims, -1.0, 1.0)
        dists = 1.0 - sims
        # Zero out self-distances exactly
        np.fill_diagonal(dists, 0.0)
        return dists, agent_ids

    def find_niche_centroids(self, k: int = 3) -> tuple[np.ndarray, dict[int, int]]:
        """K-means on latent vectors to find population clusters (niches).

        Args:
            k: Number of clusters. If population < k, k is reduced.

        Returns:
            (centroids, assignments) where centroids has shape (k, dim)
            and assignments maps agent_id -> cluster_index.
        """
        agent_ids = sorted(self._meta.keys())
        n = len(agent_ids)
        if n == 0:
            return np.zeros((k, self.dim), dtype=np.float32), {}

        k = min(k, n)

        vectors: list[np.ndarray] = []
        valid_ids: list[int] = []
        for aid in agent_ids:
            vec = self._get_vector(aid)
            if vec is not None:
                vectors.append(vec)
                valid_ids.append(aid)

        if len(valid_ids) < k:
            centroids = np.zeros((k, self.dim), dtype=np.float32)
            assignments = {aid: 0 for aid in agent_ids}
            return centroids, assignments

        X = np.array(vectors, dtype=np.float32)
        rng = np.random.RandomState(42)
        indices = rng.choice(len(X), size=k, replace=False)
        centroids = X[indices].copy()

        max_iter = 100
        labels = np.zeros(len(X), dtype=int)
        for _ in range(max_iter):
            distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new_centroids[j] = X[mask].mean(axis=0)
                else:
                    new_centroids[j] = X[rng.randint(len(X))]

            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids

        assignments = {aid: int(labels[i]) for i, aid in enumerate(valid_ids)}
        # Agents without vectors get assigned to cluster 0
        for aid in agent_ids:
            if aid not in assignments:
                assignments[aid] = 0
        return centroids, assignments

    def search_diverse_parents(self, n_results: int = 2) -> list[tuple[int, int]]:
        """Find maximally diverse parent pairs using novelty scoring + tournament results.

        Uses the diversity matrix to select pairs that are far apart
        while both having high fitness + novelty scores.

        Returns:
            List of (parent_a, parent_b) tuples, sorted by combined score.
        """
        agent_ids = sorted(self._meta.keys())
        if len(agent_ids) < 2:
            return []

        dists, ids = self.compute_diversity_matrix()
        id_to_idx = {aid: i for i, aid in enumerate(ids)}

        # Build population centroid for novelty
        vectors: list[np.ndarray] = []
        for aid in agent_ids:
            vec = self._get_vector(aid)
            if vec is not None:
                vectors.append(vec)
        centroid = np.mean(np.array(vectors), axis=0) if vectors else np.zeros(self.dim)

        # Score each agent: fitness + novelty (distance from centroid)
        scored: list[tuple[int, float, float, float]] = []
        for aid in agent_ids:
            meta = self._meta.get(aid)
            fitness = meta.fitness if meta else 0.0
            vec = self._get_vector(aid)
            novelty = 0.0
            if vec is not None and np.linalg.norm(centroid) > 0:
                vn = np.linalg.norm(vec)
                cn = np.linalg.norm(centroid)
                if vn > 0:
                    sim = float(np.dot(vec, centroid) / (vn * cn))
                    sim = max(-1.0, min(1.0, sim))
                    novelty = 1.0 - sim
            score = fitness + novelty
            scored.append((aid, score, fitness, novelty))

        scored.sort(key=lambda x: x[1], reverse=True)

        pairs: list[tuple[int, int]] = []
        used: set[int] = set()

        for _ in range(n_results):
            if len(scored) < 2:
                break

            parent_a = None
            for aid, _, _, _ in scored:
                if aid not in used:
                    parent_a = aid
                    break
            if parent_a is None:
                break

            # Find most diverse compatible candidate
            best_b: int | None = None
            best_dist = -1.0
            idx_a = id_to_idx.get(parent_a)

            for aid, _, _, _ in scored:
                if aid == parent_a or aid in used:
                    continue
                idx_b = id_to_idx.get(aid)
                if idx_a is not None and idx_b is not None:
                    dist = dists[idx_a, idx_b]
                    if dist > best_dist:
                        best_dist = dist
                        best_b = aid

            if best_b is None:
                for aid, _, _, _ in scored:
                    if aid != parent_a and aid not in used:
                        best_b = aid
                        break

            if best_b is None:
                break

            pairs.append((parent_a, best_b))
            used.add(parent_a)
            used.add(best_b)

        return pairs

    def recommend_breed_pair(self) -> tuple[int, int] | None:
        """Pick parents from different niches with high fitness.

        Uses k-means clustering to identify niches, then selects the
        highest-fitness pair from two different niches.

        Returns:
            (parent_a, parent_b) or None if population < 2.
        """
        agent_ids = sorted(self._meta.keys())
        if len(agent_ids) < 2:
            return None

        k = min(3, len(agent_ids))
        centroids, assignments = self.find_niche_centroids(k=k)

        # Group agents by niche
        niche_agents: dict[int, list[int]] = {}
        for aid in agent_ids:
            niche = assignments.get(aid, 0)
            niche_agents.setdefault(niche, []).append(aid)

        # Sort within each niche by fitness descending
        for niche in niche_agents:
            niche_agents[niche].sort(
                key=lambda aid: self._meta.get(aid, AgentMeta()).fitness,
                reverse=True,
            )

        niche_ids = sorted(niche_agents.keys())
        if len(niche_ids) < 2:
            # All in one niche — fall back to diverse search
            pairs = self.search_diverse_parents(n_results=1)
            return pairs[0] if pairs else None

        # Pick top agents from two different niches
        best_pair: tuple[int, int] | None = None
        best_score = -1.0

        for i in range(len(niche_ids)):
            for j in range(i + 1, len(niche_ids)):
                for a in niche_agents[niche_ids[i]][:2]:
                    for b in niche_agents[niche_ids[j]][:2]:
                        meta_a = self._meta.get(a, AgentMeta())
                        meta_b = self._meta.get(b, AgentMeta())
                        # Score = sum of fitness + niche separation bonus
                        score = meta_a.fitness + meta_b.fitness + 0.5
                        if score > best_score:
                            best_score = score
                            best_pair = (a, b)

        return best_pair

    def _build_allowlist(
        self,
        explicit: list[int] | None,
        min_fitness: float | None,
        max_thermal: float | None,
    ) -> list[int] | None:
        """Build an agent-ID allowlist from filters.

        Returns ``None`` when no filtering is needed (search full index).
        """
        # Fast path: no filters at all
        if (
            explicit is None
            and min_fitness is None
            and max_thermal is None
            and self._capability_filter is None
        ):
            return None

        candidates: set[int] = set(explicit) if explicit else set(self._meta.keys())

        if self._capability_filter is not None:
            candidates = {
                aid
                for aid in candidates
                if (self._meta[aid].capability_mask & self._capability_filter)
                != 0
            }

        if min_fitness is not None:
            candidates = {
                aid for aid in candidates if self._meta[aid].fitness >= min_fitness
            }

        if max_thermal is not None:
            candidates = {
                aid
                for aid in candidates
                if self._meta[aid].thermal_pressure <= max_thermal
            }

        return list(candidates)
