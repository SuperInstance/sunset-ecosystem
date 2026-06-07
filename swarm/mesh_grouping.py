"""MeshGrouping — centroid clustering and pattern discovery for MeshVectorTables.

Discovers emergent groups from vector populations using:
- **K-means clustering** for dense population partitioning
- **Centroid tracking** for stable group identities over time
- **Group quality metrics** (cohesion, separation, novelty)
- **Automatic group merging/splitting** based on drift thresholds
- **Pattern labels** for human-readable group descriptions

Use Cases
---------
- **Pattern Discovery**: "What kinds of agents are in the fleet?" → auto-discover clusters
- **Cohort Analysis**: Track agent behavior patterns over time
- **Anomaly Detection**: Identify agents that don't fit any group (outliers)
- **Diversity-aware Breeding**: Select parents from different groups for diversity
- **Fleet Health**: Detect group collapse (agents leaving a cluster)

Architecture
------------
Groups are represented as:
  Group {
    group_id: str
    centroid: np.ndarray
    members: set[agent_id]
    cohesion: float     # avg similarity to centroid
    separation: float   # distance to nearest other centroid
    birth_time: float
    last_update: float
    label: str          # auto-generated description
  }

Clustering algorithms:
- "kmeans": Scikit-learn KMeans (if available) or custom implementation
- "hierarchical": Agglomerative clustering for dendrogram analysis
- "dbscan": Density-based for irregular shapes
- "single_pass": Online incremental clustering (streaming)

Reference: docs/QUANTA_VDB_DEEP_DIVE.md — Remaining Gaps: Grouping
"""

from __future__ import annotations

__all__ = ["MeshGrouping", "GroupProfile", "ClusterConfig"]

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# ── sklearn availability ───────────────────────────────────
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.metrics import silhouette_score

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    KMeans = AgglomerativeClustering = DBSCAN = None  # type: ignore
    silhouette_score = None  # type: ignore


@dataclass
class ClusterConfig:
    """Configuration for clustering algorithms."""
    algorithm: str = "kmeans"  # "kmeans", "hierarchical", "dbscan", "single_pass"
    n_clusters: int = 5
    max_iterations: int = 100
    tolerance: float = 1e-4
    random_seed: int | None = None
    # Single-pass parameters
    similarity_threshold: float = 0.85  # cosine similarity for group membership
    max_group_size: int = 100
    # DBSCAN parameters
    eps: float = 0.5
    min_samples: int = 5


@dataclass
class GroupProfile:
    """A discovered group of agents."""
    group_id: str
    centroid: np.ndarray
    members: set[str] = field(default_factory=set)
    cohesion: float = 0.0
    separation: float = 0.0
    silhouette: float = 0.0
    birth_time: float = 0.0
    last_update: float = 0.0
    label: str = ""
    generation_count: int = 0  # number of times this group was updated

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "centroid": self.centroid.tolist(),
            "members": list(self.members),
            "cohesion": self.cohesion,
            "separation": self.separation,
            "silhouette": self.silhouette,
            "birth_time": self.birth_time,
            "last_update": self.last_update,
            "label": self.label,
            "generation_count": self.generation_count,
        }


class MeshGrouping:
    """Clustering and pattern discovery for MeshVectorTables.

    Parameters
    ----------
    table : MeshVectorTable
        The table to cluster.
    config : ClusterConfig
        Clustering algorithm configuration.
    """

    def __init__(
        self,
        table: Any,
        config: ClusterConfig | None = None,
    ) -> None:
        self.table = table
        self.config = config or ClusterConfig()
        self.groups: dict[str, GroupProfile] = {}
        self._group_counter = 0
        self._last_cluster_time: float = 0.0
        self._cluster_count = 0

    # ── clustering ────────────────────────────────────────────

    def cluster(self) -> list[GroupProfile]:
        """Run clustering on all entries in the table.

        Returns
        -------
        list[GroupProfile]
            Discovered groups.
        """
        entries = list(self.table.all_entries())
        if not entries:
            return []

        agent_ids = [e.agent_id for e in entries]
        vectors = np.array([e.vector for e in entries], dtype=np.float32)

        if len(vectors) < self.config.n_clusters:
            # Too few entries — create one group per entry
            return self._single_element_groups(entries)

        if self.config.algorithm == "kmeans":
            labels = self._kmeans_cluster(vectors)
        elif self.config.algorithm == "hierarchical":
            labels = self._hierarchical_cluster(vectors)
        elif self.config.algorithm == "dbscan":
            labels = self._dbscan_cluster(vectors)
        elif self.config.algorithm == "single_pass":
            labels = self._single_pass_cluster(vectors)
        else:
            labels = self._custom_kmeans(vectors)

        # Build groups from labels
        groups: dict[int, list[tuple[str, np.ndarray]]] = {}
        for i, label in enumerate(labels):
            if label < 0:  # noise in DBSCAN
                continue
            groups.setdefault(label, []).append((agent_ids[i], vectors[i]))

        # Create/update group profiles
        self.groups.clear()
        for label, members in groups.items():
            group_id = f"group_{label:03d}"
            self._create_group(group_id, members)

        self._compute_quality_metrics(vectors, labels)
        self._last_cluster_time = time.time()
        self._cluster_count += 1

        return list(self.groups.values())

    def incremental_update(self, new_entry: Any) -> GroupProfile | None:
        """Update groups incrementally when a new entry arrives.

        Uses single-pass clustering for online updates.

        Parameters
        ----------
        new_entry : VectorTableEntry
            The new entry to incorporate.

        Returns
        -------
        GroupProfile | None
            The group the entry was assigned to, or None if outlier.
        """
        if not self.groups:
            # No existing groups — create one
            group_id = f"group_{self._group_counter:03d}"
            self._group_counter += 1
            group = GroupProfile(
                group_id=group_id,
                centroid=new_entry.vector.copy(),
                members={new_entry.agent_id},
                birth_time=time.time(),
                last_update=time.time(),
                label=f"group_{group_id}",
            )
            self.groups[group_id] = group
            return group

        # Find nearest group
        best_group_id: str | None = None
        best_similarity = -1.0

        for group_id, group in self.groups.items():
            sim = self._cosine_similarity(new_entry.vector, group.centroid)
            if sim > best_similarity:
                best_similarity = sim
                best_group_id = group_id

        if best_similarity >= self.config.similarity_threshold and best_group_id:
            # Add to existing group
            group = self.groups[best_group_id]
            group.members.add(new_entry.agent_id)
            # Update centroid (incremental mean)
            n = len(group.members)
            group.centroid = group.centroid * ((n - 1) / n) + new_entry.vector / n
            group.last_update = time.time()
            group.generation_count += 1
            return group
        elif len(self.groups) < self.config.n_clusters:
            # Create new group
            group_id = f"group_{self._group_counter:03d}"
            self._group_counter += 1
            group = GroupProfile(
                group_id=group_id,
                centroid=new_entry.vector.copy(),
                members={new_entry.agent_id},
                birth_time=time.time(),
                last_update=time.time(),
                label=f"group_{group_id}",
            )
            self.groups[group_id] = group
            return group
        else:
            # Outlier — add to nearest group anyway
            if best_group_id:
                group = self.groups[best_group_id]
                group.members.add(new_entry.agent_id)
                group.last_update = time.time()
                return group
            return None

    # ── group queries ─────────────────────────────────────────

    def find_outliers(self) -> list[str]:
        """Find agents that don't fit well in any group (low silhouette)."""
        outliers = []
        for group in self.groups.values():
            if group.silhouette < 0.0:
                # All members are potential outliers in this group
                for agent_id in group.members:
                    entry = self.table.query(agent_id)
                    if entry is not None:
                        # Check distance to centroid
                        dist = np.linalg.norm(entry.vector - group.centroid)
                        if dist > np.mean([np.linalg.norm(e.vector - group.centroid) for e in self.table.all_entries() if e.agent_id in group.members]):
                            outliers.append(agent_id)
        return outliers

    def find_dense_regions(self, k: int = 3) -> list[tuple[str, float]]:
        """Find groups with highest cohesion (dense clusters).

        Returns
        -------
        list[tuple[str, float]]
            (group_id, cohesion) sorted descending.
        """
        scores = [(g.group_id, g.cohesion) for g in self.groups.values()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def find_sparse_regions(self, k: int = 3) -> list[tuple[str, float]]:
        """Find groups with lowest cohesion (sparse/diffuse clusters).

        Returns
        -------
        list[tuple[str, float]]
            (group_id, cohesion) sorted ascending.
        """
        scores = [(g.group_id, g.cohesion) for g in self.groups.values()]
        scores.sort(key=lambda x: x[1])
        return scores[:k]

    def get_group_members(self, group_id: str) -> list[Any]:
        """Get all entries in a group."""
        group = self.groups.get(group_id)
        if not group:
            return []
        return [self.table.query(aid) for aid in group.members if self.table.query(aid) is not None]

    def get_group_centroid(self, group_id: str) -> np.ndarray | None:
        """Get the centroid of a group."""
        group = self.groups.get(group_id)
        return group.centroid if group else None

    # ── metrics ───────────────────────────────────────────────

    def compute_diversity_index(self) -> float:
        """Compute a diversity score: number of groups × average separation.

        Higher = more diverse fleet.
        """
        if not self.groups:
            return 0.0
        n = len(self.groups)
        avg_separation = np.mean([g.separation for g in self.groups.values()])
        return float(n * avg_separation)

    def compute_cohesion_map(self) -> dict[str, float]:
        """Map group_id -> cohesion score."""
        return {g.group_id: g.cohesion for g in self.groups.values()}

    # ── stats ─────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "table_id": self.table.table_id if hasattr(self.table, "table_id") else "unknown",
            "group_count": len(self.groups),
            "total_members": sum(len(g.members) for g in self.groups.values()),
            "algorithm": self.config.algorithm,
            "cluster_count": self._cluster_count,
            "last_cluster_time": self._last_cluster_time,
            "diversity_index": self.compute_diversity_index(),
            "groups": {gid: g.to_dict() for gid, g in self.groups.items()},
        }

    # ── internal ────────────────────────────────────────────

    def _kmeans_cluster(self, vectors: np.ndarray) -> np.ndarray:
        """K-means clustering using sklearn or custom fallback."""
        if _SKLEARN_AVAILABLE and KMeans is not None:
            kmeans = KMeans(
                n_clusters=min(self.config.n_clusters, len(vectors)),
                max_iter=self.config.max_iterations,
                tol=self.config.tolerance,
                random_state=self.config.random_seed,
                n_init=1,
            )
            return kmeans.fit_predict(vectors)
        return self._custom_kmeans(vectors)

    def _custom_kmeans(self, vectors: np.ndarray) -> np.ndarray:
        """Custom K-means implementation (no sklearn)."""
        n = len(vectors)
        k = min(self.config.n_clusters, n)
        rng = np.random.RandomState(self.config.random_seed)

        # Initialize centroids randomly
        indices = rng.choice(n, size=k, replace=False)
        centroids = vectors[indices].copy()

        labels = np.zeros(n, dtype=int)
        for _ in range(self.config.max_iterations):
            # Assign
            new_labels = np.array([
                int(np.argmin([np.linalg.norm(v - c) for c in centroids]))
                for v in vectors
            ])
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

            # Update centroids
            for j in range(k):
                members = vectors[labels == j]
                if len(members) > 0:
                    centroids[j] = members.mean(axis=0)

        return labels

    def _hierarchical_cluster(self, vectors: np.ndarray) -> np.ndarray:
        """Hierarchical clustering using sklearn or custom fallback."""
        if _SKLEARN_AVAILABLE and AgglomerativeClustering is not None:
            n = min(self.config.n_clusters, len(vectors))
            agg = AgglomerativeClustering(n_clusters=n)
            return agg.fit_predict(vectors)
        # Fallback: use kmeans
        return self._custom_kmeans(vectors)

    def _dbscan_cluster(self, vectors: np.ndarray) -> np.ndarray:
        """DBSCAN clustering using sklearn or custom fallback."""
        if _SKLEARN_AVAILABLE and DBSCAN is not None:
            dbscan = DBSCAN(eps=self.config.eps, min_samples=self.config.min_samples)
            return dbscan.fit_predict(vectors)
        # Fallback: single-pass with threshold
        return self._single_pass_cluster(vectors)

    def _single_pass_cluster(self, vectors: np.ndarray) -> np.ndarray:
        """Single-pass incremental clustering."""
        labels = np.full(len(vectors), -1, dtype=int)
        centroids: list[np.ndarray] = []

        for i, v in enumerate(vectors):
            best_idx = -1
            best_sim = -1.0
            for idx, c in enumerate(centroids):
                sim = self._cosine_similarity(v, c)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx

            if best_sim >= self.config.similarity_threshold and best_idx >= 0:
                labels[i] = best_idx
                # Update centroid
                n = np.sum(labels == best_idx)
                centroids[best_idx] = centroids[best_idx] * ((n - 1) / n) + v / n
            else:
                labels[i] = len(centroids)
                centroids.append(v.copy())

        return labels

    def _single_element_groups(self, entries: list[Any]) -> list[GroupProfile]:
        """Create one group per entry when too few for clustering."""
        self.groups.clear()
        for i, entry in enumerate(entries):
            group_id = f"group_{i:03d}"
            group = GroupProfile(
                group_id=group_id,
                centroid=entry.vector.copy(),
                members={entry.agent_id},
                cohesion=1.0,
                separation=0.0,
                birth_time=time.time(),
                last_update=time.time(),
                label=f"singleton_{entry.agent_id}",
            )
            self.groups[group_id] = group
        return list(self.groups.values())

    def _create_group(self, group_id: str, members: list[tuple[str, np.ndarray]]) -> None:
        """Create a GroupProfile from members."""
        agent_ids = [m[0] for m in members]
        vectors = np.array([m[1] for m in members], dtype=np.float32)
        centroid = vectors.mean(axis=0)

        group = GroupProfile(
            group_id=group_id,
            centroid=centroid,
            members=set(agent_ids),
            birth_time=time.time(),
            last_update=time.time(),
            label=f"group_{group_id}",
        )
        self.groups[group_id] = group

    def _compute_quality_metrics(self, vectors: np.ndarray, labels: np.ndarray) -> None:
        """Compute cohesion, separation, and silhouette for groups."""
        if len(vectors) < 2 or len(self.groups) < 2:
            return

        # Cohesion: average distance to centroid within group
        for group in self.groups.values():
            member_vectors = np.array([
                vectors[i] for i, l in enumerate(labels) if l >= 0
                and self._get_group_label(l) == group.group_id
            ], dtype=np.float32)
            if len(member_vectors) > 0:
                distances = np.linalg.norm(member_vectors - group.centroid, axis=1)
                group.cohesion = float(1.0 / (1.0 + np.mean(distances)))

        # Separation: distance to nearest other centroid
        centroids = [g.centroid for g in self.groups.values()]
        for i, group in enumerate(self.groups.values()):
            others = [c for j, c in enumerate(centroids) if j != i]
            if others:
                group.separation = float(np.min([np.linalg.norm(group.centroid - c) for c in others]))

        # Silhouette (sklearn if available)
        if _SKLEARN_AVAILABLE and silhouette_score is not None and len(set(labels)) > 1:
            try:
                score = silhouette_score(vectors, labels)
                for group in self.groups.values():
                    group.silhouette = float(score)
            except Exception:
                pass

    def _get_group_label(self, label_idx: int) -> str:
        """Map label index to group_id."""
        return f"group_{label_idx:03d}"

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
