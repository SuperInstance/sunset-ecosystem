"""HnswMeshTable — HNSW-powered approximate nearest neighbor for MeshVectorTables.

Adds O(log n) vector search to the O(n) brute-force MeshVectorTable.
Uses hnswlib (optional dependency) with graceful fallback.

Integration:
- Wraps MeshVectorTable as backing store
- HNSW index is a hot-cache overlay (rebuildable from backing store)
- CRDT sync still operates on the backing store
- ANN queries are served from HNSW with verification from backing store

Reference: docs/QUANTA_VDB_DEEP_DIVE.md — Phase 1: Hybrid Index
"""

from __future__ import annotations

__all__ = ["HnswMeshTable", "HnswIndexConfig"]

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional hnswlib — installed via: pip install hnswlib
_HNSWLIB_AVAILABLE = False
try:
    import hnswlib

    _HNSWLIB_AVAILABLE = True
except ImportError:
    logger.info("hnswlib not available; HNSW index will use brute-force fallback")


@dataclass(frozen=True)
class HnswIndexConfig:
    """Configuration for HNSW index."""

    dim: int = 64
    space: str = "l2"  # "l2" or "ip" (inner product = cosine similarity)
    max_elements: int = 10000
    ef_construction: int = 200
    ef_search: int = 50
    M: int = 16


class HnswMeshTable:
    """MeshVectorTable with HNSW approximate nearest neighbor overlay.

    Parameters
    ----------
    base_table : MeshVectorTable
        The underlying CRDT table (all writes go here).
    config : HnswIndexConfig
        HNSW index parameters.
    auto_rebuild_threshold : float
        Rebuild HNSW index when fraction of entries changed exceeds this.
    """

    def __init__(
        self,
        base_table: Any,  # MeshVectorTable
        config: HnswIndexConfig | None = None,
        auto_rebuild_threshold: float = 0.1,
    ) -> None:
        self.base = base_table
        self.config = config or HnswIndexConfig(dim=64)
        self.auto_rebuild_threshold = auto_rebuild_threshold

        self._lock = threading.RLock()
        self._hnsw_index: Any | None = None
        self._id_to_label: dict[str, int] = {}  # agent_id -> HNSW label
        self._label_to_id: dict[int, str] = {}  # HNSW label -> agent_id
        self._next_label: int = 0
        self._index_count: int = 0  # entries in HNSW
        self._insert_count_since_rebuild: int = 0
        self._rebuild_count: int = 0

        # Build initial index from existing entries
        self._rebuild_index()

    # ── public API ──────────────────────────────────────────

    def insert(self, entry: Any) -> bool:  # VectorTableEntry
        """Insert into base table + HNSW index."""
        ok = self.base.insert(entry)
        if ok:
            with self._lock:
                self._insert_count_since_rebuild += 1
                self._add_to_hnsw(entry)
                self._maybe_trigger_rebuild()
        return ok

    def knn_search(
        self,
        query_vector: np.ndarray | list[float],
        k: int = 10,
        filter_fn: Any | None = None,
    ) -> list[tuple[Any, float]]:  # (VectorTableEntry, distance)
        """Approximate k-nearest neighbor search.

        Parameters
        ----------
        query_vector : np.ndarray
            The query vector.
        k : int
            Number of nearest neighbors to return.
        filter_fn : callable | None
            Optional post-filter (entry -> bool).

        Returns
        -------
        list[tuple[VectorTableEntry, float]]
            Results sorted by distance ascending.
        """
        vec = np.array(query_vector, dtype=np.float32)
        if vec.shape[0] != self.config.dim:
            raise ValueError(f"Query dim {vec.shape[0]} != index dim {self.config.dim}")

        with self._lock:
            if self._hnsw_index is None or self._index_count == 0:
                # Fallback to brute-force from base table
                return self._brute_force_knn(vec, k, filter_fn)

            # HNSW search
            try:
                labels, distances = self._hnsw_index.knn_query(vec.reshape(1, -1), k=k)
                labels = labels[0]
                distances = distances[0]
            except Exception as exc:
                logger.warning("HNSW query failed: %s", exc)
                return self._brute_force_knn(vec, k, filter_fn)

        # Resolve labels to entries and verify
        results: list[tuple[Any, float]] = []
        for label, dist in zip(labels, distances):
            agent_id = self._label_to_id.get(int(label))
            if agent_id is None:
                continue
            entry = self.base.query(agent_id)
            if entry is None:
                # Stale index — entry was deleted
                continue
            if filter_fn is not None and not filter_fn(entry):
                continue
            # Convert distance based on space
            if self.config.space == "ip":
                # hnswlib returns negative inner product for cosine
                sim = -float(dist)
                results.append((entry, 1.0 - sim))  # distance = 1 - similarity
            else:
                results.append((entry, float(dist)))

        return results

    def range_search(
        self,
        query_vector: np.ndarray | list[float],
        radius: float,
        max_results: int = 100,
    ) -> list[tuple[Any, float]]:
        """Find all entries within *radius* of query_vector."""
        vec = np.array(query_vector, dtype=np.float32)
        # Use knn_search with large k, then filter by radius
        candidates = self.knn_search(vec, k=max_results * 2)
        return [(e, d) for e, d in candidates if d <= radius][:max_results]

    def get_novelty_neighbors(
        self,
        entry: Any,  # VectorTableEntry
        k: int = 5,
    ) -> list[tuple[Any, float]]:
        """Find the k nearest neighbors of *entry* and return distances.
        Novelty = average distance to neighbors."""
        neighbors = self.knn_search(entry.vector, k=k + 1)
        # Exclude self (first neighbor is usually self)
        neighbors = [(e, d) for e, d in neighbors if e.agent_id != entry.agent_id]
        return neighbors[:k]

    def compute_local_density(
        self,
        entry: Any,  # VectorTableEntry
        k: int = 10,
    ) -> float:
        """Local density = 1 / (average distance to k nearest neighbors).
        High density = region is crowded. Low density = sparse region."""
        neighbors = self.get_novelty_neighbors(entry, k=k)
        if not neighbors:
            return 0.0
        avg_dist = sum(d for _, d in neighbors) / len(neighbors)
        if avg_dist == 0:
            return float("inf")
        return 1.0 / avg_dist

    def find_sparse_regions(
        self,
        k: int = 10,
        n_samples: int = 20,
    ) -> list[tuple[Any, float]]:
        """Find entries in sparse (low density) regions.
        Returns (entry, density) sorted by density ascending."""
        all_entries = self.base.all_entries()
        if len(all_entries) < k + 1:
            return []

        # Sample for efficiency
        sample = all_entries
        if len(all_entries) > n_samples:
            indices = np.random.choice(len(all_entries), n_samples, replace=False)
            sample = [all_entries[i] for i in indices]

        results: list[tuple[Any, float]] = []
        for e in sample:
            density = self.compute_local_density(e, k=k)
            results.append((e, density))

        results.sort(key=lambda x: x[1])
        return results

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "table_id": self.base.table_id,
                "hnsw_available": _HNSWLIB_AVAILABLE,
                "index_count": self._index_count,
                "rebuild_count": self._rebuild_count,
                "insert_since_rebuild": self._insert_count_since_rebuild,
                **self.base.stats,
            }

    # ── internal ────────────────────────────────────────────

    def _rebuild_index(self) -> None:
        """Rebuild HNSW index from base table."""
        with self._lock:
            if not _HNSWLIB_AVAILABLE:
                self._hnsw_index = None
                return

            entries = self.base.all_entries()
            count = len(entries)
            max_elements = max(self.config.max_elements, count + 1000)

            space = hnswlib.Index(space=self.config.space, dim=self.config.dim)
            space.init_index(
                max_elements=max_elements,
                ef_construction=self.config.ef_construction,
                M=self.config.M,
            )
            space.set_ef(self.config.ef_search)

            self._id_to_label.clear()
            self._label_to_id.clear()
            self._next_label = 0

            if count > 0:
                vectors = np.stack([e.vector for e in entries])
                labels = np.arange(count, dtype=np.int64)
                space.add_items(vectors, labels)

                for i, e in enumerate(entries):
                    self._id_to_label[e.agent_id] = i
                    self._label_to_id[i] = e.agent_id

                self._next_label = count

            self._hnsw_index = space
            self._index_count = count
            self._insert_count_since_rebuild = 0
            self._rebuild_count += 1
            logger.info(
                "Rebuilt HNSW index for %s: %d entries (rebuild #%d)",
                self.base.table_id,
                count,
                self._rebuild_count,
            )

    def _add_to_hnsw(self, entry: Any) -> None:
        """Add a single entry to HNSW index."""
        if self._hnsw_index is None:
            return

        agent_id = entry.agent_id
        # Remove old label if exists
        old_label = self._id_to_label.get(agent_id)
        if old_label is not None:
            try:
                self._hnsw_index.mark_deleted(old_label)
            except Exception:
                pass

        label = self._next_label
        self._next_label += 1
        self._id_to_label[agent_id] = label
        self._label_to_id[label] = agent_id

        try:
            self._hnsw_index.add_items(
                entry.vector.reshape(1, -1),
                np.array([label], dtype=np.int64),
            )
            self._index_count += 1
        except Exception as exc:
            logger.warning("HNSW add_items failed: %s", exc)
            self._index_count -= 1

    def _maybe_trigger_rebuild(self) -> None:
        """Auto-rebuild if too many changes."""
        base_count = self.base.stats["entry_count"]
        if base_count == 0:
            return
        ratio = self._insert_count_since_rebuild / base_count
        if ratio >= self.auto_rebuild_threshold:
            self._rebuild_index()

    def _brute_force_knn(
        self,
        vec: np.ndarray,
        k: int,
        filter_fn: Any | None,
    ) -> list[tuple[Any, float]]:
        """Fallback brute-force search."""
        entries = self.base.all_entries()
        scored: list[tuple[float, Any]] = []
        for e in entries:
            if filter_fn is not None and not filter_fn(e):
                continue
            dist = float(np.linalg.norm(vec - e.vector))
            scored.append((dist, e))
        scored.sort(key=lambda x: x[0])
        return [(e, d) for d, e in scored[:k]]
