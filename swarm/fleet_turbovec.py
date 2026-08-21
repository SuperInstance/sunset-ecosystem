"""
FleetTurboVec — Rust-Accelerated Vector Index for Fleet Cognition

Wraps `RyanCodrai/turbovec` (TurboQuant algorithm, Rust + Python bindings)
to provide a high-performance vector index backend for FleetVectorIndex.

Key advantages over pure-NumPy:
- **16× compression** at 2-bit (31 GB → 4 GB for 10M documents)
- **No training step** — data-oblivious quantization, online ingest
- **Filtered search** at query time (mask/allowlist)
- **Faster than FAISS** on ARM and competitive on x86
- **SIMD-optimized** (AVX2, NEON) with blocked code layout

Integration Points
--------------------
- FleetVectorIndex: optional Rust-backed backend via ``backend="turbovec"``
- FleetDiversitySelector: re-rank turbovec search results with DPP/MMR
- MeshVectorGossip: serialized index snapshots for cross-node sync
- BreederDaemonV2: fast nearest-neighbor lookup for parent selection

API Mapping
-----------
| FleetTurboVec | turbovec.Rust | Purpose |
|---------------|---------------|---------|
| ``add_entries(entries)`` | ``IdMapIndex.add_with_ids()`` | Ingest with string→uint64 ID map |
| ``search(query, k, filter_fn)`` | ``IdMapIndex.search()`` | NN + fleet-level filtering |
| ``remove(agent_id)`` | ``IdMapIndex.remove()`` | O(1) by external ID |
| ``save(path)`` | ``IdMapIndex.write()`` | .tvim snapshot |
| ``load(path)`` | ``IdMapIndex.load()`` | Restore from .tvim |

Dependencies
------------
- ``turbovec`` (pip installable when Rust toolchain available)
- ``numpy``

References
----------
- turbovec: https://github.com/RyanCodrai/turbovec
- TurboQuant: Google Research, data-oblivious quantization
"""

from __future__ import annotations

__all__ = [
    "FleetTurboVecIndex",
    "TurboVecEntry",
    "TurboVecSearchResult",
    "TurboVecConfig",
]

import hashlib
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ── Data Classes ───────────────────────────────────────────────────────────


@dataclass
class TurboVecEntry:
    """A fleet entry stored in the TurboVec index."""

    agent_id: str
    vector: np.ndarray
    fitness: float = 0.0
    generation: int = 0
    node_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurboVecSearchResult:
    """Result from a TurboVec search."""

    agent_id: str
    score: float
    rank: int
    entry: Optional[TurboVecEntry] = None


@dataclass
class TurboVecConfig:
    """Configuration for FleetTurboVecIndex."""

    dim: Optional[int] = None  # None = lazy dimension from first add
    bit_width: int = 4  # 2, 4, or 8 bits per dimension
    diversity_rerank: bool = True  # Apply DPP/MMR after NN search
    diversity_k: int = 10  # Rerank top-diversity_k from NN
    diversity_strategy: str = "dpp"  # dpp, mmr, msd, cover, ssd
    diversity_lambda: float = 0.5  # Diversity-relevance tradeoff


# ── Core Adapter ────────────────────────────────────────────────────────────


class FleetTurboVecIndex:
    """Rust-accelerated vector index for fleet cognition.

    Wraps ``turbovec.IdMapIndex`` (external uint64 IDs) to store
    agent embeddings with O(1) lookup, deletion, and filtered search.

    When ``turbovec`` is not installed, falls back to a pure-NumPy
    brute-force index for compatibility.
    """

    def __init__(self, config: Optional[TurboVecConfig] = None) -> None:
        self.config = config or TurboVecConfig()
        self._entries: Dict[str, TurboVecEntry] = {}
        self._id_map: Dict[str, int] = {}  # agent_id → uint64 hash
        self._rev_map: Dict[int, str] = {}  # uint64 hash → agent_id
        self._index: Optional[Any] = None  # turbovec IdMapIndex or fallback
        self._fallback_vectors: Optional[np.ndarray] = None  # (n, dim) float32
        self._fallback_ids: List[str] = []
        self._ready: bool = False

        self._init_backend()

    # ── Public API ─────────────────────────────────────────────────────────

    def add_entries(self, entries: List[TurboVecEntry]) -> None:
        """Ingest entries into the index.

        :param entries: List of TurboVecEntry to add.
        """
        if not entries:
            return

        # Build batch arrays
        vectors = np.stack([e.vector for e in entries]).astype(np.float32)
        ids = np.array([self._hash_id(e.agent_id) for e in entries], dtype=np.uint64)

        # Update entry registry
        for e in entries:
            self._entries[e.agent_id] = e
            h = self._hash_id(e.agent_id)
            self._id_map[e.agent_id] = h
            self._rev_map[h] = e.agent_id

        if self._index is not None:
            # Rust backend
            self._index.add_with_ids(vectors, ids)
        else:
            # Fallback: append to numpy array
            if self._fallback_vectors is None:
                self._fallback_vectors = vectors
                self._fallback_ids = [e.agent_id for e in entries]
            else:
                self._fallback_vectors = np.vstack([self._fallback_vectors, vectors])
                self._fallback_ids.extend([e.agent_id for e in entries])

        self._ready = True

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filter_fn: Optional[Callable[[TurboVecEntry], bool]] = None,
        diversity_rerank: Optional[bool] = None,
    ) -> List[TurboVecSearchResult]:
        """Search for nearest neighbors with optional filtering and diversity re-ranking.

        :param query: Query vector (1-D float32).
        :param k: Number of results.
        :param filter_fn: Optional predicate ``(entry) -> bool`` for fleet-level filtering.
        :param diversity_rerank: Override config.diversity_rerank.
        :return: List of TurboVecSearchResult sorted by relevance/diversity.
        """
        if not self._ready:
            return []

        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        do_rerank = (
            diversity_rerank
            if diversity_rerank is not None
            else self.config.diversity_rerank
        )

        # Step 1: Retrieve coarse candidates from index
        if self._index is not None:
            coarse_k = max(k, self.config.diversity_k) if do_rerank else k
            if filter_fn is not None:
                # Build allowlist from filter_fn
                allowed_ids = [
                    self._hash_id(aid)
                    for aid, entry in self._entries.items()
                    if filter_fn(entry)
                ]
                if not allowed_ids:
                    return []
                allowlist = np.array(allowed_ids, dtype=np.uint64)
                scores, indices = self._index.search(
                    query, coarse_k, allowlist=allowlist
                )
            else:
                scores, indices = self._index.search(query, coarse_k)

            # Flatten single-query result
            scores = scores[0]
            indices = indices[0]
            candidates = []
            for rank, (score, idx) in enumerate(zip(scores, indices)):
                agent_id = self._rev_map.get(int(idx))
                if agent_id and agent_id in self._entries:
                    candidates.append((agent_id, float(score), rank))
        else:
            # Fallback: brute-force cosine similarity
            candidates = self._fallback_search(query[0], k, filter_fn)

        if not candidates:
            return []

        # Step 2: Diversity re-ranking (optional)
        if do_rerank and len(candidates) > k:
            candidates = self._diversity_rerank(candidates, k)

        # Step 3: Build results
        results = []
        for rank, (agent_id, score, _) in enumerate(candidates[:k]):
            entry = self._entries.get(agent_id)
            results.append(
                TurboVecSearchResult(
                    agent_id=agent_id,
                    score=score,
                    rank=rank,
                    entry=entry,
                )
            )
        return results

    def remove(self, agent_id: str) -> bool:
        """Remove an entry from the index.

        :param agent_id: Agent ID to remove.
        :return: True if removed.
        """
        if agent_id not in self._entries:
            return False

        h = self._id_map.pop(agent_id, None)
        if h is not None:
            self._rev_map.pop(h, None)

        self._entries.pop(agent_id, None)

        if self._index is not None:
            self._index.remove(h)
        else:
            # Fallback: rebuild index (inefficient but correct)
            self._rebuild_fallback()

        return True

    def save(self, path: str) -> None:
        """Serialize index to disk.

        :param path: Directory path to save (.tvim + metadata.json).
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            try:
                self._index.write(str(p / "index.tvim"))
            except Exception as exc:
                log.warning("turbovec write failed: %s; falling back to numpy", exc)
                if self._fallback_vectors is not None:
                    np.save(p / "vectors.npy", self._fallback_vectors)
                    import json as _json

                    with open(p / "ids.json", "w") as f:
                        _json.dump(self._fallback_ids, f)
        elif self._fallback_vectors is not None:
            np.save(p / "vectors.npy", self._fallback_vectors)
            import json as _json

            with open(p / "ids.json", "w") as f:
                _json.dump(self._fallback_ids, f)

        # Save metadata
        meta = {
            "n_entries": len(self._entries),
            "config": {
                "dim": self.config.dim,
                "bit_width": self.config.bit_width,
                "diversity_rerank": self.config.diversity_rerank,
                "diversity_k": self.config.diversity_k,
                "diversity_strategy": self.config.diversity_strategy,
                "diversity_lambda": self.config.diversity_lambda,
            },
            "id_map": self._id_map,
        }
        import json as _json

        with open(p / "metadata.json", "w") as f:
            _json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FleetTurboVecIndex":
        """Restore index from disk.

        :param path: Directory path previously saved with ``save()``.
        :return: Restored FleetTurboVecIndex.
        """
        p = Path(path)
        import json as _json

        with open(p / "metadata.json") as f:
            meta = _json.load(f)

        config = TurboVecConfig(**meta["config"])
        inst = cls(config)
        inst._id_map = meta.get("id_map", {})
        inst._rev_map = {v: k for k, v in inst._id_map.items()}

        tvim_path = p / "index.tvim"
        if tvim_path.exists():
            try:
                import turbovec

                inst._index = turbovec.IdMapIndex.load(str(tvim_path))
                inst._ready = True
            except ImportError:
                log.warning("turbovec not installed; cannot load .tvim")
        else:
            vectors_path = p / "vectors.npy"
            if vectors_path.exists():
                inst._fallback_vectors = np.load(vectors_path)
                with open(p / "ids.json") as f:
                    inst._fallback_ids = _json.load(f)
                inst._ready = True

        return inst

    def prepare(self) -> None:
        """Warm up search caches (rotation matrix, centroids, SIMD layout).

        Call once after adding all entries and before heavy search load.
        """
        if self._index is not None:
            self._index.prepare()

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        backend = "turbovec" if self._index is not None else "numpy"
        return (
            f"FleetTurboVecIndex("
            f"backend={backend}, "
            f"n={len(self)}, "
            f"dim={self.config.dim}, "
            f"bit_width={self.config.bit_width})"
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _init_backend(self) -> None:
        """Try to create turbovec IdMapIndex; fall back to numpy."""
        try:
            import turbovec

            self._index = turbovec.IdMapIndex(
                dim=self.config.dim,
                bit_width=self.config.bit_width,
            )
            log.info("FleetTurboVecIndex: turbovec backend initialized")
        except ImportError:
            log.warning("turbovec not installed; using numpy fallback")
            self._index = None

    @staticmethod
    def _hash_id(agent_id: str) -> int:
        """Stable uint64 hash from agent_id string."""
        # Use first 8 bytes of SHA-256 as uint64
        digest = hashlib.sha256(agent_id.encode("utf-8")).digest()
        return struct.unpack("Q", digest[:8])[0]

    def _fallback_search(
        self,
        query: np.ndarray,
        k: int,
        filter_fn: Optional[Callable[[TurboVecEntry], bool]],
    ) -> List[Tuple[str, float, int]]:
        """Brute-force cosine similarity search (fallback)."""
        if self._fallback_vectors is None or len(self._fallback_ids) == 0:
            return []

        # Compute cosine similarities
        q_norm = query / (np.linalg.norm(query) + 1e-10)
        v_norms = np.linalg.norm(self._fallback_vectors, axis=1, keepdims=True)
        v_normed = self._fallback_vectors / (v_norms + 1e-10)
        sims = v_normed @ q_norm

        # Apply filter
        if filter_fn is not None:
            mask = np.array(
                [
                    filter_fn(
                        self._entries.get(
                            aid, TurboVecEntry(agent_id=aid, vector=np.zeros(1))
                        )
                    )
                    for aid in self._fallback_ids
                ]
            )
            sims = np.where(mask, sims, -np.inf)

        # Top-k
        top_k = min(k, len(sims))
        top_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        return [
            (self._fallback_ids[i], float(sims[i]), rank)
            for rank, i in enumerate(top_idx)
        ]

    def _rebuild_fallback(self) -> None:
        """Rebuild fallback index after deletion."""
        remaining = [
            (aid, self._entries[aid].vector)
            for aid in self._fallback_ids
            if aid in self._entries
        ]
        if remaining:
            self._fallback_ids = [aid for aid, _ in remaining]
            self._fallback_vectors = np.stack([vec for _, vec in remaining]).astype(
                np.float32
            )
        else:
            self._fallback_ids = []
            self._fallback_vectors = None

    def _diversity_rerank(
        self,
        candidates: List[Tuple[str, float, int]],
        k: int,
    ) -> List[Tuple[str, float, int]]:
        """Apply diversity re-ranking to coarse candidates."""
        try:
            from swarm.fleet_diversity import (
                FleetDiversitySelector,
                DiversityStrategy,
                PopulationItem,
            )

            strategy_map = {
                "dpp": DiversityStrategy.DPP,
                "mmr": DiversityStrategy.MMR,
                "msd": DiversityStrategy.MSD,
                "cover": DiversityStrategy.COVER,
                "ssd": DiversityStrategy.SSD,
            }
            strategy = strategy_map.get(
                self.config.diversity_strategy, DiversityStrategy.DPP
            )

            pop = [
                PopulationItem(
                    id=aid,
                    embedding=self._entries[aid].vector,
                    fitness=score,
                )
                for aid, score, _ in candidates
                if aid in self._entries
            ]
            if len(pop) <= k:
                return candidates

            sel = FleetDiversitySelector(
                strategy=strategy,
                diversity=self.config.diversity_lambda,
                default_k=k,
            )
            diverse = sel.select_parents(pop, k=k)
            diverse_ids = {d.id for d in diverse}

            # Preserve original scores but reorder
            score_map = {aid: score for aid, score, _ in candidates}
            return [(d.id, score_map.get(d.id, 0.0), 0) for d in diverse]
        except Exception as exc:
            log.warning("Diversity re-rank failed: %s; returning raw candidates", exc)
            return candidates

    # ── Integration Helpers ─────────────────────────────────────────────────

    @classmethod
    def from_fleet_vector_index(
        cls,
        fvi: Any,  # FleetVectorIndex
        config: Optional[TurboVecConfig] = None,
    ) -> "FleetTurboVecIndex":
        """Migrate entries from a FleetVectorIndex into TurboVec.

        :param fvi: FleetVectorIndex instance.
        :param config: TurboVecConfig (optional).
        :return: New FleetTurboVecIndex with all fleet entries.
        """
        inst = cls(config)
        entries = fvi.all_entries_fleet_wide()
        tve_entries = [
            TurboVecEntry(
                agent_id=e.agent_id,
                vector=e.vector,
                fitness=e.fitness,
                generation=e.generation,
                node_id=e.node_id,
                metadata={
                    "thermal_pressure": e.thermal_pressure,
                    "capability_mask": e.capability_mask,
                },
            )
            for e in entries
        ]
        inst.add_entries(tve_entries)
        return inst

    def to_fleet_entries(self) -> List[Any]:
        """Export all entries as VectorTableEntry-compatible dicts.

        :return: List of entries for FleetVectorIndex ingestion.
        """
        from swarm.mesh_vector_tables import VectorTableEntry

        results = []
        for e in self._entries.values():
            results.append(
                VectorTableEntry(
                    agent_id=e.agent_id,
                    vector=e.vector,
                    timestamp=0.0,
                    node_id=e.node_id,
                    generation=e.generation,
                    fitness=e.fitness,
                    signature="",
                    capability_mask=e.metadata.get("capability_mask", 0xFFFF),
                    thermal_pressure=e.metadata.get("thermal_pressure", 0.0),
                    extra=e.metadata,
                )
            )
        return results
