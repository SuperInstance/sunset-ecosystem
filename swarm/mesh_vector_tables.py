"""MeshVectorTables — Federated CRDT-based vector tables for shared fleet cognition.

Provides structured vector tables on top of the MeshVectorGossip anti-entropy
layer, enabling cross-node sharing of:
- Diversity scores and population state
- Breeding history (generation lineage)
- Agent capabilities (skill masks)

Integration touchpoints
-----------------------
- MeshVectorGossip: carries ``MeshVectorTable.get_sync_payload()`` as gossip delta.
- HebbianMeshLayer: uses ``FleetVectorIndex.get_novelty_score()`` for chaos routing.
- BreederDaemonV2: queries ``FleetVectorIndex.get_breedable_pool()`` for cross-node parents.
- AgentIdentity: signs every table entry via ``sign_task()``.

Reference: docs/SPEC_MULTI_INSTANCE_MESH.md — sections 3 & 4.
docs/MESH_VECTOR_TABLES.md — architecture guide.
"""

from __future__ import annotations

__all__ = [
    "VectorTableEntry",
    "MeshVectorTable",
    "FleetVectorIndex",
    "SignatureError",
    "MergeConflictError",
]

import base64
import hashlib
import json
import logging
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── exceptions ──────────────────────────────────────────────────


class SignatureError(ValueError):
    """Raised when an entry signature fails verification."""

    pass


class MergeConflictError(RuntimeError):
    """Raised when a CRDT merge encounters an unresolvable conflict."""

    pass


# ── VectorTableEntry ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class VectorTableEntry:
    """A single row in a MeshVectorTable.

    Immutable, hashable, and comparable for CRDT conflict resolution.
    The *signature* field is produced by ``AgentIdentity.sign_task()`` over
    the canonical JSON of the other fields (excluding signature itself).
    """

    agent_id: str  # e.g. "Oracle1::agent_42"
    vector: np.ndarray  # float32, shape (dim,)
    timestamp: float  # Unix time, monotonic from the creating node
    node_id: str  # e.g. "Oracle1"
    generation: int  # Breeding generation (0 = seed)
    fitness: float  # [0, 1], trinity product
    signature: str  # base64 Ed25519 or SHA-256 fallback
    capability_mask: int = 0xFFFF  # 16-bit skill mask
    thermal_pressure: float = 0.0  # [0, 1]
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure vector is a contiguous float32 numpy array
        if not isinstance(self.vector, np.ndarray):
            object.__setattr__(self, "vector", np.array(self.vector, dtype=np.float32))
        else:
            object.__setattr__(
                self, "vector", self.vector.astype(np.float32, copy=False)
            )

    @property
    def dim(self) -> int:
        return int(self.vector.shape[0])

    def to_dict(self, include_vector: bool = True) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON / gossip."""
        d: dict[str, Any] = {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "generation": self.generation,
            "fitness": self.fitness,
            "signature": self.signature,
            "capability_mask": self.capability_mask,
            "thermal_pressure": self.thermal_pressure,
            "extra": dict(self.extra),
        }
        if include_vector:
            # Base64-encoded float32 bytes for compact transport
            d["vector_b64"] = base64.b64encode(self.vector.tobytes()).decode("ascii")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VectorTableEntry":
        """Reconstruct from a plain dict (inverse of ``to_dict``)."""
        vec: np.ndarray
        if "vector_b64" in d:
            raw = base64.b64decode(d["vector_b64"])
            vec = np.frombuffer(raw, dtype=np.float32).copy()
        elif "vector" in d:
            vec = np.array(d["vector"], dtype=np.float32)
        else:
            raise ValueError("Dict missing 'vector_b64' or 'vector' field")
        return cls(
            agent_id=str(d["agent_id"]),
            vector=vec,
            timestamp=float(d["timestamp"]),
            node_id=str(d["node_id"]),
            generation=int(d.get("generation", 0)),
            fitness=float(d.get("fitness", 0.0)),
            signature=str(d["signature"]),
            capability_mask=int(d.get("capability_mask", 0xFFFF)),
            thermal_pressure=float(d.get("thermal_pressure", 0.0)),
            extra=dict(d.get("extra", {})),
        )

    def canonical_payload(self) -> dict[str, Any]:
        """Return the payload that was (or should be) signed."""
        return {
            "agent_id": self.agent_id,
            "vector_b64": base64.b64encode(self.vector.tobytes()).decode("ascii"),
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "generation": self.generation,
            "fitness": self.fitness,
            "capability_mask": self.capability_mask,
            "thermal_pressure": self.thermal_pressure,
            "extra": self.extra,
        }


# ── MeshVectorTable ─────────────────────────────────────────────


class MeshVectorTable:
    """CRDT vector table for a single domain (one generation, or one skill).

    Thread-safe.  All public methods acquire an internal RLock.

    Parameters
    ----------
    table_id : str
        Unique identifier for this table, e.g. "gen_5" or "skill_flux".
    identity : AgentIdentity | None
        If provided, all ``insert()`` calls are signed automatically.
        If *None*, signatures must be supplied by the caller.
    """

    def __init__(self, table_id: str, identity: Any | None = None) -> None:
        self.table_id = table_id
        self._identity = identity
        self._lock = threading.RLock()

        # Core storage: agent_id -> VectorTableEntry
        self._entries: dict[str, VectorTableEntry] = {}

        # Indexes (rebuilt on demand, cached for performance)
        self._fitness_index: list[
            tuple[str, float]
        ] = []  # (agent_id, fitness) sorted desc
        self._fitness_index_stale: bool = True
        self._node_breakdown: dict[str, int] = {}
        self._node_breakdown_stale: bool = True

        # Statistics
        self._insert_count: int = 0
        self._merge_count: int = 0
        self._reject_count: int = 0

    # ── insertion ───────────────────────────────────────────

    def insert(
        self,
        entry: VectorTableEntry,
        skip_verify: bool = False,
    ) -> bool:
        """Add or overwrite an entry after signature verification.

        Parameters
        ----------
        entry : VectorTableEntry
            The row to insert.
        skip_verify : bool
            If True, skip signature check (useful for locally-generated
            entries or testing).  Default False.

        Returns
        -------
        bool
            True if the entry was accepted (and either inserted or
            merged).  False if rejected (bad signature or CRDT loser).

        Raises
        ------
        SignatureError
            If signature verification fails and *skip_verify* is False.
        """
        with self._lock:
            if not skip_verify:
                self._verify_entry(entry)

            existing = self._entries.get(entry.agent_id)
            if existing is not None:
                winner = self._crdt_winner(existing, entry)
                if winner is existing:
                    self._reject_count += 1
                    return False
                # winner is entry — fall through to overwrite

            self._entries[entry.agent_id] = entry
            self._insert_count += 1
            self._fitness_index_stale = True
            self._node_breakdown_stale = True
            logger.debug(
                "Inserted entry %s into table %s (gen=%d, fitness=%.3f)",
                entry.agent_id,
                self.table_id,
                entry.generation,
                entry.fitness,
            )
            return True

    def insert_signed(
        self,
        agent_id: str,
        vector: np.ndarray | list[float],
        node_id: str,
        generation: int = 0,
        fitness: float = 0.0,
        capability_mask: int = 0xFFFF,
        thermal_pressure: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> VectorTableEntry:
        """Create, sign (if identity available), and insert an entry.

        Returns the fully-constructed entry (useful for tests).
        """
        if self._identity is None:
            raise RuntimeError(
                "Cannot auto-sign: MeshVectorTable was created without an AgentIdentity"
            )
        entry = VectorTableEntry(
            agent_id=agent_id,
            vector=np.array(vector, dtype=np.float32),
            timestamp=time.time(),
            node_id=node_id,
            generation=generation,
            fitness=fitness,
            signature="",  # placeholder
            capability_mask=capability_mask,
            thermal_pressure=thermal_pressure,
            extra=extra or {},
        )
        payload = entry.canonical_payload()
        entry = VectorTableEntry(
            agent_id=entry.agent_id,
            vector=entry.vector,
            timestamp=entry.timestamp,
            node_id=entry.node_id,
            generation=entry.generation,
            fitness=entry.fitness,
            signature=self._identity.sign_task(payload),
            capability_mask=entry.capability_mask,
            thermal_pressure=entry.thermal_pressure,
            extra=entry.extra,
        )
        self.insert(entry, skip_verify=True)
        return entry

    # ── queries ─────────────────────────────────────────────

    def query(self, agent_id: str) -> VectorTableEntry | None:
        """Return the entry for *agent_id*, or None if absent."""
        with self._lock:
            return self._entries.get(agent_id)

    def query_by_fitness(
        self,
        min_fitness: float = 0.0,
        max_results: int = 50,
    ) -> list[VectorTableEntry]:
        """Return entries with fitness >= *min_fitness*, sorted descending.

        Parameters
        ----------
        min_fitness : float
            Lower-bound fitness threshold.
        max_results : int
            Maximum number of entries to return.

        Returns
        -------
        list[VectorTableEntry]
            Sorted best-first.
        """
        with self._lock:
            self._rebuild_fitness_index_if_stale()
            results: list[VectorTableEntry] = []
            for aid, fit in self._fitness_index:
                if fit < min_fitness:
                    break
                entry = self._entries.get(aid)
                if entry is not None:
                    results.append(entry)
                if len(results) >= max_results:
                    break
            return results

    def query_by_diversity(
        self,
        reference_vector: np.ndarray | list[float],
        min_distance: float = 0.1,
        max_results: int = 50,
        metric: str = "euclidean",
    ) -> list[VectorTableEntry]:
        """Return entries whose vector is at least *min_distance* from
        the *reference_vector*.

        Parameters
        ----------
        reference_vector : np.ndarray
            The centre from which we measure divergence.
        min_distance : float
            Minimum distance required for inclusion.
        max_results : int
            Cap on returned entries.
        metric : str
            "euclidean" or "cosine".

        Returns
        -------
        list[VectorTableEntry]
            Entries sorted by distance descending (most diverse first).
        """
        ref = np.array(reference_vector, dtype=np.float32)
        with self._lock:
            scored: list[tuple[float, VectorTableEntry]] = []
            for entry in self._entries.values():
                d = self._compute_distance(ref, entry.vector, metric)
                if d >= min_distance:
                    scored.append((d, entry))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [e for _, e in scored[:max_results]]

    def get_population_summary(self) -> dict[str, Any]:
        """Return a snapshot of the current population.

        Returns
        -------
        dict
            - count : int
            - mean_fitness : float
            - diversity_score : float  (0.0 = collapse, 1.0 = max spread)
            - node_breakdown : dict[str, int]
            - generation_range : tuple[int, int] | None
        """
        with self._lock:
            count = len(self._entries)
            if count == 0:
                return {
                    "count": 0,
                    "mean_fitness": 0.0,
                    "diversity_score": 0.0,
                    "node_breakdown": {},
                    "generation_range": None,
                }

            fitnesses = [e.fitness for e in self._entries.values()]
            mean_fitness = sum(fitnesses) / count

            vectors = np.stack([e.vector for e in self._entries.values()])
            diversity_score = self._compute_diversity_score(vectors)

            self._rebuild_node_breakdown_if_stale()
            gen_min = min(e.generation for e in self._entries.values())
            gen_max = max(e.generation for e in self._entries.values())

            return {
                "count": count,
                "mean_fitness": mean_fitness,
                "diversity_score": diversity_score,
                "node_breakdown": dict(self._node_breakdown),
                "generation_range": (gen_min, gen_max),
            }

    # ── CRDT merge ──────────────────────────────────────────

    def merge_remote_table(self, remote_table: "MeshVectorTable") -> dict[str, Any]:
        """CRDT-merge all entries from *remote_table* into this table.

        Conflict resolution:
        1. Higher timestamp wins (the node that wrote later is authoritative).
        2. If timestamps are equal, tie-break by signature hash — the entry
           whose signature has the **lower** SHA-256 hash wins.  This is
           deterministic and unbiased.
        3. Vectors are deep-copied so the tables remain independent.

        Parameters
        ----------
        remote_table : MeshVectorTable
            The foreign table to merge.  Must have a different *table_id*
            or be from a different node (same table_id is allowed for tests).

        Returns
        -------
        dict
            Statistics: merged, rejected, skipped.
        """
        with self._lock:
            merged = 0
            rejected = 0
            skipped = 0

            # Snapshot remote entries (avoid holding both locks — remote is
            # responsible for its own thread safety).
            remote_entries = list(remote_table.all_entries())

            for remote_entry in remote_entries:
                existing = self._entries.get(remote_entry.agent_id)
                if existing is None:
                    self._entries[remote_entry.agent_id] = VectorTableEntry(
                        agent_id=remote_entry.agent_id,
                        vector=remote_entry.vector.copy(),
                        timestamp=remote_entry.timestamp,
                        node_id=remote_entry.node_id,
                        generation=remote_entry.generation,
                        fitness=remote_entry.fitness,
                        signature=remote_entry.signature,
                        capability_mask=remote_entry.capability_mask,
                        thermal_pressure=remote_entry.thermal_pressure,
                        extra=dict(remote_entry.extra),
                    )
                    merged += 1
                    self._fitness_index_stale = True
                    self._node_breakdown_stale = True
                else:
                    winner = self._crdt_winner(existing, remote_entry)
                    if winner is remote_entry:
                        self._entries[remote_entry.agent_id] = VectorTableEntry(
                            agent_id=remote_entry.agent_id,
                            vector=remote_entry.vector.copy(),
                            timestamp=remote_entry.timestamp,
                            node_id=remote_entry.node_id,
                            generation=remote_entry.generation,
                            fitness=remote_entry.fitness,
                            signature=remote_entry.signature,
                            capability_mask=remote_entry.capability_mask,
                            thermal_pressure=remote_entry.thermal_pressure,
                            extra=dict(remote_entry.extra),
                        )
                        merged += 1
                        self._fitness_index_stale = True
                        self._node_breakdown_stale = True
                    elif winner is existing:
                        rejected += 1
                    else:
                        skipped += 1

            self._merge_count += merged
            logger.info(
                "Merged table %s ← remote: %d merged, %d rejected, %d skipped",
                self.table_id,
                merged,
                rejected,
                skipped,
            )
            return {"merged": merged, "rejected": rejected, "skipped": skipped}

    # ── sync payloads (gossip wire format) ──────────────────

    def get_sync_payload(self) -> bytes:
        """Return a compressed, wire-ready payload of all entries.

        The payload is zlib-compressed JSON.  It can be carried by
        ``MeshVectorGossip`` as a gossip delta.

        Returns
        -------
        bytes
            zlib-compressed blob.
        """
        with self._lock:
            serial = {
                "table_id": self.table_id,
                "timestamp": time.time(),
                "entries": [
                    e.to_dict(include_vector=True) for e in self._entries.values()
                ],
            }
            json_bytes = json.dumps(serial, separators=(",", ":")).encode("utf-8")
            return zlib.compress(json_bytes, level=6)

    def apply_sync_payload(self, payload: bytes) -> dict[str, Any]:
        """Decode and merge a sync payload produced by ``get_sync_payload``.

        Parameters
        ----------
        payload : bytes
            The compressed blob.

        Returns
        -------
        dict
            Statistics: merged, rejected, errors.
        """
        with self._lock:
            try:
                json_bytes = zlib.decompress(payload)
                serial = json.loads(json_bytes.decode("utf-8"))
            except Exception as exc:
                logger.warning("Failed to decode sync payload: %s", exc)
                return {"merged": 0, "rejected": 0, "errors": [str(exc)]}

            merged = 0
            rejected = 0
            errors: list[str] = []

            for entry_dict in serial.get("entries", []):
                try:
                    entry = VectorTableEntry.from_dict(entry_dict)
                    ok = self.insert(entry, skip_verify=False)
                    if ok:
                        merged += 1
                    else:
                        rejected += 1
                except Exception as exc:
                    errors.append(str(exc))
                    logger.debug("Sync payload entry failed: %s", exc)

            return {"merged": merged, "rejected": rejected, "errors": errors}

    # ── bulk helpers ────────────────────────────────────────

    def all_entries(self) -> list[VectorTableEntry]:
        """Return a shallow copy of all entries (safe to iterate)."""
        with self._lock:
            return list(self._entries.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "table_id": self.table_id,
                "entry_count": len(self._entries),
                "insert_count": self._insert_count,
                "merge_count": self._merge_count,
                "reject_count": self._reject_count,
            }

    # ── internal CRDT & index helpers ───────────────────────

    def _crdt_winner(
        self,
        local: VectorTableEntry,
        remote: VectorTableEntry,
    ) -> VectorTableEntry:
        """Return the winning entry under CRDT rules.

        1. Higher timestamp wins.
        2. If equal timestamp, lower SHA-256(signature) wins (deterministic).
        """
        if remote.timestamp > local.timestamp:
            return remote
        if local.timestamp > remote.timestamp:
            return local
        # Timestamps equal — tiebreak by signature hash
        remote_hash = hashlib.sha256(remote.signature.encode("utf-8")).hexdigest()
        local_hash = hashlib.sha256(local.signature.encode("utf-8")).hexdigest()
        if remote_hash < local_hash:
            return remote
        return local

    def _verify_entry(self, entry: VectorTableEntry) -> None:
        """Verify the entry's signature against its canonical payload.

        If the table has no AgentIdentity, we fall back to a length check
        (accepts SHA-256 fallback signatures from environments without
        Ed25519).
        """
        if self._identity is None:
            # No identity available — accept any non-empty signature or
            # the 64-char SHA-256 fallback.
            if not entry.signature or len(entry.signature) < 8:
                raise SignatureError(f"Entry {entry.agent_id} has invalid signature")
            return

        payload = entry.canonical_payload()
        ok = self._identity.verify_task(payload, entry.signature)
        if not ok:
            raise SignatureError(
                f"Entry {entry.agent_id} failed signature verification"
            )

    def _rebuild_fitness_index_if_stale(self) -> None:
        if not self._fitness_index_stale:
            return
        self._fitness_index = sorted(
            ((aid, e.fitness) for aid, e in self._entries.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        self._fitness_index_stale = False

    def _rebuild_node_breakdown_if_stale(self) -> None:
        if not self._node_breakdown_stale:
            return
        breakdown: dict[str, int] = {}
        for e in self._entries.values():
            breakdown[e.node_id] = breakdown.get(e.node_id, 0) + 1
        self._node_breakdown = breakdown
        self._node_breakdown_stale = False

    @staticmethod
    def _compute_distance(
        a: np.ndarray,
        b: np.ndarray,
        metric: str = "euclidean",
    ) -> float:
        if metric == "euclidean":
            return float(np.linalg.norm(a - b))
        if metric == "cosine":
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(1.0 - np.dot(a, b) / (na * nb))
        raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def _compute_diversity_score(vectors: np.ndarray) -> float:
        """0.0 = all identical, 1.0 = maximally spread."""
        if vectors.shape[0] <= 1:
            return 0.0
        centroid = np.mean(vectors, axis=0)
        distances = np.linalg.norm(vectors - centroid, axis=1)
        avg_distance = float(np.mean(distances))
        dim = vectors.shape[1]
        healthy_spread = 0.5 * np.sqrt(dim)
        return min(1.0, avg_distance / healthy_spread)


# ── FleetVectorIndex ────────────────────────────────────────────


class FleetVectorIndex:
    """Manages multiple ``MeshVectorTables`` across generations and
    capability types, providing fleet-wide queries.

    Each generation gets its own ``MeshVectorTable`` so that lineage
    queries are fast.  Capability tables are synthetic views — they
    reference the same underlying entries by agent_id.

    Parameters
    ----------
    node_id : str
        Identifier for the local node (used in auto-generated entries).
    identity : AgentIdentity | None
        Shared identity for signing entries in all sub-tables.
    """

    def __init__(self, node_id: str, identity: Any | None = None) -> None:
        self.node_id = node_id
        self._identity = identity
        self._lock = threading.RLock()

        # Generation tables: generation_int -> MeshVectorTable
        self._gen_tables: dict[int, MeshVectorTable] = {}

        # Capability tables: skill_name -> MeshVectorTable
        # These are *views* — they contain entries copied from gen tables.
        self._skill_tables: dict[str, MeshVectorTable] = {}

        # Fleet-wide centroid (cached, rebuilt on demand)
        self._fleet_centroid: np.ndarray | None = None
        self._fleet_centroid_stale: bool = True

    # ── table management ────────────────────────────────────

    def get_gen_table(self, generation: int) -> MeshVectorTable:
        """Return (or create) the table for a specific generation."""
        with self._lock:
            table = self._gen_tables.get(generation)
            if table is None:
                table = MeshVectorTable(
                    table_id=f"gen_{generation}",
                    identity=self._identity,
                )
                self._gen_tables[generation] = table
            return table

    def get_skill_table(self, skill_name: str) -> MeshVectorTable:
        """Return (or create) the table for a specific skill."""
        with self._lock:
            table = self._skill_tables.get(skill_name)
            if table is None:
                table = MeshVectorTable(
                    table_id=f"skill_{skill_name}",
                    identity=self._identity,
                )
                self._skill_tables[skill_name] = table
            return table

    def insert_fleet_entry(self, entry: VectorTableEntry) -> bool:
        """Insert an entry into the appropriate generation table *and*
        every skill table that matches its capability mask.

        Returns True if the entry was accepted into the generation table.
        """
        with self._lock:
            gen_table = self.get_gen_table(entry.generation)
            ok = gen_table.insert(entry, skip_verify=False)
            if not ok:
                return False

            # Update skill views
            for skill_name, skill_table in self._skill_tables.items():
                if self._entry_has_skill(entry, skill_name):
                    skill_table.insert(entry, skip_verify=True)

            self._fleet_centroid_stale = True
            return True

    def all_entries_fleet_wide(self) -> list[VectorTableEntry]:
        """Return a deduplicated list of all entries across all generations."""
        with self._lock:
            seen: dict[str, VectorTableEntry] = {}
            for table in self._gen_tables.values():
                for e in table.all_entries():
                    seen[e.agent_id] = e
            return list(seen.values())

    # ── fleet-wide queries ──────────────────────────────────

    def get_breedable_pool(
        self,
        min_fitness: float = 0.7,
        max_thermal: float = 0.5,
        diversity_target: float = 0.3,
        max_results: int = 20,
    ) -> list[VectorTableEntry]:
        """Return cross-node parent candidates that meet all filters.

        Parameters
        ----------
        min_fitness : float
            Minimum fitness threshold.
        max_thermal : float
            Maximum thermal pressure (hot agents are excluded).
        diversity_target : float
            Desired minimum distance from the fleet centroid.  Entries
            closer than this are considered "too mainstream" and skipped
            unless the pool would be empty.
        max_results : int
            Maximum candidates to return.

        Returns
        -------
        list[VectorTableEntry]
            Sorted by fitness descending.
        """
        with self._lock:
            candidates: list[VectorTableEntry] = []
            centroid = self._get_fleet_centroid()

            for table in self._gen_tables.values():
                for entry in table.all_entries():
                    if entry.fitness < min_fitness:
                        continue
                    if entry.thermal_pressure > max_thermal:
                        continue
                    if centroid is not None:
                        dist = MeshVectorTable._compute_distance(
                            centroid, entry.vector, metric="euclidean"
                        )
                        if dist < diversity_target:
                            continue
                    candidates.append(entry)

            candidates.sort(key=lambda e: e.fitness, reverse=True)
            return candidates[:max_results]

    def get_capability_map(self, skill_name: str) -> dict[str, list[str]]:
        """Return a mapping of node_id → list of agent_ids that possess
        *skill_name*, across all nodes in the fleet.

        Parameters
        ----------
        skill_name : str
            The skill to look up.

        Returns
        -------
        dict[str, list[str]]
            node_id -> [agent_id, ...]
        """
        with self._lock:
            mapping: dict[str, list[str]] = {}
            for table in self._gen_tables.values():
                for entry in table.all_entries():
                    if self._entry_has_skill(entry, skill_name):
                        mapping.setdefault(entry.node_id, []).append(entry.agent_id)
            return mapping

    def get_novelty_score(
        self,
        agent_vector: np.ndarray | list[float],
        population_vectors: list[np.ndarray] | None = None,
    ) -> float:
        """Compute how novel *agent_vector* is versus the entire fleet population.

        The score is the Euclidean distance from the fleet centroid,
        normalised by the expected spread of a healthy fleet.

        As the fleet population grows, the centroid becomes more stable and
        the effective novelty of any single vector decreases — this is
        intentional (the fleet is converging).

        Parameters
        ----------
        agent_vector : np.ndarray
            The vector to evaluate.
        population_vectors : list[np.ndarray] | None
            If provided, use this instead of the current fleet population.

        Returns
        -------
        float
            0.0 = identical to fleet average, 1.0 = maximally divergent.
        """
        vec = np.array(agent_vector, dtype=np.float32)
        with self._lock:
            if population_vectors is None:
                all_vecs = [e.vector for e in self.all_entries_fleet_wide()]
            else:
                all_vecs = population_vectors

            if not all_vecs:
                # Empty fleet — everything is maximally novel
                return 1.0

            pop = np.stack(all_vecs)
            centroid = np.mean(pop, axis=0)
            distance = float(np.linalg.norm(vec - centroid))

            dim = vec.shape[0]
            healthy_spread = 0.5 * np.sqrt(dim)
            # Novelty *decreases* as population grows: scale by log(count)
            count_factor = 1.0 / (1.0 + 0.1 * np.log1p(len(all_vecs)))
            score = min(1.0, (distance / healthy_spread) * count_factor)
            return score

    # ── sync helpers (fleet-wide) ───────────────────────────

    def get_fleet_sync_payload(self) -> bytes:
        """Compress all generation tables into a single payload.

        Returns
        -------
        bytes
            zlib-compressed JSON containing every generation table.
        """
        with self._lock:
            serial = {
                "node_id": self.node_id,
                "timestamp": time.time(),
                "tables": {},
            }
            for gen, table in self._gen_tables.items():
                serial["tables"][str(gen)] = [
                    e.to_dict(include_vector=True) for e in table.all_entries()
                ]
            json_bytes = json.dumps(serial, separators=(",", ":")).encode("utf-8")
            return zlib.compress(json_bytes, level=6)

    def apply_fleet_sync_payload(self, payload: bytes) -> dict[str, Any]:
        """Decode a fleet-wide payload and merge into all generation tables.

        Returns
        -------
        dict
            Per-generation merge stats.
        """
        with self._lock:
            try:
                json_bytes = zlib.decompress(payload)
                serial = json.loads(json_bytes.decode("utf-8"))
            except Exception as exc:
                return {"errors": [str(exc)]}

            stats: dict[str, Any] = {"errors": [], "per_gen": {}}
            for gen_str, entries in serial.get("tables", {}).items():
                generation = int(gen_str)
                table = self.get_gen_table(generation)
                merged = 0
                rejected = 0
                for entry_dict in entries:
                    try:
                        entry = VectorTableEntry.from_dict(entry_dict)
                        ok = table.insert(entry, skip_verify=False)
                        if ok:
                            merged += 1
                        else:
                            rejected += 1
                    except Exception as exc:
                        stats["errors"].append(str(exc))
                stats["per_gen"][gen_str] = {"merged": merged, "rejected": rejected}
                self._fleet_centroid_stale = True
            return stats

    # ── internal helpers ────────────────────────────────────

    def _entry_has_skill(self, entry: VectorTableEntry, skill_name: str) -> bool:
        """Check whether *entry* advertises *skill_name* via capability mask.

        For now we use a simple string-to-bit mapping.  In production this
        would be driven by a canonical skill registry.
        """
        # Simple hash-to-bit mapping for 16-bit mask
        skill_hash = hashlib.sha256(skill_name.encode("utf-8")).hexdigest()
        bit_idx = int(skill_hash, 16) % 16
        return bool(entry.capability_mask & (1 << bit_idx))

    def _get_fleet_centroid(self) -> np.ndarray | None:
        if not self._fleet_centroid_stale and self._fleet_centroid is not None:
            return self._fleet_centroid
        all_entries = self.all_entries_fleet_wide()
        if not all_entries:
            self._fleet_centroid = None
            self._fleet_centroid_stale = False
            return None
        vecs = np.stack([e.vector for e in all_entries])
        self._fleet_centroid = np.mean(vecs, axis=0)
        self._fleet_centroid_stale = False
        return self._fleet_centroid

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "node_id": self.node_id,
                "gen_tables": len(self._gen_tables),
                "skill_tables": len(self._skill_tables),
                "total_entries": sum(len(t) for t in self._gen_tables.values()),
            }
