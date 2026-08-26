"""QuantaVdbBridge — High-density streaming vector DB bridge for the fleet.

Integrates Quanta's PartitionedVdb (C++ / Hnswlib) into the sunset-ecosystem
via xlang's Python interop layer.  Provides:

- Three-tier partitioning (time × tag × RAM limit) for fleet-wide vector storage
- WAL async merging for crash-safe write paths
- TTL eviction for memory-safe long-running deployments
- Hnswlib approximate-nearest-neighbor search with 100% GIL bypass
- CRDT gossip compatibility with MeshVectorTable sync payloads

Architecture
------------
The bridge maintains two storage layers:

1. **Quanta PartitionedVdb** — native C++ vectors with time-series sharding,
   used for the actual high-dimensional embeddings and fast ANN search.

2. **SQLite CRDT Manifest** — a lightweight SQLite shadow that tracks
   per-entry metadata (agent_id, fitness, generation, capability_mask,
   thermal_pressure, signature) and supports deterministic CRDT merge.

This dual-layer design lets Quanta do what it does best (fast vectors,
Hnswlib graphs, memory management) while the sunset-ecosystem retains
its CRDT semantics, gossip protocol, and agent identity system.

Reference
---------
- Quanta architecture: https://github.com/CantorAI/Quanta
- xlang IPC: https://github.com/xlang-foundation/xlang/Docs/DISTRIBUTED.md
"""

from __future__ import annotations

__all__ = [
    "QuantaVdbBridge",
    "QuantaTableEntry",
    "VdbSyncPayload",
]

import base64
import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── QuantaTableEntry ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class QuantaTableEntry:
    """A single row in the fleet vector store, split into:

    - **vector** → stored in Quanta's native C++ Hnswlib graph
    - **metadata** → stored in the SQLite CRDT manifest
    """

    agent_id: str
    vector: np.ndarray
    timestamp: float
    node_id: str
    generation: int
    fitness: float
    signature: str
    capability_mask: int = 0xFFFF
    thermal_pressure: float = 0.0
    partition_tag: str = "default"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.vector, np.ndarray):
            object.__setattr__(self, "vector", np.array(self.vector, dtype=np.float32))
        else:
            object.__setattr__(
                self, "vector", self.vector.astype(np.float32, copy=False)
            )

    @property
    def dim(self) -> int:
        return int(self.vector.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "vector_b64": base64.b64encode(self.vector.tobytes()).decode("ascii"),
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "generation": self.generation,
            "fitness": self.fitness,
            "signature": self.signature,
            "capability_mask": self.capability_mask,
            "thermal_pressure": self.thermal_pressure,
            "partition_tag": self.partition_tag,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QuantaTableEntry":
        raw = base64.b64decode(d["vector_b64"])
        vec = np.frombuffer(raw, dtype=np.float32).copy()
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
            partition_tag=str(d.get("partition_tag", "default")),
            extra=dict(d.get("extra", {})),
        )


# ── VdbSyncPayload ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class VdbSyncPayload:
    """Wire-ready sync payload for gossip / mesh transport."""

    node_id: str
    timestamp: float
    entries: list[QuantaTableEntry]

    def to_bytes(self) -> bytes:
        serial = {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "entries": [e.to_dict() for e in self.entries],
        }
        return json.dumps(serial, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, blob: bytes) -> "VdbSyncPayload":
        serial = json.loads(blob.decode("utf-8"))
        entries = [QuantaTableEntry.from_dict(d) for d in serial.get("entries", [])]
        return cls(
            node_id=serial["node_id"],
            timestamp=serial["timestamp"],
            entries=entries,
        )


# ── QuantaVdbBridge ─────────────────────────────────────────────


class QuantaVdbBridge:
    """Fleet-wide vector database backed by Quanta PartitionedVdb.

    Parameters
    ----------
    prefix : str
        Namespace for this bridge instance (e.g. "fleet_breed").
    data_path : Path | str
        Directory where Quanta stores `.hnsw` / `.vdb` / `.db` files.
    dim : int
        Vector dimension (e.g. 256 for fleet embeddings).
    granularity : str
        Time sharding: "hourly", "daily", or "monthly".
    max_memory_gb : float
        Hard RAM ceiling per partition (e.g. 1.5 for edge nodes).
    ttl_minutes : int
        Idle partition eviction threshold.
    node_id : str
        Identifier for the local fleet node.
    """

    def __init__(
        self,
        prefix: str,
        data_path: Path | str,
        dim: int,
        granularity: str = "daily",
        max_memory_gb: float = 1.0,
        ttl_minutes: int = 3,
        node_id: str = "local",
    ) -> None:
        self.prefix = prefix
        self.data_path = Path(data_path)
        self.dim = dim
        self.granularity = granularity
        self.max_memory_gb = max_memory_gb
        self.ttl_minutes = ttl_minutes
        self.node_id = node_id

        self._lock = threading.RLock()
        self._quanta: Any | None = None  # lazy-loaded C++ module
        self._vdb: Any | None = None  # lazy-loaded PartitionedVdb handle

        # SQLite CRDT manifest
        self._manifest_path = self.data_path / f"{prefix}_crdt_manifest.db"
        self._ensure_manifest_schema()

        # In-memory hot cache of recent entries (fast path before Quanta)
        self._hot_cache: dict[str, QuantaTableEntry] = {}
        self._hot_cache_max = 1000

        # Stats
        self._insert_count = 0
        self._search_count = 0
        self._sync_count = 0

    # ── lazy Quanta loading ─────────────────────────────────────

    def _load_quanta(self) -> Any:
        """Lazy-import the xlang Quanta module (GIL-bypassed C++)."""
        if self._quanta is not None:
            return self._quanta
        try:
            import xlang  # type: ignore[import-untyped]

            self._quanta = xlang.importModule("quanta", fromPath="Quanta")
            logger.info("Quanta C++ module loaded via xlang interop")
        except Exception as exc:
            logger.warning("Failed to load Quanta C++ module: %s", exc)
            self._quanta = None
        return self._quanta

    def _get_vdb(self) -> Any | None:
        """Return (or create) the PartitionedVdb handle."""
        if self._vdb is not None:
            return self._vdb
        quanta = self._load_quanta()
        if quanta is None:
            return None
        try:
            self._vdb = quanta.partitioned_vdb(
                prefix=self.prefix,
                path=str(self.data_path),
                dim=self.dim,
                granularity=self.granularity,
                max_memory_gb=self.max_memory_gb,
                ttl_minutes=self.ttl_minutes,
            )
            logger.info(
                "PartitionedVdb created: prefix=%s dim=%d granularity=%s",
                self.prefix,
                self.dim,
                self.granularity,
            )
        except Exception as exc:
            logger.error("Failed to create PartitionedVdb: %s", exc)
            self._vdb = None
        return self._vdb

    # ── SQLite CRDT manifest ────────────────────────────────────

    def _ensure_manifest_schema(self) -> None:
        """Bootstrap the SQLite CRDT shadow table."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._manifest_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                agent_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                node_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                fitness REAL NOT NULL,
                signature TEXT NOT NULL,
                capability_mask INTEGER NOT NULL DEFAULT 65535,
                thermal_pressure REAL NOT NULL DEFAULT 0.0,
                partition_tag TEXT NOT NULL DEFAULT 'default',
                extra_json TEXT NOT NULL DEFAULT '{}',
                vector_b64 TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entries_node_gen
            ON entries(node_id, generation)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entries_fitness
            ON entries(fitness DESC)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_log (
                sync_id INTEGER PRIMARY KEY AUTOINCREMENT,
                remote_node_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                entries_merged INTEGER NOT NULL DEFAULT 0,
                entries_rejected INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.commit()
        conn.close()

    def _manifest_insert(self, entry: QuantaTableEntry) -> None:
        """Upsert into the SQLite CRDT manifest."""
        conn = sqlite3.connect(str(self._manifest_path))
        conn.execute(
            """
            INSERT INTO entries
            (agent_id, timestamp, node_id, generation, fitness,
             signature, capability_mask, thermal_pressure, partition_tag, extra_json, vector_b64)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                timestamp=excluded.timestamp,
                node_id=excluded.node_id,
                generation=excluded.generation,
                fitness=excluded.fitness,
                signature=excluded.signature,
                capability_mask=excluded.capability_mask,
                thermal_pressure=excluded.thermal_pressure,
                partition_tag=excluded.partition_tag,
                extra_json=excluded.extra_json,
                vector_b64=excluded.vector_b64
            WHERE excluded.timestamp > entries.timestamp
               OR (excluded.timestamp = entries.timestamp
                   AND excluded.signature < entries.signature)
            """,
            (
                entry.agent_id,
                entry.timestamp,
                entry.node_id,
                entry.generation,
                entry.fitness,
                entry.signature,
                entry.capability_mask,
                entry.thermal_pressure,
                entry.partition_tag,
                json.dumps(entry.extra, separators=(",", ":")),
                base64.b64encode(entry.vector.tobytes()).decode("ascii"),
            ),
        )
        conn.commit()
        conn.close()

    def _manifest_query(self, agent_id: str) -> QuantaTableEntry | None:
        """Fetch a single entry from the SQLite manifest."""
        conn = sqlite3.connect(str(self._manifest_path))
        row = conn.execute(
            "SELECT * FROM entries WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return self._row_to_entry(row)

    def _manifest_query_by_fitness(
        self, min_fitness: float = 0.0, max_results: int = 50
    ) -> list[QuantaTableEntry]:
        conn = sqlite3.connect(str(self._manifest_path))
        rows = conn.execute(
            """
            SELECT * FROM entries
            WHERE fitness >= ?
            ORDER BY fitness DESC
            LIMIT ?
            """,
            (min_fitness, max_results),
        ).fetchall()
        conn.close()
        return [self._row_to_entry(r) for r in rows]

    def _manifest_all_entries(self) -> list[QuantaTableEntry]:
        conn = sqlite3.connect(str(self._manifest_path))
        rows = conn.execute("SELECT * FROM entries").fetchall()
        conn.close()
        return [self._row_to_entry(r) for r in rows]

    def _manifest_stats(self) -> dict[str, Any]:
        conn = sqlite3.connect(str(self._manifest_path))
        count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        avg_fitness = (
            conn.execute("SELECT AVG(fitness) FROM entries").fetchone()[0] or 0.0
        )
        node_breakdown = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT node_id, COUNT(*) FROM entries GROUP BY node_id"
            ).fetchall()
        }
        conn.close()
        return {
            "count": count,
            "avg_fitness": avg_fitness,
            "node_breakdown": node_breakdown,
        }

    def _row_to_entry(self, row: tuple[Any, ...]) -> QuantaTableEntry:
        # row order matches schema
        raw = base64.b64decode(row[10])
        vec = np.frombuffer(raw, dtype=np.float32).copy()
        return QuantaTableEntry(
            agent_id=row[0],
            vector=vec,
            timestamp=row[1],
            node_id=row[2],
            generation=row[3],
            fitness=row[4],
            signature=row[5],
            capability_mask=row[6],
            thermal_pressure=row[7],
            partition_tag=row[8],
            extra=json.loads(row[9]),
        )

    # ── insertion ───────────────────────────────────────────────

    def insert(self, entry: QuantaTableEntry) -> bool:
        """Add an entry to both Quanta VDB and SQLite manifest.

        Returns True if accepted (CRDT winner or new entry).
        """
        with self._lock:
            # 1. Hot cache fast path
            self._hot_cache[entry.agent_id] = entry
            if len(self._hot_cache) > self._hot_cache_max:
                # Evict oldest by timestamp
                oldest = min(
                    self._hot_cache, key=lambda k: self._hot_cache[k].timestamp
                )
                del self._hot_cache[oldest]

            # 2. SQLite CRDT manifest (deterministic merge)
            existing = self._manifest_query(entry.agent_id)
            if existing is not None:
                winner = self._crdt_winner(existing, entry)
                if winner is existing:
                    return False
            self._manifest_insert(entry)

            # 3. Quanta native VDB (if available)
            vdb = self._get_vdb()
            if vdb is not None:
                try:
                    now_ms = int(entry.timestamp * 1000)
                    vdb.AddVectors(
                        int(hashlib.sha256(entry.agent_id.encode()).hexdigest(), 16)
                        % (2**63),
                        entry.vector,
                        timestamp=now_ms,
                        partition=entry.partition_tag,
                        chunks=json.dumps(entry.extra, separators=(",", ":")),
                    )
                except Exception as exc:
                    logger.debug("Quanta AddVectors failed: %s", exc)

            self._insert_count += 1
            return True

    # ── search ──────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray | list[float],
        k: int = 10,
        partition: str | None = None,
        ts_start: float | None = None,
        ts_end: float | None = None,
    ) -> list[dict[str, Any]]:
        """Approximate nearest-neighbor search via Quanta / Hnswlib.

        Falls back to brute-force search if Quanta C++ module is unavailable.
        """
        vec = np.array(query_vector, dtype=np.float32)
        with self._lock:
            vdb = self._get_vdb()
            if vdb is not None:
                try:
                    now_ms = int(time.time() * 1000)
                    start_ms = int(ts_start * 1000) if ts_start else now_ms - 86400000
                    end_ms = int(ts_end * 1000) if ts_end else now_ms
                    results = vdb.Lookup(
                        vec,
                        k,
                        partition=partition or "default",
                        ts_start=start_ms,
                        ts_end=end_ms,
                    )
                    self._search_count += 1
                    return [
                        {
                            "id": r[0],
                            "score": r[1],
                            "metadata": json.loads(r[2]) if len(r) > 2 else {},
                            "partition": r[3] if len(r) > 3 else partition,
                        }
                        for r in results
                    ]
                except Exception as exc:
                    logger.warning("Quanta Lookup failed: %s", exc)

            # Fallback: brute-force search over SQLite manifest
            return self._brute_force_search(vec, k, partition, ts_start, ts_end)

    def _brute_force_search(
        self,
        query: np.ndarray,
        k: int,
        partition: str | None,
        ts_start: float | None,
        ts_end: float | None,
    ) -> list[dict[str, Any]]:
        """Brute-force fallback when Quanta C++ is unavailable."""
        entries = self._manifest_all_entries()
        scored: list[tuple[float, QuantaTableEntry]] = []
        for e in entries:
            if partition is not None and e.partition_tag != partition:
                continue
            if ts_start is not None and e.timestamp < ts_start:
                continue
            if ts_end is not None and e.timestamp > ts_end:
                continue
            dist = float(np.linalg.norm(query - e.vector))
            scored.append((dist, e))
        scored.sort(key=lambda x: x[0])
        return [
            {
                "id": e.agent_id,
                "score": 1.0 / (1.0 + dist),  # convert distance to similarity
                "metadata": e.extra,
                "partition": e.partition_tag,
            }
            for dist, e in scored[:k]
        ]

    # ── sync / gossip ───────────────────────────────────────────

    def get_sync_payload(self) -> bytes:
        """Return a compressed sync payload for MeshVectorGossip."""
        with self._lock:
            entries = self._manifest_all_entries()
            payload = VdbSyncPayload(
                node_id=self.node_id,
                timestamp=time.time(),
                entries=entries,
            )
            return payload.to_bytes()

    def apply_sync_payload(self, payload: bytes) -> dict[str, Any]:
        """Decode and merge a remote sync payload."""
        with self._lock:
            try:
                sync = VdbSyncPayload.from_bytes(payload)
            except Exception as exc:
                return {"merged": 0, "rejected": 0, "errors": [str(exc)]}

            merged = 0
            rejected = 0
            for entry in sync.entries:
                ok = self.insert(entry)
                if ok:
                    merged += 1
                else:
                    rejected += 1

            self._sync_count += 1
            # Log sync
            conn = sqlite3.connect(str(self._manifest_path))
            conn.execute(
                "INSERT INTO sync_log (remote_node_id, timestamp, entries_merged, entries_rejected) VALUES (?, ?, ?, ?)",
                (sync.node_id, sync.timestamp, merged, rejected),
            )
            conn.commit()
            conn.close()

            return {"merged": merged, "rejected": rejected, "remote_node": sync.node_id}

    # ── fleet queries ────────────────────────────────────────────

    def get_breedable_pool(
        self,
        min_fitness: float = 0.7,
        max_thermal: float = 0.5,
        max_results: int = 20,
    ) -> list[QuantaTableEntry]:
        """Return cross-node parent candidates."""
        entries = self._manifest_query_by_fitness(min_fitness, max_results * 2)
        return [e for e in entries if e.thermal_pressure <= max_thermal][:max_results]

    def get_population_summary(self) -> dict[str, Any]:
        """Return fleet-wide population snapshot."""
        stats = self._manifest_stats()
        entries = self._manifest_all_entries()
        if not entries:
            return {**stats, "diversity_score": 0.0, "generation_range": None}

        vectors = np.stack([e.vector for e in entries])
        centroid = np.mean(vectors, axis=0)
        distances = np.linalg.norm(vectors - centroid, axis=1)
        avg_dist = float(np.mean(distances))
        dim = vectors.shape[1]
        diversity = min(1.0, avg_dist / (0.5 * np.sqrt(dim)))

        gen_min = min(e.generation for e in entries)
        gen_max = max(e.generation for e in entries)

        return {
            **stats,
            "diversity_score": diversity,
            "generation_range": (gen_min, gen_max),
        }

    # ── internal helpers ──────────────────────────────────────────

    def _crdt_winner(
        self, a: QuantaTableEntry, b: QuantaTableEntry
    ) -> QuantaTableEntry:
        """Higher timestamp wins; equal timestamp → lower signature hash wins."""
        if b.timestamp > a.timestamp:
            return b
        if a.timestamp > b.timestamp:
            return a
        # Tiebreak: lower SHA-256 hash wins
        b_hash = hashlib.sha256(b.signature.encode()).hexdigest()
        a_hash = hashlib.sha256(a.signature.encode()).hexdigest()
        return b if b_hash < a_hash else a

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "prefix": self.prefix,
                "node_id": self.node_id,
                "insert_count": self._insert_count,
                "search_count": self._search_count,
                "sync_count": self._sync_count,
                "hot_cache_size": len(self._hot_cache),
                **self._manifest_stats(),
            }
