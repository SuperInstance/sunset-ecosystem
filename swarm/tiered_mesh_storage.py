"""TieredMeshStorage — Hot / Warm / Cold tiers for MeshVectorTables.

Implements a three-tier storage system:
- **Hot**: In-memory with HNSW ANN index (recent + high-fitness entries)
- **Warm**: SQLite-backed with B-tree index (older, still accessible)
- **Cold**: Compressed file archives (rarely accessed, bulk retrieval)

Promotion/demotion based on:
- Recency (timestamp)
- Fitness (higher = more likely to stay hot)
- Access frequency (tracked per entry)
- Thermal pressure (hot agents get demoted to cool down)

Reference: docs/QUANTA_VDB_DEEP_DIVE.md — Phase 2: Tiered Storage
"""

from __future__ import annotations

__all__ = ["TieredMeshStorage", "TierConfig", "PromotionPolicy"]

import json
import logging
import os
import sqlite3
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TierConfig:
    """Configuration for tiered storage thresholds."""
    hot_max_entries: int = 1000
    hot_min_fitness: float = 0.5
    hot_max_age_seconds: float = 86400.0  # 24h
    warm_max_entries: int = 10000
    warm_max_age_seconds: float = 604800.0  # 7d
    cold_archive_interval: int = 1000  # Archive every N cold entries
    promotion_access_threshold: int = 3  # Access count to promote warm→hot
    demotion_thermal_threshold: float = 0.8  # Thermal pressure to demote


@dataclass
class PromotionPolicy:
    """Rules for tier transitions."""
    promote_on_access: bool = True
    promote_on_fitness_spike: bool = True
    demote_on_thermal: bool = True
    demote_on_age: bool = True
    emergency_hot_capacity: int = 2000  # Absolute max hot entries


class TieredMeshStorage:
    """Tiered storage wrapper for MeshVectorTable.

    Parameters
    ----------
    base_table : MeshVectorTable
        The underlying mesh table (becomes the "hot" tier).
    db_path : Path | str
        Path to SQLite database for warm tier.
    cold_path : Path | str
        Directory for cold archive files.
    config : TierConfig
        Tier thresholds.
    policy : PromotionPolicy
        Promotion/demotion rules.
    """

    def __init__(
        self,
        base_table: Any,  # MeshVectorTable
        db_path: Path | str = "tiered_warm.db",
        cold_path: Path | str = "tiered_cold",
        config: TierConfig | None = None,
        policy: PromotionPolicy | None = None,
    ) -> None:
        self.base = base_table
        self.config = config or TierConfig()
        self.policy = policy or PromotionPolicy()

        self._db_path = Path(db_path)
        self._cold_path = Path(cold_path)
        self._cold_path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._access_counts: dict[str, int] = {}  # agent_id -> access count
        self._last_access: dict[str, float] = {}   # agent_id -> last access time
        self._warm_count: int = 0
        self._cold_count: int = 0

        # SQLite warm tier
        self._init_sqlite()

        # Maintenance thread
        self._stop_maintenance = False
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop, daemon=True,
        )
        self._maintenance_thread.start()

    # ── SQLite warm tier ──────────────────────────────────────

    def _init_sqlite(self) -> None:
        """Initialize SQLite schema for warm tier."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS warm_entries (
                agent_id TEXT PRIMARY KEY,
                vector_b64 TEXT NOT NULL,
                timestamp REAL NOT NULL,
                node_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                fitness REAL NOT NULL,
                signature TEXT NOT NULL,
                capability_mask INTEGER DEFAULT 65535,
                thermal_pressure REAL DEFAULT 0.0,
                extra_json TEXT,
                tier TEXT DEFAULT 'warm',
                inserted_at REAL DEFAULT (julianday('now') * 86400.0 - 2440588.0 * 86400.0)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_warm_fitness ON warm_entries(fitness DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_warm_timestamp ON warm_entries(timestamp DESC)
        """)
        conn.commit()
        conn.close()

    def _warm_insert(self, entry: Any) -> None:
        """Insert entry into warm SQLite tier."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                """INSERT OR REPLACE INTO warm_entries
                (agent_id, vector_b64, timestamp, node_id, generation, fitness,
                 signature, capability_mask, thermal_pressure, extra_json, tier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'warm')""",
                (
                    entry.agent_id,
                    self._vec_to_b64(entry.vector),
                    entry.timestamp,
                    entry.node_id,
                    entry.generation,
                    entry.fitness,
                    entry.signature,
                    entry.capability_mask,
                    entry.thermal_pressure,
                    json.dumps(entry.extra, separators=(",", ":")),
                ),
            )
            conn.commit()
            self._warm_count += 1
        finally:
            conn.close()

    def _warm_query(self, agent_id: str) -> Any | None:  # VectorTableEntry | None
        """Query warm tier by agent_id."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            cursor = conn.execute(
                "SELECT * FROM warm_entries WHERE agent_id = ?", (agent_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_entry(row)
        finally:
            conn.close()

    def _warm_query_by_fitness(
        self,
        min_fitness: float = 0.0,
        max_results: int = 50,
    ) -> list[Any]:
        """Query warm tier by fitness threshold."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            cursor = conn.execute(
                """SELECT * FROM warm_entries
                WHERE fitness >= ?
                ORDER BY fitness DESC
                LIMIT ?""",
                (min_fitness, max_results),
            )
            rows = cursor.fetchall()
            return [self._row_to_entry(row) for row in rows]
        finally:
            conn.close()

    def _warm_delete(self, agent_id: str) -> None:
        """Delete from warm tier."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("DELETE FROM warm_entries WHERE agent_id = ?", (agent_id,))
            conn.commit()
            self._warm_count -= 1
        finally:
            conn.close()

    def _warm_count_entries(self) -> int:
        """Count entries in warm tier."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM warm_entries")
            return cursor.fetchone()[0]
        finally:
            conn.close()

    # ── cold tier (compressed files) ───────────────────────────

    def _cold_archive(self, entries: list[Any]) -> str:
        """Archive entries to a compressed file. Returns filename."""
        timestamp = int(time.time())
        filename = f"cold_archive_{timestamp}.jsonl.zlib"
        filepath = self._cold_path / filename

        data = []
        for entry in entries:
            data.append(entry.to_dict(include_vector=True))

        blob = json.dumps(data, separators=(",", ":")).encode("utf-8")
        compressed = zlib.compress(blob, level=6)
        filepath.write_bytes(compressed)
        self._cold_count += len(entries)
        return str(filepath)

    def _cold_query(self, agent_id: str) -> Any | None:
        """Query cold archives for agent_id. Slow — scans all archives."""
        for archive_file in sorted(self._cold_path.glob("cold_archive_*.jsonl.zlib")):
            compressed = archive_file.read_bytes()
            try:
                blob = zlib.decompress(compressed)
                data = json.loads(blob.decode("utf-8"))
                for entry_dict in data:
                    if entry_dict.get("agent_id") == agent_id:
                        from swarm.mesh_vector_tables import VectorTableEntry
                        return VectorTableEntry.from_dict(entry_dict)
            except Exception as exc:
                logger.warning("Failed to read cold archive %s: %s", archive_file, exc)
        return None

    # ── public API ────────────────────────────────────────────

    def query(self, agent_id: str) -> Any | None:
        """Query across all tiers: hot -> warm -> cold."""
        # Track access
        with self._lock:
            self._access_counts[agent_id] = self._access_counts.get(agent_id, 0) + 1
            self._last_access[agent_id] = time.time()

        # Hot tier
        entry = self.base.query(agent_id)
        if entry is not None:
            return entry

        # Warm tier
        entry = self._warm_query(agent_id)
        if entry is not None:
            # Consider promotion
            self._maybe_promote(entry)
            return entry

        # Cold tier
        return self._cold_query(agent_id)

    def insert(self, entry: Any) -> bool:
        """Insert into appropriate tier based on fitness/age/thermal."""
        now = time.time()
        age = now - entry.timestamp
        hot = self._should_be_hot(entry, age)

        if hot:
            # Check hot capacity
            hot_count = len(self.base)
            if hot_count >= self.policy.emergency_hot_capacity:
                # Demote oldest/lowest fitness hot entry
                self._demote_oldest_hot()
            return self.base.insert(entry)
        else:
            # Insert to warm tier
            self._warm_insert(entry)
            return True

    def query_by_fitness(
        self,
        min_fitness: float = 0.0,
        max_results: int = 50,
        include_warm: bool = True,
    ) -> list[Any]:
        """Query across hot and warm tiers by fitness."""
        hot_results = self.base.query_by_fitness(min_fitness, max_results)
        if not include_warm:
            return hot_results

        warm_results = self._warm_query_by_fitness(min_fitness, max_results)
        # Merge and deduplicate (hot wins)
        seen = {e.agent_id for e in hot_results}
        merged = list(hot_results)
        for e in warm_results:
            if e.agent_id not in seen:
                merged.append(e)
        merged.sort(key=lambda e: e.fitness, reverse=True)
        return merged[:max_results]

    def get_tier_stats(self) -> dict[str, Any]:
        """Return statistics for each tier."""
        return {
            "hot": {
                "entry_count": len(self.base),
                "max_entries": self.config.hot_max_entries,
            },
            "warm": {
                "entry_count": self._warm_count_entries(),
                "db_path": str(self._db_path),
            },
            "cold": {
                "entry_count": self._cold_count,
                "archive_count": len(list(self._cold_path.glob("cold_archive_*.jsonl.zlib"))),
                "cold_path": str(self._cold_path),
            },
        }

    def close(self) -> None:
        """Stop maintenance thread."""
        self._stop_maintenance = True
        self._maintenance_thread.join(timeout=5.0)

    # ── promotion / demotion ──────────────────────────────────

    def _should_be_hot(self, entry: Any, age: float) -> bool:
        """Determine if entry should be in hot tier."""
        if entry.fitness >= self.config.hot_min_fitness and age <= self.config.hot_max_age_seconds:
            return True
        if entry.thermal_pressure >= self.config.demotion_thermal_threshold:
            return False
        return False

    def _maybe_promote(self, entry: Any) -> None:
        """Promote warm entry to hot if access threshold met."""
        if not self.policy.promote_on_access:
            return
        access_count = self._access_counts.get(entry.agent_id, 0)
        if access_count >= self.config.promotion_access_threshold:
            # Check hot capacity
            if len(self.base) < self.config.hot_max_entries:
                self._warm_delete(entry.agent_id)
                self.base.insert(entry, skip_verify=True)
                logger.info("Promoted %s to hot tier (access count: %d)", entry.agent_id, access_count)

    def _demote_oldest_hot(self) -> None:
        """Demote oldest/lowest-fitness hot entry to warm."""
        entries = self.base.all_entries()
        if not entries:
            return
        # Find oldest or lowest fitness
        victim = min(entries, key=lambda e: (e.fitness, e.timestamp))
        self.base._entries.pop(victim.agent_id, None)  # type: ignore
        self._warm_insert(victim)
        logger.info("Demoted %s to warm tier (fitness: %.3f)", victim.agent_id, victim.fitness)

    # ── maintenance loop ──────────────────────────────────────

    def _maintenance_loop(self) -> None:
        """Background maintenance: demote old hot entries, archive warm to cold."""
        while not self._stop_maintenance:
            time.sleep(60.0)  # Run every 60 seconds
            try:
                self._run_maintenance()
            except Exception as exc:
                logger.warning("Maintenance error: %s", exc)

    def _run_maintenance(self) -> None:
        """Single maintenance pass."""
        now = time.time()

        # Demote hot entries that are too old or have high thermal pressure
        hot_entries = self.base.all_entries()
        to_demote: list[Any] = []
        for entry in hot_entries:
            age = now - entry.timestamp
            if age > self.config.hot_max_age_seconds:
                to_demote.append(entry)
            elif self.policy.demote_on_thermal and entry.thermal_pressure >= self.config.demotion_thermal_threshold:
                to_demote.append(entry)

        for entry in to_demote:
            self.base._entries.pop(entry.agent_id, None)  # type: ignore
            self._warm_insert(entry)
            logger.info("Demoted %s to warm (age: %.0fs, thermal: %.2f)", entry.agent_id, age, entry.thermal_pressure)

        # Archive old warm entries to cold
        conn = sqlite3.connect(str(self._db_path))
        try:
            cutoff = now - self.config.warm_max_age_seconds
            cursor = conn.execute(
                "SELECT * FROM warm_entries WHERE timestamp < ?", (cutoff,)
            )
            rows = cursor.fetchall()
            if len(rows) >= self.config.cold_archive_interval:
                entries = [self._row_to_entry(row) for row in rows]
                self._cold_archive(entries)
                for row in rows:
                    conn.execute("DELETE FROM warm_entries WHERE agent_id = ?", (row[0],))
                conn.commit()
                logger.info("Archived %d warm entries to cold tier", len(rows))
        finally:
            conn.close()

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _vec_to_b64(vec: np.ndarray) -> str:
        import base64
        return base64.b64encode(vec.astype(np.float32).tobytes()).decode("ascii")

    @staticmethod
    def _b64_to_vec(b64: str, dim: int) -> np.ndarray:
        import base64
        raw = base64.b64decode(b64)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def _row_to_entry(self, row: tuple[Any, ...]) -> Any:
        """Convert SQLite row to VectorTableEntry."""
        from swarm.mesh_vector_tables import VectorTableEntry
        # row: (agent_id, vector_b64, timestamp, node_id, generation, fitness,
        #       signature, capability_mask, thermal_pressure, extra_json, tier, ...)
        extra = json.loads(row[9]) if row[9] else {}
        # Infer dim from b64 length
        raw = self._b64_to_vec(row[1], 0)
        dim = raw.shape[0]
        return VectorTableEntry(
            agent_id=row[0],
            vector=raw,
            timestamp=row[2],
            node_id=row[3],
            generation=row[4],
            fitness=row[5],
            signature=row[6],
            capability_mask=row[7],
            thermal_pressure=row[8],
            extra=extra,
        )
