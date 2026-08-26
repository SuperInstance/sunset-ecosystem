"""MeshTableStore — SQLite persistence for MeshVectorTables.

Survives process restarts. One SQLite file per fleet node.

Usage::
    from swarm.mesh_table_store import MeshTableStore

    store = MeshTableStore("/tmp/fleet_mesh.db")
    store.save_table(table)          # persist a MeshVectorTable
    table2 = store.load_table("pool_a")  # restore
"""

from __future__ import annotations

__all__ = ["MeshTableStore"]

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from swarm.mesh_vector_tables import VectorTableEntry, MeshVectorTable, FleetVectorIndex


class MeshTableStore:
    """SQLite-backed persistence for mesh vector tables."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mesh_tables (
                    table_id TEXT PRIMARY KEY,
                    created_at REAL DEFAULT (julianday('now')),
                    updated_at REAL DEFAULT (julianday('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mesh_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    vector_b64 TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    node_id TEXT NOT NULL,
                    generation INTEGER NOT NULL DEFAULT 0,
                    fitness REAL NOT NULL DEFAULT 0.0,
                    signature TEXT NOT NULL,
                    capability_mask INTEGER NOT NULL DEFAULT 65535,
                    thermal_pressure REAL NOT NULL DEFAULT 0.0,
                    extra_json TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(table_id, agent_id, timestamp)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entries_table
                ON mesh_entries(table_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entries_fitness
                ON mesh_entries(table_id, fitness)
            """)
            conn.commit()

    # ── save ────────────────────────────────────────────────────

    def save_table(self, table: MeshVectorTable) -> int:
        """Persist all entries from a MeshVectorTable. Returns count."""
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            # Upsert table metadata
            conn.execute(
                """INSERT INTO mesh_tables(table_id) VALUES (?)
                   ON CONFLICT(table_id) DO UPDATE SET updated_at=julianday('now')""",
                (table.table_id,),
            )
            # Clear old entries for this table
            conn.execute(
                "DELETE FROM mesh_entries WHERE table_id = ?", (table.table_id,)
            )
            # Insert current entries
            entries = table.all_entries()
            for e in entries:
                conn.execute(
                    """INSERT INTO mesh_entries
                       (table_id, agent_id, vector_b64, timestamp, node_id,
                        generation, fitness, signature, capability_mask,
                        thermal_pressure, extra_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        table.table_id,
                        e.agent_id,
                        _vec_to_b64(e.vector),
                        e.timestamp,
                        e.node_id,
                        e.generation,
                        e.fitness,
                        e.signature,
                        e.capability_mask,
                        e.thermal_pressure,
                        json.dumps(e.extra),
                    ),
                )
            conn.commit()
            return len(entries)

    def save_index(self, index: FleetVectorIndex) -> dict[str, int]:
        """Persist all tables in a FleetVectorIndex. Returns {table_id: count}."""
        counts = {}
        with index._lock:
            # Save generation tables
            for gen, table in index._gen_tables.items():
                counts[table.table_id] = self.save_table(table)
            # Save skill tables
            for skill, table in index._skill_tables.items():
                counts[table.table_id] = self.save_table(table)
        return counts

    # ── load ────────────────────────────────────────────────────

    def load_table(self, table_id: str) -> MeshVectorTable:
        """Restore a MeshVectorTable from SQLite."""
        table = MeshVectorTable(table_id=table_id)
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                """SELECT agent_id, vector_b64, timestamp, node_id, generation,
                          fitness, signature, capability_mask, thermal_pressure, extra_json
                   FROM mesh_entries WHERE table_id = ? ORDER BY timestamp""",
                (table_id,),
            )
            for row in cursor:
                extra = json.loads(row[9]) if row[9] else {}
                entry = VectorTableEntry(
                    agent_id=row[0],
                    vector=_b64_to_vec(row[1]),
                    timestamp=row[2],
                    node_id=row[3],
                    generation=row[4],
                    fitness=row[5],
                    signature=row[6],
                    capability_mask=row[7],
                    thermal_pressure=row[8],
                    extra=extra,
                )
                table._entries[entry.agent_id] = entry
        return table

    def load_index(
        self, node_id: str = "restored", prefix: str = ""
    ) -> FleetVectorIndex:
        """Restore a FleetVectorIndex with all persisted tables."""
        index = FleetVectorIndex(node_id=node_id)
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "SELECT DISTINCT table_id FROM mesh_entries WHERE table_id LIKE ?",
                (f"{prefix}%",) if prefix else ("%",),
            )
            table_ids = [r[0] for r in cursor.fetchall()]
        for tid in table_ids:
            table = self.load_table(tid)
            # Determine if it's a gen table or skill table by prefix
            if tid.startswith("gen_"):
                try:
                    gen = int(tid.split("_", 1)[1])
                    index._gen_tables[gen] = table
                except ValueError:
                    # Non-numeric gen suffix — store as fallback
                    next_key = max(index._gen_tables.keys(), default=-1) + 1
                    index._gen_tables[next_key] = table
            elif tid.startswith("skill_"):
                skill = tid.split("_", 1)[1]
                index._skill_tables[skill] = table
            else:
                # Unknown prefix, store in gen_tables as fallback
                next_key = max(index._gen_tables.keys(), default=-1) + 1
                index._gen_tables[next_key] = table
        return index

    # ── query ───────────────────────────────────────────────────

    def query_by_fitness(
        self, table_id: str, min_fitness: float = 0.0, limit: int = 1000
    ):
        """Query entries by fitness threshold."""
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                """SELECT agent_id, vector_b64, timestamp, node_id, generation,
                          fitness, signature, capability_mask, thermal_pressure, extra_json
                   FROM mesh_entries
                   WHERE table_id = ? AND fitness >= ?
                   ORDER BY fitness DESC
                   LIMIT ?""",
                (table_id, min_fitness, limit),
            )
            results = []
            for row in cursor:
                extra = json.loads(row[9]) if row[9] else {}
                results.append(
                    VectorTableEntry(
                        agent_id=row[0],
                        vector=_b64_to_vec(row[1]),
                        timestamp=row[2],
                        node_id=row[3],
                        generation=row[4],
                        fitness=row[5],
                        signature=row[6],
                        capability_mask=row[7],
                        thermal_pressure=row[8],
                        extra=extra,
                    )
                )
            return results

    def count_entries(self, table_id: str | None = None) -> int:
        """Count total entries, optionally filtered by table."""
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            if table_id:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM mesh_entries WHERE table_id = ?",
                    (table_id,),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM mesh_entries")
            return cursor.fetchone()[0]

    # ── delete ──────────────────────────────────────────────────

    def drop_table(self, table_id: str) -> int:
        """Remove all entries for a table. Returns deleted count."""
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "DELETE FROM mesh_entries WHERE table_id = ?", (table_id,)
            )
            conn.execute("DELETE FROM mesh_tables WHERE table_id = ?", (table_id,))
            conn.commit()
            return cursor.rowcount

    def delete_older_than(self, age_seconds: float) -> int:
        """Delete entries older than age_seconds. Returns deleted count."""
        cutoff = time.time() - age_seconds
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "DELETE FROM mesh_entries WHERE timestamp < ?", (cutoff,)
            )
            conn.commit()
            return cursor.rowcount


# ── helpers ─────────────────────────────────────────────────────


def _vec_to_b64(vec: np.ndarray) -> str:
    """Float32 vector → base64 string."""
    return (
        __import__("base64").b64encode(vec.astype(np.float32).tobytes()).decode("ascii")
    )


def _b64_to_vec(b64: str) -> np.ndarray:
    """Base64 string → float32 vector."""
    raw = __import__("base64").b64decode(b64)
    return np.frombuffer(raw, dtype=np.float32).copy()
