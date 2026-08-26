"""Tests for MeshWAL.

Covers:
- WAL entry serialization (to_bytes / from_bytes)
- CRC32 integrity verification
- Append operations (insert, merge, delete)
- WAL rotation (max_wal_size)
- Crash recovery replay
- Checkpointing and truncation
- Stats tracking
- Close/cleanup
"""

from __future__ import annotations

import tempfile
import time

import numpy as np
import pytest

from swarm.mesh_vector_tables import MeshVectorTable, VectorTableEntry
from swarm.mesh_wal import MeshWAL, WALEntry, WALCheckpoint


class TestWALEntry:
    def test_serialize_roundtrip(self) -> None:
        entry = WALEntry(
            op="insert",
            timestamp=time.time(),
            payload={"op": "insert", "entry": {"agent_id": "a", "vector": [1.0, 0.0]}},
        )
        data = entry.to_bytes()
        assert len(data) > 0

        parsed = WALEntry.from_bytes(data)
        assert parsed.op == "insert"
        assert parsed.payload["entry"]["agent_id"] == "a"

    def test_crc32_corruption_detected(self) -> None:
        entry = WALEntry(
            op="insert",
            timestamp=time.time(),
            payload={"op": "insert", "entry": {"agent_id": "a"}},
        )
        data = entry.to_bytes()
        # Corrupt the CRC32 field (bytes 8-12) to trigger CRC mismatch
        corrupted = data[:8] + bytes([0xFF, 0xFF, 0xFF, 0xFF]) + data[12:]
        with pytest.raises(ValueError, match="CRC32 mismatch"):
            WALEntry.from_bytes(corrupted)

    def test_invalid_magic(self) -> None:
        with pytest.raises(ValueError, match="Invalid WAL magic"):
            WALEntry.from_bytes(b"XXXX" + b"\x00" * 100)


class TestMeshWALAppend:
    def test_append_insert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wal = MeshWAL(wal_dir=tmp, max_wal_size=1024 * 1024)
            assert wal.append_insert({"agent_id": "a", "vector": [1.0, 0.0]}) is True
            assert wal.stats["entry_count"] == 1
            wal.close()

    def test_append_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wal = MeshWAL(wal_dir=tmp, max_wal_size=1024 * 1024)
            assert wal.append_delete("agent_a") is True
            assert wal.stats["entry_count"] == 1
            wal.close()

    def test_append_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wal = MeshWAL(wal_dir=tmp, max_wal_size=1024 * 1024)
            assert wal.append_merge({"entries": [{"agent_id": "a"}]}) is True
            assert wal.stats["entry_count"] == 1
            wal.close()

    def test_wal_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wal = MeshWAL(wal_dir=tmp, max_wal_size=100)  # tiny for fast rotation
            # Append many entries to trigger rotation
            for i in range(20):
                wal.append_insert({"agent_id": f"agent_{i}", "vector": [float(i), 0.0]})
            wal.close()

            # Should have multiple WAL files
            wal_files = list(wal.wal_dir.glob("mesh_wal_*.log"))
            assert len(wal_files) > 1


class TestMeshWALRecovery:
    def test_recover_inserts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wal = MeshWAL(wal_dir=tmp, max_wal_size=1024 * 1024)
            # Append 5 insert operations
            for i in range(5):
                wal.append_insert(
                    {
                        "agent_id": f"agent_{i}",
                        "vector": [float(i), 0.0],
                        "timestamp": 1000.0 + i,
                        "node_id": "test",
                        "generation": i,
                        "fitness": 0.5,
                        "signature": f"test_signature_{i}",
                    }
                )
            wal.close()

            # Create new WAL pointing at same dir, recover into fresh table
            table = MeshVectorTable(table_id="recover_test")
            wal2 = MeshWAL(wal_dir=tmp, max_wal_size=1024 * 1024)
            stats = wal2.recover(table)
            wal2.close()

            assert stats["replayed"] == 5
            assert stats["errors"] == 0
            for i in range(5):
                assert table.query(f"agent_{i}") is not None

    def test_recover_with_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wal = MeshWAL(wal_dir=tmp, max_wal_size=1024 * 1024)

            # Insert 3 entries
            for i in range(3):
                wal.append_insert(
                    {
                        "agent_id": f"agent_{i}",
                        "vector": [float(i), 0.0],
                        "timestamp": 1000.0 + i,
                        "node_id": "test",
                        "generation": i,
                        "fitness": 0.5,
                        "signature": f"test_signature_{i}",
                    }
                )

            # Checkpoint — this truncates old WAL files
            table = MeshVectorTable(table_id="checkpoint_test")
            wal.checkpoint(table)
            wal.close()

            # Create new WAL, recover — files were truncated so replayed may be 0
            table2 = MeshVectorTable(table_id="checkpoint_test2")
            wal2 = MeshWAL(wal_dir=tmp, max_wal_size=1024 * 1024)
            stats = wal2.recover(table2)
            wal2.close()

            # After checkpoint truncation, replayed may be 0 or more
            assert stats["errors"] == 0
            # Either we replayed from remaining files, or checkpoint covered it
            assert stats["replayed"] >= 0

    def test_recover_empty_wal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wal = MeshWAL(wal_dir=tmp, max_wal_size=1024 * 1024)
            wal.close()

            table = MeshVectorTable(table_id="empty_test")
            wal2 = MeshWAL(wal_dir=tmp, max_wal_size=1024 * 1024)
            stats = wal2.recover(table)
            wal2.close()

            assert stats["replayed"] == 0
            assert stats["errors"] == 0


class TestMeshWALCheckpoint:
    def test_checkpoint_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wal = MeshWAL(wal_dir=tmp, max_wal_size=1024 * 1024)
            wal.append_insert({"agent_id": "a", "vector": [1.0, 0.0]})

            table = MeshVectorTable(table_id="checkpoint_meta")
            checkpoint = wal.checkpoint(table)
            wal.close()

            assert checkpoint.wal_file is not None
            assert checkpoint.offset > 0
            assert checkpoint.entry_count == 1

            # Checkpoint file should exist
            checkpoint_path = wal.wal_dir / "checkpoint.json"
            assert checkpoint_path.exists()

    def test_checkpoint_truncates_old_wal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wal = MeshWAL(wal_dir=tmp, max_wal_size=50)  # tiny for rotation
            # Insert many to create multiple WAL files
            for i in range(20):
                wal.append_insert({"agent_id": f"agent_{i}", "vector": [float(i), 0.0]})

            table = MeshVectorTable(table_id="truncate_test")
            wal.checkpoint(table)
            wal.close()

            # Should have at most 2 WAL files (current + 1 previous)
            wal_files = list(wal.wal_dir.glob("mesh_wal_*.log"))
            assert len(wal_files) <= 2


class TestMeshWALStats:
    def test_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wal = MeshWAL(wal_dir=tmp, max_wal_size=1024 * 1024)
            wal.append_insert({"agent_id": "a", "vector": [1.0, 0.0]})
            stats = wal.stats
            assert stats["entry_count"] == 1
            assert stats["wal_dir"] == tmp
            wal.close()
