"""Tests for swarm.wal — FleetWAL append-only log."""

import tempfile
from pathlib import Path

import pytest

from swarm.wal import FleetWAL, WALEntry, WALCheckpoint


class TestFleetWAL:
    def test_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            wal = FleetWAL(tmp)
            assert wal.total_entries == 0
            assert wal.segment_count == 0

    def test_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            wal = FleetWAL(tmp)
            entry = wal.append("knowledge", room="forge", doc_id="abc")
            assert isinstance(entry, WALEntry)
            assert entry.layer == "knowledge"
            assert entry.sequence == 1
            assert wal.total_entries == 1

    def test_multiple_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            wal = FleetWAL(tmp)
            for i in range(5):
                wal.append("agent", agent_id=i, fitness=0.9)
            assert wal.total_entries == 5

    def test_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            wal = FleetWAL(tmp)
            received = []
            wal.register_layer("knowledge", lambda e: received.append(e))
            wal.append("knowledge", room="forge", doc_id="abc")
            wal.append("agent", agent_id=1)
            wal.append("knowledge", room="forge", doc_id="def")
            count = wal.replay()
            assert count == 3
            assert len(received) == 2
            assert all(e.layer == "knowledge" for e in received)

    def test_replay_since(self):
        with tempfile.TemporaryDirectory() as tmp:
            wal = FleetWAL(tmp)
            received = []
            wal.register_layer("knowledge", lambda e: received.append(e))
            wal.append("knowledge", room="forge", doc_id="abc")
            wal.append("knowledge", room="forge", doc_id="def")
            count = wal.replay(since_sequence=1)
            assert count == 1
            assert len(received) == 1

    def test_segment_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            wal = FleetWAL(tmp, segment_size=256)
            for i in range(10):
                wal.append("test", data="x" * 100)
            assert wal.segment_count >= 2

    def test_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            wal = FleetWAL(tmp, checkpoint_interval=2)
            wal.register_layer("knowledge", MockLayer())
            wal.append("knowledge", room="forge", doc_id="abc")
            wal.append("knowledge", room="forge", doc_id="def")
            # Checkpoint should trigger automatically
            assert wal.total_entries == 2
            chk = wal.load_checkpoint()
            assert chk is not None
            assert chk.sequence == 2

    def test_load_checkpoint_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            wal = FleetWAL(tmp, checkpoint_interval=1)
            wal.register_layer("knowledge", MockLayer())
            wal.append("knowledge", room="forge", doc_id="abc")
            wal.append("knowledge", room="forge", doc_id="def")
            chk = wal.load_checkpoint(sequence=1)
            assert chk is not None
            assert chk.sequence == 1

    def test_checksum_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            wal = FleetWAL(tmp)
            entry = wal.append("test", data="hello")
            # Verify checksum is non-empty
            assert entry.checksum
            # Re-read and verify checksum matches
            entries = list(wal._read_segment(wal._current_segment))
            assert len(entries) == 1
            assert entries[0].checksum == entry.checksum

    def test_repr(self):
        with tempfile.TemporaryDirectory() as tmp:
            wal = FleetWAL(tmp)
            assert "FleetWAL" in repr(wal)


class MockLayer:
    """Mock layer for checkpoint testing."""

    def __init__(self):
        self.docs = {}

    def __call__(self, entry: WALEntry):
        self.docs[entry.payload.get("doc_id", "")] = entry.payload

    def checkpoint(self):
        return {"docs": self.docs}
