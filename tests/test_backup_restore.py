"""Tests for backup_restore.py — Fleet state snapshot and restore.

Run: python3 -m pytest tests/test_backup_restore.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.backup_restore import BackupRestore, Snapshot


class TestBackupRestore:
    def test_create(self):
        br = BackupRestore(compress=True)
        assert br._compress is True

    def test_snapshot_restore(self):
        br = BackupRestore()
        state = {"agents": ["a", "b"], "config": {"version": 1}}
        snap = br.snapshot(state)
        assert snap.version == "1.0"
        assert snap.compressed is True
        assert snap.checksum != ""

        restored = br.restore(snap)
        assert restored == state

    def test_snapshot_no_compress(self):
        br = BackupRestore(compress=False)
        state = {"key": "value"}
        snap = br.snapshot(state)
        assert snap.compressed is False
        restored = br.restore(snap)
        assert restored == state

    def test_validate(self):
        br = BackupRestore()
        state = {"key": "value"}
        snap = br.snapshot(state)
        assert br.validate(snap) is True

    def test_validate_corrupted(self):
        br = BackupRestore()
        state = {"key": "value"}
        snap = br.snapshot(state)
        # Corrupt data
        snap = Snapshot(
            version=snap.version,
            timestamp=snap.timestamp,
            checksum=snap.checksum,
            compressed=snap.compressed,
            data=b"corrupted data",
            metadata=snap.metadata,
        )
        assert br.validate(snap) is False

    def test_restore_corrupted(self):
        br = BackupRestore()
        state = {"key": "value"}
        snap = br.snapshot(state)
        snap = Snapshot(
            version=snap.version,
            timestamp=snap.timestamp,
            checksum=snap.checksum,
            compressed=snap.compressed,
            data=b"corrupted data",
            metadata=snap.metadata,
        )
        with pytest.raises(ValueError):
            br.restore(snap)

    def test_snapshot_metadata(self):
        br = BackupRestore()
        state = {"key": "value"}
        snap = br.snapshot(state, metadata={"source": "test", "id": 42})
        assert snap.metadata["source"] == "test"
        assert snap.metadata["id"] == 42

    def test_size_bytes(self):
        br = BackupRestore()
        state = {"key": "value" * 1000}
        snap = br.snapshot(state)
        assert br.size_bytes(snap) > 0

    def test_info(self):
        br = BackupRestore()
        state = {"key": "value"}
        snap = br.snapshot(state, metadata={"m": 1})
        info = br.info(snap)
        assert info["version"] == "1.0"
        assert info["compressed"] is True
        assert "size_bytes" in info
        assert info["metadata"] == {"m": 1}

    def test_delta(self):
        br = BackupRestore()
        state1 = {"a": 1, "b": 2}
        br.snapshot(state1)
        state2 = {"a": 1, "b": 3, "c": 4}
        delta = br.delta(state2)
        assert delta is not None
        assert delta["b"] == 3
        assert delta["c"] == 4

    def test_delta_no_change(self):
        br = BackupRestore()
        state = {"a": 1}
        br.snapshot(state)
        assert br.delta(state) is None

    def test_delta_first_snapshot(self):
        br = BackupRestore()
        assert br.delta({"a": 1}) is None

    def test_repr(self):
        br = BackupRestore()
        assert "BackupRestore" in repr(br)
