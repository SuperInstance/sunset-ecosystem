"""Tests for snapshot_manager.py — State snapshot and restore.

Run: python3 -m pytest tests/test_snapshot_manager.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.snapshot_manager import SnapshotManager


class TestSnapshotManager:
    def test_create(self):
        mgr = SnapshotManager()
        assert mgr.snapshot_count() == 0

    def test_snapshot_and_restore(self):
        mgr = SnapshotManager()
        state = {"nodes": ["a", "b"], "count": 2}
        mgr.snapshot("v1", state)
        restored = mgr.restore("v1")
        assert restored == state

    def test_delete(self):
        mgr = SnapshotManager()
        mgr.snapshot("v1", {"x": 1})
        assert mgr.delete("v1") is True
        assert mgr.restore("v1") is None
        assert mgr.delete("missing") is False

    def test_delta_between(self):
        mgr = SnapshotManager()
        mgr.snapshot("v1", {"x": 1, "y": 2})
        mgr.snapshot("v2", {"x": 1, "y": 3, "z": 4})
        deltas = mgr.delta_between("v1", "v2")
        assert len(deltas) == 2
        paths = {d["path"] for d in deltas}
        assert "/y" in paths
        assert "/z" in paths

    def test_delta_add_remove(self):
        mgr = SnapshotManager()
        mgr.snapshot("v1", {"x": 1})
        mgr.snapshot("v2", {"y": 2})
        deltas = mgr.delta_between("v1", "v2")
        ops = {d["op"] for d in deltas}
        assert "add" in ops
        assert "remove" in ops

    def test_deltas_since(self):
        mgr = SnapshotManager()
        mgr.snapshot("v1", {"x": 1})
        deltas = mgr.deltas_since("v1", {"x": 2})
        assert len(deltas) == 1
        assert deltas[0]["op"] == "replace"

    def test_apply_delta(self):
        mgr = SnapshotManager()
        state = {"x": 1}
        deltas = [{"op": "replace", "path": "/x", "value": 2}]
        result = mgr.apply_delta(state, deltas)
        assert result["x"] == 2

    def test_list_snapshots(self):
        mgr = SnapshotManager()
        mgr.snapshot("v1", {})
        mgr.snapshot("v2", {})
        assert sorted(mgr.list_snapshots()) == ["v1", "v2"]

    def test_oldest_newest(self):
        mgr = SnapshotManager()
        mgr.snapshot("v1", {}, metadata={})
        import time
        time.sleep(0.01)
        mgr.snapshot("v2", {}, metadata={})
        assert mgr.oldest_snapshot() == "v1"
        assert mgr.newest_snapshot() == "v2"

    def test_repr(self):
        mgr = SnapshotManager()
        mgr.snapshot("v1", {})
        assert "SnapshotManager" in repr(mgr)
