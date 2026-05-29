"""Tests for local_wal.py — Write-ahead log for durability.

Run: python3 -m pytest tests/test_local_wal.py -v --tb=short
"""
from __future__ import annotations

import os
import tempfile

import pytest

from fleet.local_wal import LocalWAL


class TestLocalWAL:
    def test_create(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = LocalWAL(os.path.join(tmpdir, "wal"))
            assert wal.stats()["appended"] == 0

    def test_append_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = LocalWAL(os.path.join(tmpdir, "wal"))
            wal.append({"op": "set", "key": "x", "value": 1})
            wal.append({"op": "set", "key": "y", "value": 2})
            entries = wal.read_all()
            assert len(entries) == 2
            assert entries[0]["key"] == "x"

    def test_append_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = LocalWAL(os.path.join(tmpdir, "wal"))
            wal.append_batch([
                {"op": "a"},
                {"op": "b"},
            ])
            assert wal.stats()["appended"] == 2

    def test_truncate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = LocalWAL(os.path.join(tmpdir, "wal"))
            wal.append({"op": "a"})
            wal.append({"op": "b"})
            wal.truncate()
            assert wal.read_all() == []
            assert wal.stats()["truncated"] == 1

    def test_reopen_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "wal")
            wal = LocalWAL(path)
            wal.append({"x": 1})
            wal2 = LocalWAL(path)
            entries = wal2.read_all()
            assert len(entries) == 1
            assert entries[0]["x"] == 1

    def test_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = LocalWAL(os.path.join(tmpdir, "wal"))
            wal.append({"op": "set", "key": "x", "value": 1})
            results = []
            wal.replay(lambda entry: results.append(entry["key"]))
            assert results == ["x"]

    def test_bad_lines_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "wal")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write("not json\n")
                f.write('{"valid": true}\n')
            wal = LocalWAL(path)
            entries = wal.read_all()
            assert len(entries) == 1
            assert entries[0]["valid"] is True

    def test_repr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = LocalWAL(os.path.join(tmpdir, "wal"))
            assert "LocalWAL" in repr(wal)
