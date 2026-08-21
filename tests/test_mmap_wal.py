"""Tests for mmap_wal.py — Memory-mapped append-only WAL.

Run: python3 -m pytest tests/test_mmap_wal.py -v --tb=short
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from logos.mmap_wal import MmapWAL, WALEntry


class TestMmapWALBasics:
    def test_create_and_append(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.wal"
            wal = MmapWAL(path)
            offset = wal.append(b"hello")
            assert offset >= 0
            assert wal.entry_count == 1
            wal.close()

    def test_read_back(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.wal"
            wal = MmapWAL(path)
            wal.append(b"payload_one")
            wal.append(b"payload_two")
            wal.close()

            # Reopen read-only
            wal2 = MmapWAL(path, readonly=True)
            entries = list(wal2)
            assert len(entries) == 2
            assert entries[0].payload == b"payload_one"
            assert entries[1].payload == b"payload_two"
            wal2.close()

    def test_checksum_verify(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.wal"
            wal = MmapWAL(path)
            wal.append(b"verify_me")
            wal.close()

            wal2 = MmapWAL(path, readonly=True)
            entries = list(wal2)
            assert len(entries) == 1
            assert entries[0].verify() is True
            wal2.close()

    def test_multiple_appends(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.wal"
            with MmapWAL(path) as wal:
                for i in range(100):
                    wal.append(f"entry_{i:03d}".encode())
                assert wal.entry_count == 100

            with MmapWAL(path, readonly=True) as wal2:
                entries = list(wal2)
                assert len(entries) == 100
                for i, e in enumerate(entries):
                    assert e.payload == f"entry_{i:03d}".encode()

    def test_read_at_offset(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.wal"
            with MmapWAL(path) as wal:
                off0 = wal.append(b"first")
                wal.append(b"second")

            with MmapWAL(path, readonly=True) as wal2:
                e = wal2.read_at(off0)
                assert e is not None
                assert e.payload == b"first"

    def test_read_at_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.wal"
            with MmapWAL(path) as wal:
                wal.append(b"only")

            with MmapWAL(path, readonly=True) as wal2:
                assert wal2.read_at(9999) is None

    def test_batch_append(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.wal"
            with MmapWAL(path) as wal:
                offsets = wal.append_batch([b"a", b"b", b"c"])
                assert len(offsets) == 3
                assert wal.entry_count == 3

    def test_tail(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.wal"
            with MmapWAL(path) as wal:
                for i in range(10):
                    wal.append(f"e{i}".encode())

            with MmapWAL(path, readonly=True) as wal2:
                tail = wal2.tail(n=3)
                assert len(tail) == 3
                # tail returns newest first
                assert tail[0].payload == b"e9"
                assert tail[1].payload == b"e8"
                assert tail[2].payload == b"e7"

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.wal"
            with MmapWAL(path) as wal:
                wal.append(b"cm")
            # After exit, should be closed
            assert wal._fd is None or wal._mm is None

    def test_empty_wal(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "empty.wal"
            with MmapWAL(path) as wal:
                pass
            with MmapWAL(path, readonly=True) as wal2:
                assert wal2.entry_count == 0
                assert list(wal2) == []

    def test_repr(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.wal"
            with MmapWAL(path) as wal:
                wal.append(b"x")
                r = repr(wal)
                assert "MmapWAL" in r
                assert "entries=1" in r

    def test_large_payload(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.wal"
            big = b"x" * (1024 * 1024)  # 1MB
            with MmapWAL(path) as wal:
                wal.append(big)
                assert wal.entry_count == 1
            with MmapWAL(path, readonly=True) as wal2:
                entries = list(wal2)
                assert len(entries) == 1
                assert entries[0].payload == big
                assert entries[0].verify() is True

    def test_file_growth(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.wal"
            with MmapWAL(path) as wal:
                # Write enough to trigger chunk growth
                for _ in range(1000):
                    wal.append(b"x" * 4096)
                # File should have grown beyond initial 4MB
                assert wal.file_size >= wal.byte_size
                assert wal.entry_count == 1000
