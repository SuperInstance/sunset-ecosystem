"""Tests for log_rotator.py — Log file rotation.

Run: python3 -m pytest tests/test_log_rotator.py -v --tb=short
"""

from __future__ import annotations

import os
import tempfile

import pytest

from fleet.log_rotator import LogRotator, RotationPolicy


class TestLogRotator:
    def test_create(self):
        rotator = LogRotator()
        assert rotator.stats()["rotations"] == 0

    def test_write_and_rotate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.log")
            rotator = LogRotator(policy=RotationPolicy(max_size=10), compress=False)
            rotator.write(path, "a" * 20)
            assert rotator.should_rotate(path) is True
            rotated = rotator.rotate(path)
            assert os.path.exists(rotated)
            assert not os.path.exists(path)
            assert rotator.stats()["rotations"] == 1

    def test_no_rotate_small_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.log")
            rotator = LogRotator(policy=RotationPolicy(max_size=1000))
            rotator.write(path, "small")
            assert rotator.should_rotate(path) is False

    def test_cleanup_old(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.log")
            rotator = LogRotator(
                policy=RotationPolicy(max_size=1, max_files=2), compress=False
            )
            for _ in range(4):
                rotator.write(path, "x")
                rotator.rotate(path)
            rotated = rotator.list_rotated(path)
            assert len(rotated) <= 2

    def test_list_rotated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.log")
            rotator = LogRotator(policy=RotationPolicy(max_size=1), compress=False)
            rotator.write(path, "x")
            rotator.rotate(path)
            assert len(rotator.list_rotated(path)) == 1

    def test_compress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.log")
            rotator = LogRotator(
                policy=RotationPolicy(max_size=1, max_files=5), compress=True
            )
            rotator.write(path, "x")
            rotated = rotator.rotate(path)
            assert rotated.endswith(".gz")

    def test_repr(self):
        rotator = LogRotator()
        assert "LogRotator" in repr(rotator)
