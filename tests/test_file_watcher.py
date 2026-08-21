"""Tests for file_watcher.py — File system watcher with debounce.

Run: python3 -m pytest tests/test_file_watcher.py -v --tb=short
"""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from fleet.file_watcher import FileWatcher


class TestFileWatcher:
    def test_create(self):
        watcher = FileWatcher()
        assert len(watcher.watched_paths()) == 0

    def test_watch_and_no_change(self):
        watcher = FileWatcher()
        called = []
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            watcher.watch(path, lambda p: called.append(p))
            changed = watcher.check()
            assert len(changed) == 0
        finally:
            os.unlink(path)

    def test_detect_change(self):
        watcher = FileWatcher(debounce_sec=0.0)
        called = []
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            watcher.watch(path, lambda p: called.append(p))
            watcher.check()  # establish baseline
            time.sleep(0.01)
            with open(path, "w") as f:
                f.write("world")
            changed = watcher.check()
            assert len(changed) == 1
            assert called[0] == path
        finally:
            os.unlink(path)

    def test_debounce(self):
        watcher = FileWatcher(debounce_sec=0.5)
        called = []
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            watcher.watch(path, lambda p: called.append(p))
            watcher.check()  # baseline
            time.sleep(0.01)
            with open(path, "w") as f:
                f.write("world")
            watcher.check()  # first change triggers
            assert len(called) == 1
            # rapid second change should debounce
            with open(path, "w") as f:
                f.write("again")
            changed = watcher.check()
            assert len(changed) == 0
            assert len(watcher._pending) == 1
        finally:
            os.unlink(path)

    def test_flush_pending(self):
        watcher = FileWatcher(debounce_sec=0.05)
        called = []
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            watcher.watch(path, lambda p: called.append(p))
            watcher.check()  # baseline
            time.sleep(0.01)
            with open(path, "w") as f:
                f.write("world")
            watcher.check()  # first change triggers
            assert len(called) == 1
            # rapid second change debounced
            with open(path, "w") as f:
                f.write("again")
            watcher.check()
            time.sleep(0.06)
            flushed = watcher.flush_pending()
            assert len(flushed) == 1
            assert len(called) == 2
        finally:
            os.unlink(path)

    def test_unwatch(self):
        watcher = FileWatcher()
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            path = f.name
        try:
            watcher.watch(path, lambda p: None)
            watcher.unwatch(path)
            assert len(watcher.watched_paths()) == 0
        finally:
            os.unlink(path)

    def test_repr(self):
        watcher = FileWatcher()
        assert "FileWatcher" in repr(watcher)
