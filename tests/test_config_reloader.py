"""Tests for config_reloader.py — Hot-reload configuration.

Run: python3 -m pytest tests/test_config_reloader.py -v --tb=short
"""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from fleet.config_reloader import ConfigReloader, ReloadResult


class TestConfigReloader:
    def test_create(self):
        cr = ConfigReloader()
        assert cr.sections() == []

    def test_register(self):
        cr = ConfigReloader()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            cr.register("test", path)
            assert "test" in cr.sections()
        finally:
            os.unlink(path)

    def test_register_missing_file(self):
        cr = ConfigReloader()
        cr.register("test", "/nonexistent/path.txt")
        # Should not raise, just warn
        assert "test" in cr.sections()

    def test_check_no_change(self):
        cr = ConfigReloader()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            cr.register("test", path)
            results = cr.check()
            # First check loads
            assert len(results) == 1
            assert results[0].changed is True
            # Second check should find no changes
            results2 = cr.check()
            assert len(results2) == 0
        finally:
            os.unlink(path)

    def test_check_change(self):
        cr = ConfigReloader()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            cr.register("test", path)
            cr.check()  # initial load

            # Modify file
            time.sleep(0.1)
            with open(path, "w") as f:
                f.write("world")

            results = cr.check()
            assert len(results) == 1
            assert results[0].changed is True
            assert results[0].success is True
        finally:
            os.unlink(path)

    def test_validator_pass(self):
        cr = ConfigReloader()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            cr.register("test", path, validator=lambda x: True)
            results = cr.check()
            assert results[0].success is True
        finally:
            os.unlink(path)

    def test_validator_fail(self):
        cr = ConfigReloader()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            cr.register("test", path, validator=lambda x: False)
            results = cr.check()
            assert results[0].success is False
            assert "validation failed" in results[0].message
        finally:
            os.unlink(path)

    def test_callback(self):
        cr = ConfigReloader()
        changes = []
        cr.on_change(lambda name, old, new: changes.append((name, old, new)))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            cr.register("test", path)
            cr.check()
            assert len(changes) == 1
            assert changes[0][0] == "test"
        finally:
            os.unlink(path)

    def test_get(self):
        cr = ConfigReloader()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            cr.register("test", path)
            cr.check()
            assert cr.get("test") == "hello"
            assert cr.get("missing") is None
        finally:
            os.unlink(path)

    def test_force_reload(self):
        cr = ConfigReloader()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            cr.register("test", path)
            cr.check()  # initial load
            result = cr.force_reload("test")
            assert result.changed is True
            assert result.success is True
        finally:
            os.unlink(path)

    def test_force_reload_missing(self):
        cr = ConfigReloader()
        result = cr.force_reload("missing")
        assert result.success is False
        assert "not registered" in result.message

    def test_repr(self):
        cr = ConfigReloader()
        assert "ConfigReloader" in repr(cr)
