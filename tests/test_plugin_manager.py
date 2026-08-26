"""Tests for plugin_manager.py — Plugin discovery and lifecycle management.

Run: python3 -m pytest tests/test_plugin_manager.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.plugin_manager import Plugin, PluginManager


class TestPlugin(Plugin):
    def run(self, *args, **kwargs):
        return "test"


class TestPluginManager:
    def test_create(self):
        mgr = PluginManager()
        assert mgr.stats()["registered"] == 0

    def test_register(self):
        mgr = PluginManager()
        mgr.register("test", TestPlugin)
        assert mgr.stats()["registered"] == 1

    def test_unregister(self):
        mgr = PluginManager()
        mgr.register("test", TestPlugin)
        assert mgr.unregister("test") is True
        assert mgr.unregister("missing") is False

    def test_load(self):
        mgr = PluginManager()
        mgr.register("test", TestPlugin)
        assert mgr.load("test") is True
        assert mgr.is_loaded("test") is True
        assert mgr.get("test") is not None

    def test_load_with_config(self):
        mgr = PluginManager()
        mgr.register("test", TestPlugin)
        mgr.load("test", config={"key": "value"})
        plugin = mgr.get("test")
        assert plugin.config == {"key": "value"}

    def test_load_missing(self):
        mgr = PluginManager()
        assert mgr.load("missing") is False

    def test_unload(self):
        mgr = PluginManager()
        mgr.register("test", TestPlugin)
        mgr.load("test")
        assert mgr.unload("test") is True
        assert mgr.is_loaded("test") is False
        assert mgr.unload("missing") is False

    def test_reload(self):
        mgr = PluginManager()
        mgr.register("test", TestPlugin)
        mgr.load("test")
        assert mgr.reload("test") is True

    def test_dependencies(self):
        mgr = PluginManager()
        mgr.register("base", TestPlugin)
        mgr.register("extended", TestPlugin, dependencies=["base"])
        mgr.load("extended")
        assert mgr.is_loaded("base") is True
        assert mgr.is_loaded("extended") is True

    def test_loaded_plugins(self):
        mgr = PluginManager()
        mgr.register("a", TestPlugin)
        mgr.register("b", TestPlugin)
        mgr.load("a")
        assert mgr.loaded_plugins() == ["a"]

    def test_available_plugins(self):
        mgr = PluginManager()
        mgr.register("a", TestPlugin)
        assert mgr.available_plugins() == ["a"]

    def test_repr(self):
        mgr = PluginManager()
        assert "PluginManager" in repr(mgr)
