"""Tests for plugin_registry.py — Plugin management and discovery.

Run: python3 -m pytest tests/test_plugin_registry.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.plugin_registry import PluginRegistry, PluginError


class TestPluginRegistry:
    def test_create(self):
        reg = PluginRegistry()
        assert reg.list_plugins() == []

    def test_register(self):
        reg = PluginRegistry()
        reg.register("test", {"data": 1}, version="1.0.0")
        assert "test" in reg.list_plugins()

    def test_get(self):
        reg = PluginRegistry()
        reg.register("test", {"data": 1})
        assert reg.get("test") == {"data": 1}

    def test_get_missing(self):
        reg = PluginRegistry()
        with pytest.raises(PluginError):
            reg.get("missing")

    def test_unregister(self):
        reg = PluginRegistry()
        reg.register("test", {})
        assert reg.unregister("test") is True
        assert reg.unregister("test") is False

    def test_find_by_capability(self):
        reg = PluginRegistry()
        reg.register("a", {}, capabilities=["breed", "eval"])
        reg.register("b", {}, capabilities=["breed"])
        reg.register("c", {}, capabilities=["eval"])
        assert sorted(reg.find_by_capability("breed")) == ["a", "b"]
        assert reg.find_by_capability("eval") == ["a", "c"]

    def test_resolve_simple(self):
        reg = PluginRegistry()
        reg.register("a", {})
        reg.register("b", {})
        order = reg.resolve(["a", "b"])
        assert order == ["a", "b"]

    def test_resolve_with_deps(self):
        reg = PluginRegistry()
        reg.register("base", {})
        reg.register("derived", {}, dependencies=["base"])
        order = reg.resolve(["derived", "base"])
        assert order == ["base", "derived"]

    def test_resolve_missing(self):
        reg = PluginRegistry()
        assert reg.resolve(["missing"]) is None

    def test_resolve_cycle(self):
        reg = PluginRegistry()
        reg.register("a", {}, dependencies=["b"])
        reg.register("b", {}, dependencies=["a"])
        assert reg.resolve(["a", "b"]) is None

    def test_check_compatibility(self):
        reg = PluginRegistry()
        reg.register("p", {}, version="2.0.0")
        assert reg.check_compatibility("p", "1.0.0") is True
        assert reg.check_compatibility("p", "3.0.0") is False
        assert reg.check_compatibility("missing", "1.0.0") is False

    def test_reload(self):
        reg = PluginRegistry()
        reg.register("test", {"old": True})
        reg.reload("test", {"new": True})
        assert reg.get("test") == {"new": True}

    def test_reload_missing(self):
        reg = PluginRegistry()
        assert reg.reload("missing", {}) is False

    def test_disable_enable(self):
        reg = PluginRegistry()
        reg.register("test", {})
        assert reg.active_plugins() == ["test"]
        reg.disable("test")
        assert reg.active_plugins() == []
        reg.enable("test")
        assert reg.active_plugins() == ["test"]

    def test_get_info(self):
        reg = PluginRegistry()
        reg.register("test", {"data": 1}, version="1.2.3", capabilities=["c1"])
        info = reg.get_info("test")
        assert info is not None
        assert info.version == "1.2.3"
        assert info.capabilities == ["c1"]

    def test_stats(self):
        reg = PluginRegistry()
        reg.register("a", {}, capabilities=["c1"])
        reg.register("b", {}, capabilities=["c2"])
        stats = reg.stats()
        assert stats["total"] == 2
        assert stats["active"] == 2
        assert sorted(stats["capabilities"]) == ["c1", "c2"]

    def test_repr(self):
        reg = PluginRegistry()
        assert "PluginRegistry" in repr(reg)
