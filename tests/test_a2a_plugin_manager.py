import pytest
from fleet.a2a_plugin_manager import A2APlugin, A2APluginManager


class TestA2APlugin:
    def test_to_dict(self):
        p = A2APlugin(name="test", version="1.0", handler=lambda x: x)
        d = p.to_dict()
        assert d["name"] == "test"
        assert d["enabled"] is True


class TestA2APluginManager:
    def test_init(self):
        pm = A2APluginManager()
        assert pm.plugins == {}
        assert pm.fleet_node_id == "default"

    def test_register(self):
        pm = A2APluginManager()
        p = pm.register("test", "1.0", lambda x: x + 1)
        assert "test" in pm.plugins
        assert p.name == "test"

    def test_register_with_hooks(self):
        pm = A2APluginManager()
        pm.register("test", "1.0", lambda x: x, hooks=["pre_breed"])
        assert "pre_breed" in pm._hooks
        assert "test" in pm._hooks["pre_breed"]

    def test_unregister(self):
        pm = A2APluginManager()
        pm.register("test", "1.0", lambda x: x)
        assert pm.unregister("test") is True
        assert "test" not in pm.plugins

    def test_unregister_missing(self):
        pm = A2APluginManager()
        assert pm.unregister("missing") is False

    def test_get(self):
        pm = A2APluginManager()
        pm.register("test", "1.0", lambda x: x)
        assert pm.get("test") is not None
        assert pm.get("missing") is None

    def test_list_plugins(self):
        pm = A2APluginManager()
        pm.register("a", "1.0", lambda x: x)
        pm.register("b", "1.0", lambda x: x)
        plugins = pm.list_plugins()
        assert len(plugins) == 2

    def test_list_plugins_enabled_only(self):
        pm = A2APluginManager()
        pm.register("a", "1.0", lambda x: x)
        pm.register("b", "1.0", lambda x: x)
        pm.disable("b")
        plugins = pm.list_plugins(enabled_only=True)
        assert len(plugins) == 1

    def test_invoke(self):
        pm = A2APluginManager()
        pm.register("adder", "1.0", lambda x, y: x + y)
        result = pm.invoke("adder", 2, 3)
        assert result == 5

    def test_invoke_not_found(self):
        pm = A2APluginManager()
        with pytest.raises(ValueError):
            pm.invoke("missing")

    def test_invoke_disabled(self):
        pm = A2APluginManager()
        pm.register("test", "1.0", lambda x: x)
        pm.disable("test")
        with pytest.raises(ValueError):
            pm.invoke("test")

    def test_invoke_hook(self):
        pm = A2APluginManager()
        pm.register("a", "1.0", lambda x: x * 2, hooks=["test_hook"])
        results = pm.invoke_hook("test_hook", 5)
        assert len(results) == 1
        assert results[0]["result"] == 10

    def test_invoke_hook_multiple(self):
        pm = A2APluginManager()
        pm.register("a", "1.0", lambda x: x * 2, hooks=["test_hook"])
        pm.register("b", "1.0", lambda x: x + 1, hooks=["test_hook"])
        results = pm.invoke_hook("test_hook", 5)
        assert len(results) == 2

    def test_enable_disable(self):
        pm = A2APluginManager()
        pm.register("test", "1.0", lambda x: x)
        pm.disable("test")
        assert pm.plugins["test"].enabled is False
        pm.enable("test")
        assert pm.plugins["test"].enabled is True

    def test_get_stats(self):
        pm = A2APluginManager()
        pm.register("a", "1.0", lambda x: x)
        pm.register("b", "1.0", lambda x: x)
        pm.disable("b")
        stats = pm.get_stats()
        assert stats["total"] == 2
        assert stats["enabled"] == 1

    def test_export_manifest(self):
        pm = A2APluginManager()
        pm.register("a", "1.0", lambda x: x)
        manifest = pm.export_manifest()
        assert "a" in manifest

    def test_to_dict(self):
        pm = A2APluginManager()
        pm.register("a", "1.0", lambda x: x)
        d = pm.to_dict()
        assert "stats" in d
