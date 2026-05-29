import pytest
from fleet.config_manager import ConfigEntry, ConfigManager


class TestConfigEntry:
    def test_to_dict(self):
        e = ConfigEntry(key="k", value="v", source="test", timestamp=0.0)
        d = e.to_dict()
        assert d["key"] == "k"
        assert d["value"] == "v"


class TestConfigManager:
    def test_init(self):
        cm = ConfigManager()
        assert cm.fleet_node_id == "default"
        assert "breeding.population_size" in cm._defaults

    def test_set_and_get(self):
        cm = ConfigManager()
        cm.set("test.key", 42)
        assert cm.get("test.key") == 42

    def test_get_default(self):
        cm = ConfigManager()
        assert cm.get("breeding.population_size") == 50

    def test_get_custom_default(self):
        cm = ConfigManager()
        assert cm.get("missing", 99) == 99

    def test_has(self):
        cm = ConfigManager()
        assert cm.has("breeding.population_size") is True
        assert cm.has("missing") is False

    def test_delete(self):
        cm = ConfigManager()
        cm.set("test", 1)
        assert cm.delete("test") is True
        assert cm.get("test") is None

    def test_delete_missing(self):
        cm = ConfigManager()
        assert cm.delete("missing") is False

    def test_get_all(self):
        cm = ConfigManager()
        all_cfg = cm.get_all()
        assert "breeding.population_size" in all_cfg
        assert all_cfg["breeding.population_size"] == 50

    def test_get_by_source(self):
        cm = ConfigManager()
        cm.set("a", 1, source="src1")
        cm.set("b", 2, source="src2")
        entries = cm.get_by_source("src1")
        assert len(entries) == 1
        assert entries[0].value == 1

    def test_get_by_prefix(self):
        cm = ConfigManager()
        cm.set("breeding.x", 1)
        cm.set("breeding.y", 2)
        cm.set("other.z", 3)
        prefixed = cm.get_by_prefix("breeding")
        assert len(prefixed) == 2

    def test_export_json(self):
        cm = ConfigManager()
        cm.set("test", 1)
        j = cm.export_json()
        assert "test" in j
        assert "defaults" in j

    def test_load_json(self):
        cm = ConfigManager()
        cm.set("test", 1)
        j = cm.export_json()
        cm2 = ConfigManager()
        cm2.load_json(j)
        assert cm2.get("test") == 1

    def test_get_stats(self):
        cm = ConfigManager()
        cm.set("a", 1, source="s1")
        cm.set("b", 2, source="s1")
        cm.set("c", 3, source="s2")
        stats = cm.get_stats()
        assert stats["total_keys"] == 3
        assert stats["sources"]["s1"] == 2

    def test_to_dict(self):
        cm = ConfigManager()
        cm.set("test", 1)
        d = cm.to_dict()
        assert d["configs"] == 1
