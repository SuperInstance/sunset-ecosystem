import pytest
from fleet.feature_flags import FeatureFlag, FeatureFlagManager


class TestFeatureFlag:
    def test_is_enabled_for_disabled(self):
        f = FeatureFlag(name="test", enabled=False)
        assert f.is_enabled_for({}) is False

    def test_is_enabled_for_enabled(self):
        f = FeatureFlag(name="test", enabled=True)
        assert f.is_enabled_for({}) is True

    def test_is_enabled_for_targeting(self):
        f = FeatureFlag(name="test", enabled=True, targeting={"env": "prod"})
        assert f.is_enabled_for({"env": "prod"}) is True
        assert f.is_enabled_for({"env": "dev"}) is False

    def test_is_enabled_for_rollout(self):
        f = FeatureFlag(name="test", enabled=True, rollout_percentage=50.0)
        # Deterministic: some contexts will be True, some False
        results = [f.is_enabled_for({"user": i}) for i in range(100)]
        assert any(results)
        assert not all(results)

    def test_to_dict(self):
        f = FeatureFlag(name="test", enabled=True, rollout_percentage=50.0)
        d = f.to_dict()
        assert d["name"] == "test"
        assert d["rollout_percentage"] == 50.0


class TestFeatureFlagManager:
    def test_init(self):
        m = FeatureFlagManager()
        assert m.fleet_node_id == "default"

    def test_create(self):
        m = FeatureFlagManager()
        f = m.create("feat", enabled=True, rollout_percentage=50.0)
        assert f.name == "feat"
        assert f.enabled is True

    def test_enable_disable(self):
        m = FeatureFlagManager()
        m.create("feat", enabled=False)
        assert m.enable("feat") is True
        assert m.check("feat") is True
        assert m.disable("feat") is True
        assert m.check("feat") is False

    def test_enable_missing(self):
        m = FeatureFlagManager()
        assert m.enable("missing") is False

    def test_set_rollout(self):
        m = FeatureFlagManager()
        m.create("feat", enabled=True, rollout_percentage=0.0)
        assert m.set_rollout("feat", 75.0) is True
        assert m.get("feat").rollout_percentage == 75.0

    def test_check_missing(self):
        m = FeatureFlagManager()
        assert m.check("missing") is False

    def test_list_flags(self):
        m = FeatureFlagManager()
        m.create("a")
        m.create("b")
        flags = m.list_flags()
        assert len(flags) == 2

    def test_get_stats(self):
        m = FeatureFlagManager()
        m.create("a", enabled=True)
        m.create("b", enabled=False)
        m.check("a")
        m.check("b")
        stats = m.get_stats()
        assert stats["total_flags"] == 2
        assert stats["enabled_flags"] == 1
        assert stats["evaluations"]["enabled"] == 1

    def test_export_json(self):
        m = FeatureFlagManager()
        m.create("feat", enabled=True)
        j = m.export_json()
        assert "feat" in j

    def test_to_dict(self):
        m = FeatureFlagManager()
        m.create("feat")
        d = m.to_dict()
        assert d["stats"]["total_flags"] == 1
