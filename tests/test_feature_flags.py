"""Tests for feature_flags.py — Dynamic feature toggle system.

Run: python3 -m pytest tests/test_feature_flags.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.feature_flags import FeatureFlags, FlagDefinition


class TestFeatureFlags:
    def test_create(self):
        ff = FeatureFlags()
        assert len(ff.all_flags()) == 0

    def test_define(self):
        ff = FeatureFlags()
        flag = ff.define("new_feature", default=False, rollout_percent=10.0)
        assert flag.name == "new_feature"
        assert flag.default is False
        assert flag.rollout_percent == 10.0

    def test_is_enabled_default(self):
        ff = FeatureFlags()
        ff.define("flag_a", default=True)
        ff.define("flag_b", default=False)
        assert ff.is_enabled("flag_a") is True
        assert ff.is_enabled("flag_b") is False

    def test_is_enabled_unknown(self):
        ff = FeatureFlags()
        assert ff.is_enabled("missing") is False

    def test_rollout(self):
        ff = FeatureFlags()
        ff.define("rollout", default=False, rollout_percent=50.0)
        enabled_count = sum(
            1 for i in range(100)
            if ff.is_enabled("rollout", agent_id=f"agent-{i}")
        )
        # With 50% rollout, should be roughly 50
        assert 30 < enabled_count < 70

    def test_override(self):
        ff = FeatureFlags()
        ff.define("flag", default=False)
        ff.set_override("flag", "agent-1", True)
        assert ff.is_enabled("flag", "agent-1") is True
        assert ff.is_enabled("flag", "agent-2") is False

    def test_remove_override(self):
        ff = FeatureFlags()
        ff.define("flag", default=False)
        ff.set_override("flag", "a", True)
        ff.remove_override("flag", "a")
        assert ff.is_enabled("flag", "a") is False

    def test_set_default(self):
        ff = FeatureFlags()
        ff.define("flag", default=False)
        ff.set_default("flag", True)
        assert ff.is_enabled("flag") is True

    def test_set_default_callback(self):
        ff = FeatureFlags()
        ff.define("flag", default=False)
        changes = []
        ff.on_change(lambda name, val: changes.append((name, val)))
        ff.set_default("flag", True)
        assert changes == [("flag", True)]

    def test_set_rollout(self):
        ff = FeatureFlags()
        ff.define("flag", rollout_percent=0.0)
        ff.set_rollout("flag", 100.0)
        assert ff.is_enabled("flag", "any") is True

    def test_enabled_flags(self):
        ff = FeatureFlags()
        ff.define("a", default=True)
        ff.define("b", default=False)
        enabled = ff.enabled_flags()
        assert enabled == ["a"]

    def test_report(self):
        ff = FeatureFlags()
        ff.define("a", default=True, rollout_percent=10.0)
        r = ff.report()
        assert "a" in r
        assert r["a"]["default"] is True
        assert r["a"]["rollout"] == 10.0

    def test_to_dict_roundtrip(self):
        ff = FeatureFlags()
        ff.define("a", default=True, rollout_percent=50.0)
        ff.set_override("a", "x", False)
        d = ff.to_dict()
        ff2 = FeatureFlags.from_dict(d)
        assert ff2.is_enabled("a") is True
        assert ff2.is_enabled("a", "x") is False
        assert ff2._flags["a"].rollout_percent == 50.0

    def test_set_default_invalid(self):
        ff = FeatureFlags()
        with pytest.raises(ValueError):
            ff.set_default("missing", True)

    def test_set_override_invalid(self):
        ff = FeatureFlags()
        with pytest.raises(ValueError):
            ff.set_override("missing", "a", True)

    def test_repr(self):
        ff = FeatureFlags()
        ff.define("a")
        assert "FeatureFlags" in repr(ff)
