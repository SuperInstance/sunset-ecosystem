"""Tests for feature_flag.py — Feature flag system with rollouts.

Run: python3 -m pytest tests/test_feature_flag.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.feature_flag import FeatureFlags


class TestFeatureFlags:
    def test_create(self):
        flags = FeatureFlags()
        assert flags.stats()["flags"] == 0

    def test_set_boolean(self):
        flags = FeatureFlags()
        flags.set("dark_mode", True)
        assert flags.get("dark_mode") is True

    def test_set_percentage(self):
        flags = FeatureFlags()
        flags.set("beta", 50)
        assert flags.get("beta") == 50

    def test_remove(self):
        flags = FeatureFlags()
        flags.set("feature", True)
        assert flags.remove("feature") is True
        assert flags.remove("missing") is False

    def test_list_flags(self):
        flags = FeatureFlags()
        flags.set("a", True)
        flags.set("b", False)
        assert sorted(flags.list_flags()) == ["a", "b"]

    def test_is_enabled_boolean_true(self):
        flags = FeatureFlags()
        flags.set("feature", True)
        assert flags.is_enabled("feature") is True

    def test_is_enabled_boolean_false(self):
        flags = FeatureFlags()
        flags.set("feature", False)
        assert flags.is_enabled("feature") is False

    def test_is_enabled_percentage(self):
        flags = FeatureFlags()
        flags.set("feature", 50)
        # Deterministic hash-based check
        results = {flags.is_enabled("feature", user_id=f"user-{i}") for i in range(100)}
        assert True in results
        assert False in results

    def test_is_enabled_percentage_zero(self):
        flags = FeatureFlags()
        flags.set("feature", 0)
        assert flags.is_enabled("feature", user_id="user-1") is False

    def test_is_enabled_percentage_hundred(self):
        flags = FeatureFlags()
        flags.set("feature", 100)
        assert flags.is_enabled("feature", user_id="user-1") is True

    def test_is_enabled_missing(self):
        flags = FeatureFlags()
        assert flags.is_enabled("missing") is False

    def test_override(self):
        flags = FeatureFlags()
        flags.set("feature", 0)
        flags.add_override("user-1", "feature")
        assert flags.is_enabled("feature", user_id="user-1") is True
        assert flags.is_enabled("feature", user_id="user-2") is False

    def test_remove_override(self):
        flags = FeatureFlags()
        flags.set("feature", 0)
        flags.add_override("user-1", "feature")
        flags.remove_override("user-1", "feature")
        assert flags.is_enabled("feature", user_id="user-1") is False

    def test_defaults(self):
        flags = FeatureFlags(defaults={"feature": True})
        assert flags.is_enabled("feature") is True

    def test_repr(self):
        flags = FeatureFlags()
        assert "FeatureFlags" in repr(flags)
