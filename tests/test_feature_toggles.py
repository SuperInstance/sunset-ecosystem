"""Tests for feature_toggles.py — Feature toggles with rollout percentages and user targeting.

Run: python3 -m pytest tests/test_feature_toggles.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.feature_toggles import FeatureToggles


class TestFeatureToggles:
    def test_create(self):
        toggles = FeatureToggles()
        assert toggles.stats()["features"] == 0

    def test_register(self):
        toggles = FeatureToggles()
        assert toggles.register("new-ui", default=False, rollout=10) is True
        assert "new-ui" in toggles.features()

    def test_register_duplicate(self):
        toggles = FeatureToggles()
        toggles.register("new-ui")
        assert toggles.register("new-ui") is False

    def test_is_enabled_default(self):
        toggles = FeatureToggles()
        toggles.register("new-ui", default=False)
        assert toggles.is_enabled("new-ui") is False
        toggles.register("always-on", default=True)
        assert toggles.is_enabled("always-on") is True

    def test_set_enabled(self):
        toggles = FeatureToggles()
        toggles.register("new-ui", default=False)
        assert toggles.set_enabled("new-ui", True) is True
        assert toggles.is_enabled("new-ui") is True
        assert toggles.set_enabled("missing", True) is False

    def test_rollout(self):
        toggles = FeatureToggles()
        toggles.register("new-ui", default=False, rollout=50)
        # With a fixed user_id, result is deterministic
        user_id = "test-user-123"
        result = toggles.is_enabled("new-ui", user_id=user_id)
        # Just verify it's deterministic
        assert toggles.is_enabled("new-ui", user_id=user_id) == result
        assert toggles.is_enabled("new-ui", user_id="other-user") == toggles.is_enabled(
            "new-ui", user_id="other-user"
        )

    def test_rollout_zero(self):
        toggles = FeatureToggles()
        toggles.register("new-ui", default=False, rollout=0)
        assert toggles.is_enabled("new-ui", user_id="any") is False

    def test_rollout_100(self):
        toggles = FeatureToggles()
        toggles.register("new-ui", default=False, rollout=100)
        assert toggles.is_enabled("new-ui", user_id="any") is True

    def test_user_override(self):
        toggles = FeatureToggles()
        toggles.register("new-ui", default=False)
        toggles.set_user_override("new-ui", "user-1", True)
        assert toggles.is_enabled("new-ui", user_id="user-1") is True
        assert toggles.is_enabled("new-ui", user_id="user-2") is False

    def test_user_override_disable(self):
        toggles = FeatureToggles()
        toggles.register("new-ui", default=True)
        toggles.set_user_override("new-ui", "user-1", False)
        assert toggles.is_enabled("new-ui", user_id="user-1") is False

    def test_remove_user_override(self):
        toggles = FeatureToggles()
        toggles.register("new-ui", default=False)
        toggles.set_user_override("new-ui", "user-1", True)
        assert toggles.remove_user_override("new-ui", "user-1") is True
        assert toggles.is_enabled("new-ui", user_id="user-1") is False
        assert toggles.remove_user_override("new-ui", "missing") is False

    def test_set_rollout(self):
        toggles = FeatureToggles()
        toggles.register("new-ui", default=False, rollout=10)
        assert toggles.set_rollout("new-ui", 50) is True
        assert toggles.get_feature("new-ui")["rollout"] == 50
        assert toggles.set_rollout("missing", 50) is False

    def test_get_feature(self):
        toggles = FeatureToggles()
        toggles.register("new-ui", default=False, rollout=10, description="New UI")
        feature = toggles.get_feature("new-ui")
        assert feature["enabled"] is False
        assert feature["rollout"] == 10
        assert feature["description"] == "New UI"

    def test_get_user_override(self):
        toggles = FeatureToggles()
        toggles.register("new-ui", default=False)
        assert toggles.get_user_override("new-ui", "user-1") is None
        toggles.set_user_override("new-ui", "user-1", True)
        assert toggles.get_user_override("new-ui", "user-1") is True

    def test_stats(self):
        toggles = FeatureToggles()
        toggles.register("a", default=True)
        toggles.register("b", default=False, rollout=10)
        toggles.register("c", default=False)
        toggles.set_user_override("a", "user-1", True)
        stats = toggles.stats()
        assert stats["features"] == 3
        assert stats["enabled"] == 1
        assert stats["rollout"] == 1
        assert stats["disabled"] == 1
        assert stats["user_overrides"] == 1

    def test_repr(self):
        toggles = FeatureToggles()
        assert "FeatureToggles" in repr(toggles)
