"""Tests for version_manager.py — Semantic versioning.

Run: python3 -m pytest tests/test_version_manager.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.version_manager import SemVer, VersionManager


class TestSemVer:
    def test_parse_basic(self):
        v = SemVer.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_parse_prerelease(self):
        v = SemVer.parse("2.0.0-alpha.1")
        assert v.prerelease == "alpha.1"

    def test_parse_build(self):
        v = SemVer.parse("1.0.0+build.42")
        assert v.build == "build.42"

    def test_parse_invalid(self):
        with pytest.raises(ValueError):
            SemVer.parse("not-a-version")

    def test_str(self):
        assert str(SemVer.parse("1.2.3")) == "1.2.3"
        assert str(SemVer.parse("1.0.0-alpha+build")) == "1.0.0-alpha+build"

    def test_comparison(self):
        assert SemVer.parse("1.0.0") < SemVer.parse("2.0.0")
        assert SemVer.parse("1.0.0") < SemVer.parse("1.1.0")
        assert SemVer.parse("1.0.0") < SemVer.parse("1.0.1")
        assert SemVer.parse("1.0.0-alpha") < SemVer.parse("1.0.0")
        assert SemVer.parse("1.0.0") == SemVer.parse("1.0.0")

    def test_equality_ignores_build(self):
        assert SemVer.parse("1.0.0+build1") == SemVer.parse("1.0.0+build2")

    def test_hash(self):
        assert hash(SemVer.parse("1.0.0")) == hash(SemVer.parse("1.0.0"))


class TestVersionManager:
    def test_create(self):
        vm = VersionManager()
        assert vm.list_components() == []

    def test_register_and_get(self):
        vm = VersionManager()
        vm.register("breeder", "1.2.3")
        assert str(vm.get("breeder")) == "1.2.3"

    def test_is_compatible_ge(self):
        vm = VersionManager()
        vm.register("breeder", "1.2.3")
        assert vm.is_compatible("breeder", ">=1.0.0") is True
        assert vm.is_compatible("breeder", ">=2.0.0") is False

    def test_is_compatible_gt(self):
        vm = VersionManager()
        vm.register("breeder", "1.2.3")
        assert vm.is_compatible("breeder", ">1.0.0") is True
        assert vm.is_compatible("breeder", ">1.2.3") is False

    def test_is_compatible_eq(self):
        vm = VersionManager()
        vm.register("breeder", "1.2.3")
        assert vm.is_compatible("breeder", "=1.2.3") is True
        assert vm.is_compatible("breeder", "=1.0.0") is False

    def test_is_compatible_caret(self):
        vm = VersionManager()
        vm.register("breeder", "1.2.3")
        assert vm.is_compatible("breeder", "^1.0.0") is True
        assert vm.is_compatible("breeder", "^2.0.0") is False

    def test_is_compatible_tilde(self):
        vm = VersionManager()
        vm.register("breeder", "1.2.3")
        assert vm.is_compatible("breeder", "~1.2.0") is True
        assert vm.is_compatible("breeder", "~1.1.0") is False

    def test_require_pass(self):
        vm = VersionManager()
        vm.register("breeder", "1.2.3")
        vm.require("breeder", ">=1.0.0")

    def test_require_fail(self):
        vm = VersionManager()
        vm.register("breeder", "1.2.3")
        with pytest.raises(ValueError):
            vm.require("breeder", ">=2.0.0")

    def test_feature_enabled(self):
        vm = VersionManager(features={"flux-gating": ("breeder", "1.2.0")})
        vm.register("breeder", "1.2.3")
        assert vm.feature_enabled("breeder", "flux-gating") is True

    def test_feature_disabled_old_version(self):
        vm = VersionManager(features={"flux-gating": ("breeder", "1.2.0")})
        vm.register("breeder", "1.1.0")
        assert vm.feature_enabled("breeder", "flux-gating") is False

    def test_feature_unknown(self):
        vm = VersionManager()
        vm.register("breeder", "1.0.0")
        assert vm.feature_enabled("breeder", "missing") is False

    def test_register_feature(self):
        vm = VersionManager()
        vm.register_feature("new-thing", "breeder", "2.0.0")
        vm.register("breeder", "2.1.0")
        assert vm.feature_enabled("breeder", "new-thing") is True

    def test_summary(self):
        vm = VersionManager()
        vm.register("a", "1.0.0")
        vm.register("b", "2.0.0")
        assert vm.summary() == {"a": "1.0.0", "b": "2.0.0"}

    def test_repr(self):
        vm = VersionManager()
        assert "VersionManager" in repr(vm)
