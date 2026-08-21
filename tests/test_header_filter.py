"""Tests for header_filter.py — HTTP header filtering and normalization.

Run: python3 -m pytest tests/test_header_filter.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.header_filter import HeaderFilter


class TestHeaderFilter:
    def test_create(self):
        hf = HeaderFilter()
        assert hf.stats()["allowed_count"] == 0

    def test_allow(self):
        hf = HeaderFilter()
        hf.allow("content-type")
        hf.allow("authorization")
        result = hf.process({"content-type": "json", "x-custom": "value"})
        assert "content-type" in result
        assert "x-custom" not in result

    def test_deny(self):
        hf = HeaderFilter()
        hf.deny("x-internal-token")
        result = hf.process({"content-type": "json", "x-internal-token": "secret"})
        assert "content-type" in result
        assert "x-internal-token" not in result

    def test_allow_and_deny(self):
        hf = HeaderFilter()
        hf.allow("content-type")
        hf.allow("authorization")
        hf.deny("authorization")
        result = hf.process({"content-type": "json", "authorization": "token"})
        assert "content-type" in result
        assert "authorization" not in result

    def test_require(self):
        hf = HeaderFilter()
        hf.require("authorization")
        result = hf.process({"content-type": "json"})
        assert "_missing_required" in result
        assert "authorization" in result["_missing_required"]

    def test_require_present(self):
        hf = HeaderFilter()
        hf.require("authorization")
        result = hf.process({"authorization": "token"})
        assert "_missing_required" not in result

    def test_transform(self):
        hf = HeaderFilter()
        hf.transform("content-type", lambda v: v.lower())
        result = hf.process({"content-type": "JSON"})
        assert result["content-type"] == "json"

    def test_validate(self):
        hf = HeaderFilter()
        hf.require("authorization")
        assert hf.validate({"authorization": "token"}) is True
        assert hf.validate({"content-type": "json"}) is False

    def test_clear_rules(self):
        hf = HeaderFilter()
        hf.allow("content-type")
        hf.deny("x-internal")
        hf.clear_rules()
        result = hf.process({"content-type": "json", "x-internal": "secret"})
        assert "content-type" in result
        assert "x-internal" in result

    def test_rules(self):
        hf = HeaderFilter()
        hf.allow("content-type")
        hf.deny("x-internal")
        hf.require("authorization")
        hf.transform("host", lambda v: v)
        rules = hf.rules()
        assert rules["allowed"] == ["content-type"]
        assert rules["denied"] == ["x-internal"]
        assert rules["required"] == ["authorization"]
        assert rules["transforms"] == ["host"]

    def test_stats(self):
        hf = HeaderFilter()
        hf.allow("a")
        hf.deny("b")
        hf.require("c")
        hf.transform("d", lambda v: v)
        stats = hf.stats()
        assert stats["allowed_count"] == 1
        assert stats["denied_count"] == 1
        assert stats["required_count"] == 1
        assert stats["transform_count"] == 1

    def test_repr(self):
        hf = HeaderFilter()
        assert "HeaderFilter" in repr(hf)
