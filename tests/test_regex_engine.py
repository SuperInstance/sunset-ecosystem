"""Tests for regex_engine.py — Regex-based rule engine.

Run: python3 -m pytest tests/test_regex_engine.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.regex_engine import RegexRuleEngine, MatchResult


class TestRegexRuleEngine:
    def test_create(self):
        engine = RegexRuleEngine()
        assert engine.rule_count() == 0

    def test_match_allow(self):
        engine = RegexRuleEngine()
        engine.add_rule("allow", r"^room\.(?P<room>\w+)\.trap$", priority=10)
        result = engine.match("room.alpha.trap")
        assert result.matched is True
        assert result.action == "allow"
        assert result.groups == {"room": "alpha"}

    def test_match_deny(self):
        engine = RegexRuleEngine()
        engine.add_rule("deny", r"^admin\..*", priority=100)
        result = engine.match("admin.secret")
        assert result.matched is True
        assert result.action == "deny"

    def test_default_action(self):
        engine = RegexRuleEngine(default_action="allow")
        result = engine.match("unknown")
        assert result.matched is False
        assert result.action == "allow"

    def test_priority_order(self):
        engine = RegexRuleEngine()
        engine.add_rule("deny", r"room\..*", priority=100)
        engine.add_rule("allow", r"room\.alpha.*", priority=10)
        result = engine.match("room.alpha.trap")
        assert result.action == "deny"

    def test_match_all(self):
        engine = RegexRuleEngine()
        engine.add_rule("tag1", r"^a.*", priority=1)
        engine.add_rule("tag2", r"^ab.*", priority=2)
        results = engine.match_all("abc")
        assert len(results) == 2

    def test_remove_rule(self):
        engine = RegexRuleEngine()
        engine.add_rule("allow", r"^test$", name="test-rule")
        assert engine.remove_rule("test-rule") is True
        assert engine.remove_rule("missing") is False

    def test_rule_with_name(self):
        engine = RegexRuleEngine()
        engine.add_rule("allow", r"^x$", name="my-rule")
        result = engine.match("x")
        assert result.rule_name == "my-rule"

    def test_repr(self):
        engine = RegexRuleEngine()
        assert "RegexRuleEngine" in repr(engine)
