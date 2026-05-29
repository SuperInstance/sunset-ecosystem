"""Tests for template_engine.py — Lightweight template renderer.

Run: python3 -m pytest tests/test_template_engine.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.template_engine import TemplateEngine, TemplateError


class TestTemplateEngine:
    def test_create(self):
        engine = TemplateEngine()
        assert repr(engine) == "<TemplateEngine>"

    def test_simple_var(self):
        engine = TemplateEngine()
        result = engine.render("Hello {{ name }}!", {"name": "Fleet"})
        assert result == "Hello Fleet!"

    def test_multiple_vars(self):
        engine = TemplateEngine()
        result = engine.render("{{ a }} and {{ b }}", {"a": 1, "b": 2})
        assert result == "1 and 2"

    def test_missing_var_unchanged(self):
        engine = TemplateEngine()
        result = engine.render("Hello {{ missing }}!", {})
        assert result == "Hello {{ missing }}!"

    def test_if_true(self):
        engine = TemplateEngine()
        result = engine.render("{% if show %}yes{% endif %}", {"show": True})
        assert result == "yes"

    def test_if_false(self):
        engine = TemplateEngine()
        result = engine.render("{% if show %}yes{% endif %}", {"show": False})
        assert result == ""

    def test_if_not(self):
        engine = TemplateEngine()
        result = engine.render("{% if not hidden %}visible{% endif %}", {"hidden": False})
        assert result == "visible"

    def test_for_loop(self):
        engine = TemplateEngine()
        result = engine.render(
            "{% for item in items %}{{ item }}{% endfor %}",
            {"items": ["a", "b", "c"]},
        )
        assert result == "abc"

    def test_for_loop_with_vars(self):
        engine = TemplateEngine()
        result = engine.render(
            "{% for num in nums %}{{ num }}*2={{ num }}{% endfor %}",
            {"nums": [1, 2]},
        )
        assert "1*2=1" in result
        assert "2*2=2" in result

    def test_nested_block(self):
        engine = TemplateEngine()
        result = engine.render(
            "{% if active %}{% for item in items %}{{ item }}{% endfor %}{% endif %}",
            {"active": True, "items": ["x", "y"]},
        )
        assert result == "xy"

    def test_invalid_for(self):
        engine = TemplateEngine()
        with pytest.raises(TemplateError):
            engine.render("{% for x %}{% endfor %}", {})

    def test_for_not_list(self):
        engine = TemplateEngine()
        with pytest.raises(TemplateError):
            engine.render("{% for x in notalist %}{% endfor %}", {"notalist": "string"})
