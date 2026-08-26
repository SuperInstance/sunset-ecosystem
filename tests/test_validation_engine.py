"""Tests for validation_engine.py — Schema validation with rules.

Run: python3 -m pytest tests/test_validation_engine.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.validation_engine import ValidationEngine


class TestValidationEngine:
    def test_create(self):
        engine = ValidationEngine()
        assert engine.stats()["rules"] == 0

    def test_required(self):
        engine = ValidationEngine()
        engine.add_rule("name", required=True)
        ok, errors = engine.validate({})
        assert ok is False
        assert "required" in errors[0]

    def test_type_check(self):
        engine = ValidationEngine()
        engine.add_rule("age", type=int)
        ok, errors = engine.validate({"age": "30"})
        assert ok is False
        assert "expected int" in errors[0]

    def test_min_max(self):
        engine = ValidationEngine()
        engine.add_rule("age", min=0, max=120)
        ok, errors = engine.validate({"age": -1})
        assert ok is False
        assert ">= 0" in errors[0]
        ok, errors = engine.validate({"age": 200})
        assert ok is False
        assert "<= 120" in errors[0]

    def test_length(self):
        engine = ValidationEngine()
        engine.add_rule("name", min_len=2, max_len=5)
        ok, errors = engine.validate({"name": "A"})
        assert ok is False
        assert "length" in errors[0]
        ok, errors = engine.validate({"name": "ABCDEF"})
        assert ok is False

    def test_regex(self):
        engine = ValidationEngine()
        engine.add_rule("email", regex=r"^\S+@\S+\.\S+$")
        ok, errors = engine.validate({"email": "bad"})
        assert ok is False
        ok, errors = engine.validate({"email": "a@b.com"})
        assert ok is True

    def test_custom_validator(self):
        engine = ValidationEngine()
        engine.add_rule("code", custom=lambda v: "bad code" if v != "OK" else None)
        ok, errors = engine.validate({"code": "FAIL"})
        assert ok is False
        assert "bad code" in errors[0]

    def test_valid(self):
        engine = ValidationEngine()
        engine.add_rule("name", required=True, type=str, min_len=1)
        engine.add_rule("age", required=True, type=int, min=0, max=120)
        ok, errors = engine.validate({"name": "Alice", "age": 30})
        assert ok is True
        assert errors == []

    def test_repr(self):
        engine = ValidationEngine()
        assert "ValidationEngine" in repr(engine)
