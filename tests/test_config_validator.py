"""Tests for config_validator.py — Configuration file validation.

Run: python3 -m pytest tests/test_config_validator.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.config_validator import ConfigValidator


class TestConfigValidator:
    def test_create(self):
        validator = ConfigValidator()
        assert validator.stats()["schemas"] == 0

    def test_add_schema(self):
        validator = ConfigValidator()
        validator.add_schema("service", {
            "name": {"required": True, "type": str},
        })
        assert validator.has_schema("service") is True

    def test_validate_required(self):
        validator = ConfigValidator()
        validator.add_schema("service", {
            "name": {"required": True},
        })
        ok, errors = validator.validate("service", {})
        assert ok is False
        assert "required" in errors[0]

    def test_validate_type(self):
        validator = ConfigValidator()
        validator.add_schema("service", {
            "port": {"required": True, "type": int},
        })
        ok, errors = validator.validate("service", {"port": "8080"})
        assert ok is False
        assert "expected int" in errors[0]

    def test_validate_range(self):
        validator = ConfigValidator()
        validator.add_schema("service", {
            "port": {"required": True, "type": int, "min": 1, "max": 65535},
        })
        ok, errors = validator.validate("service", {"port": 0})
        assert ok is False
        assert "must be >= 1" in errors[0]
        ok, errors = validator.validate("service", {"port": 70000})
        assert ok is False
        assert "must be <= 65535" in errors[0]

    def test_validate_regex(self):
        validator = ConfigValidator()
        validator.add_schema("service", {
            "email": {"regex": r"^\S+@\S+\.\S+$"},
        })
        ok, errors = validator.validate("service", {"email": "bad"})
        assert ok is False
        ok, errors = validator.validate("service", {"email": "a@b.com"})
        assert ok is True

    def test_validate_custom(self):
        validator = ConfigValidator()
        validator.add_schema("service", {
            "code": {"custom": lambda v: "bad" if v != "OK" else None},
        })
        ok, errors = validator.validate("service", {"code": "FAIL"})
        assert ok is False
        assert "bad" in errors[0]

    def test_validate_extra_fields(self):
        validator = ConfigValidator()
        validator.add_schema("service", {
            "name": {"required": True},
        })
        ok, errors = validator.validate("service", {"name": "test", "extra": "bad"})
        assert ok is False
        assert "unknown field" in errors[0]

    def test_validate_unknown_schema(self):
        validator = ConfigValidator()
        ok, errors = validator.validate("missing", {})
        assert ok is False
        assert "Unknown schema" in errors[0]

    def test_valid(self):
        validator = ConfigValidator()
        validator.add_schema("service", {
            "name": {"required": True, "type": str},
            "port": {"required": True, "type": int, "min": 1, "max": 65535},
        })
        ok, errors = validator.validate("service", {"name": "api", "port": 8080})
        assert ok is True
        assert errors == []

    def test_schemas(self):
        validator = ConfigValidator()
        validator.add_schema("a", {})
        validator.add_schema("b", {})
        assert sorted(validator.schemas()) == ["a", "b"]

    def test_repr(self):
        validator = ConfigValidator()
        assert "ConfigValidator" in repr(validator)
