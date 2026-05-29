"""Tests for data_validator.py — Data validation with schema and type checking.

Run: python3 -m pytest tests/test_data_validator.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.data_validator import DataValidator


class TestDataValidator:
    def test_create(self):
        validator = DataValidator()
        assert validator.stats()["schemas"] == 0

    def test_add_schema(self):
        validator = DataValidator()
        assert validator.add_schema("user", {"name": {"type": str, "required": True}}) is True
        assert "user" in validator.schemas()

    def test_add_schema_duplicate(self):
        validator = DataValidator()
        validator.add_schema("user", {})
        assert validator.add_schema("user", {}) is False

    def test_remove_schema(self):
        validator = DataValidator()
        validator.add_schema("user", {})
        assert validator.remove_schema("user") is True
        assert validator.remove_schema("missing") is False

    def test_validate_required(self):
        validator = DataValidator()
        validator.add_schema("user", {"name": {"type": str, "required": True}})
        errors = validator.validate("user", {})
        assert "Field 'name' is required" in errors

    def test_validate_type(self):
        validator = DataValidator()
        validator.add_schema("user", {"age": {"type": int}})
        errors = validator.validate("user", {"age": "30"})
        assert "Field 'age' must be int" in errors

    def test_validate_type_correct(self):
        validator = DataValidator()
        validator.add_schema("user", {"age": {"type": int}})
        errors = validator.validate("user", {"age": 30})
        assert errors == []

    def test_validate_min_max(self):
        validator = DataValidator()
        validator.add_schema("user", {"age": {"type": int, "min": 0, "max": 150}})
        errors = validator.validate("user", {"age": -1})
        assert "Field 'age' must be >= 0" in errors
        errors = validator.validate("user", {"age": 200})
        assert "Field 'age' must be <= 150" in errors

    def test_validate_length(self):
        validator = DataValidator()
        validator.add_schema("user", {"name": {"type": str, "min_length": 2, "max_length": 10}})
        errors = validator.validate("user", {"name": "A"})
        assert "Field 'name' length must be >= 2" in errors
        errors = validator.validate("user", {"name": "A" * 20})
        assert "Field 'name' length must be <= 10" in errors

    def test_validate_custom_validator(self):
        validator = DataValidator()
        validator.add_schema("user", {"email": {"type": str, "validator": lambda v: "@" in v}})
        errors = validator.validate("user", {"email": "invalid"})
        assert "Field 'email' failed custom validation" in errors
        errors = validator.validate("user", {"email": "valid@example.com"})
        assert errors == []

    def test_is_valid(self):
        validator = DataValidator()
        validator.add_schema("user", {"name": {"type": str, "required": True}})
        assert validator.is_valid("user", {"name": "Alice"}) is True
        assert validator.is_valid("user", {}) is False

    def test_validate_missing_schema(self):
        validator = DataValidator()
        errors = validator.validate("missing", {})
        assert "Schema 'missing' not found" in errors

    def test_optional_field(self):
        validator = DataValidator()
        validator.add_schema("user", {"name": {"type": str, "required": False}})
        errors = validator.validate("user", {})
        assert errors == []

    def test_get_schema(self):
        validator = DataValidator()
        schema = {"name": {"type": str}}
        validator.add_schema("user", schema)
        assert validator.get_schema("user") == schema
        assert validator.get_schema("missing") is None

    def test_stats(self):
        validator = DataValidator()
        validator.add_schema("a", {})
        validator.add_schema("b", {})
        assert validator.stats()["schemas"] == 2

    def test_repr(self):
        validator = DataValidator()
        assert "DataValidator" in repr(validator)
