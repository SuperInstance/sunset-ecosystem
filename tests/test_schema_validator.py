"""Tests for schema_validator.py — JSON-like schema validation.

Run: python3 -m pytest tests/test_schema_validator.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.schema_validator import SchemaValidator, ValidationError


class TestSchemaValidator:
    def test_valid_data(self):
        schema = {
            "name": {"type": str, "required": True},
            "age": {"type": int, "min": 0, "max": 150},
        }
        v = SchemaValidator(schema)
        errors = v.validate({"name": "Alice", "age": 30})
        assert errors == []

    def test_missing_required(self):
        schema = {"name": {"type": str, "required": True}}
        v = SchemaValidator(schema)
        errors = v.validate({})
        assert len(errors) == 1
        assert errors[0].path == "name"
        assert "Required" in errors[0].message

    def test_wrong_type(self):
        schema = {"age": {"type": int}}
        v = SchemaValidator(schema)
        errors = v.validate({"age": "thirty"})
        assert len(errors) == 1
        assert "Expected int" in errors[0].message

    def test_min_max(self):
        schema = {"score": {"type": int, "min": 0, "max": 100}}
        v = SchemaValidator(schema)
        assert len(v.validate({"score": -1})) == 1
        assert len(v.validate({"score": 101})) == 1
        assert len(v.validate({"score": 50})) == 0

    def test_string_length(self):
        schema = {"label": {"type": str, "min_len": 2, "max_len": 10}}
        v = SchemaValidator(schema)
        assert len(v.validate({"label": "A"})) == 1
        assert len(v.validate({"label": "A" * 11})) == 1
        assert len(v.validate({"label": "Hello"})) == 0

    def test_pattern(self):
        schema = {"email": {"type": str, "pattern": r"^\S+@\S+\.\S+$"}}
        v = SchemaValidator(schema)
        assert len(v.validate({"email": "alice@example.com"})) == 0
        assert len(v.validate({"email": "not-an-email"})) == 1

    def test_list_validation(self):
        schema = {
            "tags": {"type": list, "item_type": str, "min_len": 1, "max_len": 3},
        }
        v = SchemaValidator(schema)
        assert len(v.validate({"tags": ["a", "b"]})) == 0
        assert len(v.validate({"tags": []})) == 1
        assert len(v.validate({"tags": ["a", "b", "c", "d"]})) == 1
        assert len(v.validate({"tags": [1, 2]})) == 2  # Wrong item type

    def test_nested_schema(self):
        schema = {
            "user": {
                "type": dict,
                "schema": {
                    "name": {"type": str, "required": True},
                },
            },
        }
        v = SchemaValidator(schema)
        assert len(v.validate({"user": {"name": "Alice"}})) == 0
        assert len(v.validate({"user": {}})) == 1

    def test_list_item_schema(self):
        schema = {
            "items": {
                "type": list,
                "item_schema": {
                    "id": {"type": int, "required": True},
                },
            },
        }
        v = SchemaValidator(schema)
        assert len(v.validate({"items": [{"id": 1}, {"id": 2}]})) == 0
        assert len(v.validate({"items": [{"id": "x"}]})) == 1

    def test_custom_validator(self):
        schema = {
            "even": {"type": int, "validator": lambda x: x % 2 == 0},
        }
        v = SchemaValidator(schema)
        assert len(v.validate({"even": 4})) == 0
        assert len(v.validate({"even": 3})) == 1

    def test_custom_validator_exception(self):
        schema = {
            "field": {"validator": lambda x: 1 / x},
        }
        v = SchemaValidator(schema)
        errors = v.validate({"field": 0})
        assert len(errors) == 1
        assert "Custom validator error" in errors[0].message

    def test_unknown_field(self):
        schema = {"name": {"type": str}}
        v = SchemaValidator(schema)
        errors = v.validate({"name": "Alice", "extra": 1})
        assert any(e.path == "extra" for e in errors)

    def test_is_valid(self):
        schema = {"name": {"type": str, "required": True}}
        v = SchemaValidator(schema)
        assert v.is_valid({"name": "Alice"}) is True
        assert v.is_valid({}) is False

    def test_optional_field(self):
        schema = {"name": {"type": str, "required": False}}
        v = SchemaValidator(schema)
        assert v.validate({}) == []

    def test_repr(self):
        v = SchemaValidator({"a": {"type": str}})
        assert "SchemaValidator" in repr(v)
