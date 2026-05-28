"""Tests for config_validator.py — Schema-based configuration validation.

Run: python3 -m pytest tests/test_config_validator.py -v --tb=short
"""
from __future__ import annotations

import pytest

from logos.config_validator import (
    Field,
    Schema,
    ValidationError,
    ValidationResult,
)


class TestFieldValidation:
    def test_type_check(self):
        f = Field(type=int)
        errors = f._validate_value("hello", "port")
        assert len(errors) == 1
        assert "expected int" in errors[0].message

    def test_min_max(self):
        f = Field(type=int, min=0, max=100)
        assert len(f._validate_value(50, "x")) == 0
        assert len(f._validate_value(-1, "x")) == 1
        assert len(f._validate_value(101, "x")) == 1

    def test_string_length(self):
        f = Field(type=str, min_length=3, max_length=10)
        assert len(f._validate_value("ab", "x")) == 1
        assert len(f._validate_value("abc", "x")) == 0
        assert len(f._validate_value("a" * 11, "x")) == 1

    def test_pattern(self):
        f = Field(type=str, pattern=r"^[a-z]+$")
        assert len(f._validate_value("hello", "x")) == 0
        assert len(f._validate_value("Hello123", "x")) == 1

    def test_choices(self):
        f = Field(type=str, choices=["a", "b", "c"])
        assert len(f._validate_value("b", "x")) == 0
        assert len(f._validate_value("d", "x")) == 1

    def test_allow_none(self):
        f = Field(type=str, required=True, allow_none=True)
        assert len(f._validate_value(None, "x")) == 0

    def test_required_none(self):
        f = Field(type=str, required=True)
        assert len(f._validate_value(None, "x")) == 1

    def test_list_item_schema(self):
        f = Field(type=list, item_schema=Field(type=int, min=0))
        errors = f._validate_value([1, 2, -1], "x")
        assert len(errors) == 1
        assert "[2]" in errors[0].path  # third item

    def test_dict_schema(self):
        f = Field(type=dict, dict_schema={"host": Field(type=str, required=True)})
        assert len(f._validate_value({"host": "localhost"}, "x")) == 0
        assert len(f._validate_value({}, "x")) == 1

    def test_custom_validator(self):
        f = Field(type=int, custom_validator=lambda v: "must be even" if v % 2 else None)
        assert len(f._validate_value(4, "x")) == 0
        assert len(f._validate_value(3, "x")) == 1


class TestSchemaValidation:
    def test_valid_config(self):
        schema = Schema({
            "name": Field(type=str, required=True, min_length=1),
            "port": Field(type=int, required=True, min=1, max=65535, default=8080),
        })
        result = schema.validate({"name": "test", "port": 3000})
        assert result.is_valid
        assert result.value["name"] == "test"
        assert result.value["port"] == 3000

    def test_missing_required(self):
        schema = Schema({
            "name": Field(type=str, required=True),
        })
        result = schema.validate({})
        assert not result.is_valid
        assert len(result.errors) == 1

    def test_default_value(self):
        schema = Schema({
            "port": Field(type=int, default=8080),
        })
        result = schema.validate({})
        assert result.is_valid
        assert result.value["port"] == 8080

    def test_unknown_field(self):
        schema = Schema({"name": Field(type=str)})
        result = schema.validate({"name": "x", "extra": 1})
        assert not result.is_valid
        assert any("unknown" in e.message for e in result.errors)

    def test_multiple_errors(self):
        schema = Schema({
            "a": Field(type=int, min=0),
            "b": Field(type=str, pattern=r"^[a-z]+$"),
        })
        result = schema.validate({"a": -5, "b": "123"})
        assert len(result.errors) == 2

    def test_validate_or_raise(self):
        schema = Schema({"name": Field(type=str, required=True)})
        with pytest.raises(ValueError):
            schema.validate_or_raise({})

    def test_nested_validation(self):
        schema = Schema({
            "server": Field(type=dict, dict_schema={
                "host": Field(type=str, required=True),
                "port": Field(type=int, required=True, min=1, max=65535),
            }),
        })
        result = schema.validate({
            "server": {"host": "localhost", "port": 8080}
        })
        assert result.is_valid

    def test_nested_missing_field(self):
        schema = Schema({
            "server": Field(type=dict, dict_schema={
                "host": Field(type=str, required=True),
            }),
        })
        result = schema.validate({"server": {}})
        assert not result.is_valid

    def test_empty_schema(self):
        schema = Schema({})
        result = schema.validate({})
        assert result.is_valid

    def test_result_str(self):
        e = ValidationError("field", "bad")
        assert str(e) == "field: bad"
