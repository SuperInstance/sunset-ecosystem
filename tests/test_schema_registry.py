"""Tests for schema_registry.py — Schema versioning and compatibility.

Run: python3 -m pytest tests/test_schema_registry.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.schema_registry import SchemaRegistry


class TestSchemaRegistry:
    def test_create(self):
        reg = SchemaRegistry()
        assert reg.stats()["schemas"] == 0

    def test_register(self):
        reg = SchemaRegistry()
        v = reg.register("user", {"type": "object"})
        assert v == 1
        assert reg.latest_version("user") == 1

    def test_register_multiple_versions(self):
        reg = SchemaRegistry()
        reg.register("user", {"type": "object", "properties": {"name": {"type": "string"}}})
        v = reg.register("user", {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}})
        assert v == 2
        assert reg.latest_version("user") == 2

    def test_get(self):
        reg = SchemaRegistry()
        schema = {"type": "object"}
        reg.register("user", schema)
        assert reg.get("user") == schema
        assert reg.get("user", version=1) == schema
        assert reg.get("user", version=2) is None
        assert reg.get("missing") is None

    def test_schemas_list(self):
        reg = SchemaRegistry()
        reg.register("a", {})
        reg.register("b", {})
        assert sorted(reg.schemas()) == ["a", "b"]

    def test_versions(self):
        reg = SchemaRegistry()
        reg.register("user", {})
        reg.register("user", {})
        assert reg.versions("user") == 2
        assert reg.versions("missing") == 0

    def test_set_compatibility(self):
        reg = SchemaRegistry()
        reg.set_compatibility("user", "backward")
        reg.register("user", {"type": "object", "properties": {"name": {"type": "string"}}})
        assert reg.is_compatible("user", {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}) is True

    def test_backward_incompatible(self):
        reg = SchemaRegistry()
        reg.set_compatibility("user", "backward")
        reg.register("user", {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]})
        assert reg.is_compatible("user", {"type": "object", "properties": {}}) is False

    def test_forward_compatible(self):
        reg = SchemaRegistry()
        reg.set_compatibility("user", "forward")
        reg.register("user", {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]})
        assert reg.is_compatible("user", {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name"]}) is True

    def test_none_compatible(self):
        reg = SchemaRegistry()
        reg.set_compatibility("user", "none")
        reg.register("user", {"type": "object"})
        assert reg.is_compatible("user", {"type": "string"}) is True

    def test_no_schema_compatible(self):
        reg = SchemaRegistry()
        assert reg.is_compatible("new", {"type": "object"}) is True

    def test_stats(self):
        reg = SchemaRegistry()
        reg.register("a", {})
        reg.register("a", {})
        reg.register("b", {})
        stats = reg.stats()
        assert stats["schemas"] == 2
        assert stats["total_versions"] == 3

    def test_repr(self):
        reg = SchemaRegistry()
        assert "SchemaRegistry" in repr(reg)
