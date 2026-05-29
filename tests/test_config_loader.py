"""Tests for config_loader.py — Configuration loader with environment override and validation.

Run: python3 -m pytest tests/test_config_loader.py -v --tb=short
"""
from __future__ import annotations

import os
import pytest

from fleet.config_loader import ConfigLoader


class TestConfigLoader:
    def test_create(self):
        loader = ConfigLoader()
        assert loader.stats()["keys"] == 0

    def test_create_with_defaults(self):
        loader = ConfigLoader(defaults={"db": {"host": "localhost"}})
        assert loader.get("db.host") == "localhost"

    def test_get(self):
        loader = ConfigLoader(defaults={"db": {"host": "localhost", "port": 5432}})
        assert loader.get("db.host") == "localhost"
        assert loader.get("db.port") == 5432
        assert loader.get("db.missing") is None
        assert loader.get("db.missing", "default") == "default"

    def test_get_nested(self):
        loader = ConfigLoader(defaults={"a": {"b": {"c": "deep"}}})
        assert loader.get("a.b.c") == "deep"

    def test_set(self):
        loader = ConfigLoader()
        loader.set("db.host", "prod-db")
        assert loader.get("db.host") == "prod-db"

    def test_set_nested(self):
        loader = ConfigLoader()
        loader.set("a.b.c", "value")
        assert loader.get("a.b.c") == "value"

    def test_load(self):
        loader = ConfigLoader(defaults={"a": 1, "b": 2})
        config = loader.load()
        assert config == {"a": 1, "b": 2}

    def test_override_from_env(self, monkeypatch):
        monkeypatch.setenv("FLEET_DB__HOST", "env-db")
        monkeypatch.setenv("FLEET_DEBUG", "true")
        monkeypatch.setenv("FLEET_PORT", "8080")
        loader = ConfigLoader()
        count = loader.override_from_env()
        assert count == 3
        assert loader.get("db.host") == "env-db"
        assert loader.get("debug") is True
        assert loader.get("port") == 8080

    def test_parse_value_bool(self):
        loader = ConfigLoader()
        assert loader._parse_value("true") is True
        assert loader._parse_value("false") is False
        assert loader._parse_value("yes") is True
        assert loader._parse_value("no") is False
        assert loader._parse_value("1") is True
        assert loader._parse_value("0") is False

    def test_parse_value_int(self):
        loader = ConfigLoader()
        assert loader._parse_value("42") == 42

    def test_parse_value_float(self):
        loader = ConfigLoader()
        assert loader._parse_value("3.14") == 3.14

    def test_parse_value_string(self):
        loader = ConfigLoader()
        assert loader._parse_value("hello") == "hello"

    def test_validator(self):
        loader = ConfigLoader()
        loader.set("port", 8080)
        loader.add_validator("port", lambda v: isinstance(v, int) and v > 0)
        assert loader.validate() == []
        loader.set("port", -1)
        assert loader.validate() == ["port"]

    def test_validator_missing_key(self):
        loader = ConfigLoader()
        loader.add_validator("missing", lambda v: v is not None)
        assert loader.validate() == []  # None values are skipped

    def test_deep_update(self):
        loader = ConfigLoader(defaults={"a": {"b": 1}})
        loader._deep_update(loader._config, {"a": {"c": 2}, "d": 3})
        assert loader.get("a.b") == 1
        assert loader.get("a.c") == 2
        assert loader.get("d") == 3

    def test_stats(self):
        loader = ConfigLoader(defaults={"a": {"b": 1, "c": 2}})
        stats = loader.stats()
        assert stats["keys"] == 3
        assert stats["validators"] == 0

    def test_repr(self):
        loader = ConfigLoader()
        assert "ConfigLoader" in repr(loader)
