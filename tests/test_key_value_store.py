"""Tests for key_value_store.py — KV store with TTL.

Run: python3 -m pytest tests/test_key_value_store.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.key_value_store import KeyValueStore


class TestKeyValueStore:
    def test_create(self):
        kv = KeyValueStore()
        assert kv.size() == 0

    def test_set_and_get(self):
        kv = KeyValueStore()
        kv.set("x", 42)
        assert kv.get("x") == 42

    def test_delete(self):
        kv = KeyValueStore()
        kv.set("x", 42)
        assert kv.delete("x") is True
        assert kv.get("x") is None
        assert kv.delete("missing") is False

    def test_has(self):
        kv = KeyValueStore()
        kv.set("x", 42)
        assert kv.has("x") is True
        assert kv.has("missing") is False

    def test_ttl_expiration(self):
        kv = KeyValueStore()
        kv.set("x", 42, ttl_sec=0.05)
        assert kv.get("x") == 42
        time.sleep(0.06)
        assert kv.get("x") is None

    def test_ttl_query(self):
        kv = KeyValueStore()
        kv.set("x", 42, ttl_sec=10.0)
        ttl = kv.ttl("x")
        assert ttl is not None
        assert ttl > 9.0

    def test_expire(self):
        kv = KeyValueStore()
        kv.set("x", 42)
        assert kv.expire("x", 0.05) is True
        time.sleep(0.06)
        assert kv.get("x") is None

    def test_namespace(self):
        kv = KeyValueStore(namespace="ns")
        kv.set("x", 42)
        assert kv.get("x") == 42
        assert kv.keys() == ["x"]

    def test_batch_set(self):
        kv = KeyValueStore()
        kv.batch_set({"a": 1, "b": 2})
        assert kv.get("a") == 1
        assert kv.get("b") == 2

    def test_batch_delete(self):
        kv = KeyValueStore()
        kv.set("a", 1)
        kv.set("b", 2)
        assert kv.batch_delete(["a", "b", "c"]) == 2

    def test_keys(self):
        kv = KeyValueStore()
        kv.set("b", 2)
        kv.set("a", 1)
        assert kv.keys() == ["a", "b"]

    def test_clear(self):
        kv = KeyValueStore()
        kv.set("x", 1)
        kv.clear()
        assert kv.size() == 0

    def test_repr(self):
        kv = KeyValueStore()
        kv.set("x", 1)
        assert "KeyValueStore" in repr(kv)
