"""Tests for conflict_resolver.py — CRDTs for distributed state.

Run: python3 -m pytest tests/test_conflict_resolver.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.conflict_resolver import (
    LWWRegister,
    GCounter,
    PNCounter,
    ORSet,
)


class TestLWWRegister:
    def test_set_and_get(self):
        reg = LWWRegister(node_id="n1")
        reg.set(42)
        assert reg.value() == 42

    def test_merge_takes_newer(self):
        reg1 = LWWRegister(node_id="n1")
        reg2 = LWWRegister(node_id="n2")
        reg1.set(1)
        time.sleep(0.01)
        reg2.set(2)
        reg1.merge(reg2)
        assert reg1.value() == 2

    def test_merge_ignores_older(self):
        reg1 = LWWRegister(node_id="n1")
        reg2 = LWWRegister(node_id="n2")
        reg1.set(2)
        time.sleep(0.01)
        reg2.set(1)
        reg1.merge(reg2)
        assert reg1.value() == 1


class TestGCounter:
    def test_increment(self):
        c = GCounter(node_id="n1")
        c.increment()
        c.increment(2)
        assert c.value() == 3

    def test_merge(self):
        c1 = GCounter(node_id="n1")
        c2 = GCounter(node_id="n2")
        c1.increment(5)
        c2.increment(3)
        c1.merge(c2)
        assert c1.value() == 8

    def test_merge_idempotent(self):
        c = GCounter(node_id="n1")
        c.increment(5)
        c.merge(c)
        assert c.value() == 5

    def test_state(self):
        c = GCounter(node_id="n1")
        c.increment(3)
        assert c.state()["n1"] == 3


class TestPNCounter:
    def test_increment_decrement(self):
        c = PNCounter(node_id="n1")
        c.increment(5)
        c.decrement(2)
        assert c.value() == 3

    def test_merge(self):
        c1 = PNCounter(node_id="n1")
        c2 = PNCounter(node_id="n2")
        c1.increment(5)
        c2.decrement(3)
        c1.merge(c2)
        assert c1.value() == 2

    def test_state(self):
        c = PNCounter(node_id="n1")
        c.increment(3)
        c.decrement(1)
        p, n = c.state()
        assert p["n1"] == 3
        assert n["n1"] == 1


class TestORSet:
    def test_add_and_contains(self):
        s = ORSet(node_id="n1")
        s.add("a")
        assert s.contains("a")
        assert not s.contains("b")

    def test_remove(self):
        s = ORSet(node_id="n1")
        s.add("a")
        s.remove("a")
        assert not s.contains("a")

    def test_merge(self):
        s1 = ORSet(node_id="n1")
        s2 = ORSet(node_id="n2")
        s1.add("a")
        s2.add("b")
        s1.merge(s2)
        assert s1.contains("a")
        assert s1.contains("b")

    def test_merge_add_wins(self):
        s1 = ORSet(node_id="n1")
        s2 = ORSet(node_id="n2")
        s1.add("x")
        s2.add("x")
        s1.remove("x")
        s1.merge(s2)
        # s2 still has x
        assert s1.contains("x")

    def test_len(self):
        s = ORSet(node_id="n1")
        s.add("a")
        s.add("b")
        assert len(s) == 2
