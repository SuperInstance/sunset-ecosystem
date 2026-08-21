"""Tests for vector_clock.py — Vector clocks for causality.

Run: python3 -m pytest tests/test_vector_clock.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.vector_clock import VectorClock


class TestVectorClock:
    def test_create(self):
        vc = VectorClock()
        assert vc.to_dict() == {}

    def test_increment(self):
        vc = VectorClock()
        vc.increment("a")
        vc.increment("a")
        assert vc["a"] == 2

    def test_merge(self):
        vc1 = VectorClock({"a": 1, "b": 2})
        vc2 = VectorClock({"b": 3, "c": 1})
        vc1.merge(vc2)
        assert vc1.to_dict() == {"a": 1, "b": 3, "c": 1}

    def test_compare_equal(self):
        vc1 = VectorClock({"a": 1, "b": 2})
        vc2 = VectorClock({"a": 1, "b": 2})
        assert vc1.compare(vc2) == "equal"

    def test_compare_greater(self):
        vc1 = VectorClock({"a": 2, "b": 2})
        vc2 = VectorClock({"a": 1, "b": 2})
        assert vc1.compare(vc2) == "greater"

    def test_compare_less(self):
        vc1 = VectorClock({"a": 1, "b": 1})
        vc2 = VectorClock({"a": 2, "b": 2})
        assert vc1.compare(vc2) == "less"

    def test_compare_concurrent(self):
        vc1 = VectorClock({"a": 2, "b": 1})
        vc2 = VectorClock({"a": 1, "b": 2})
        assert vc1.compare(vc2) == "concurrent"

    def test_happens_before(self):
        vc1 = VectorClock({"a": 1})
        vc2 = VectorClock({"a": 2})
        assert vc1.happens_before(vc2) is True
        assert vc2.happens_before(vc1) is False

    def test_is_concurrent(self):
        vc1 = VectorClock({"a": 2, "b": 1})
        vc2 = VectorClock({"a": 1, "b": 2})
        assert vc1.is_concurrent(vc2) is True

    def test_serialization(self):
        vc = VectorClock({"a": 1, "b": 2})
        data = vc.to_dict()
        restored = VectorClock.from_dict(data)
        assert restored == vc

    def test_equality(self):
        vc1 = VectorClock({"a": 1})
        vc2 = VectorClock({"a": 1})
        assert vc1 == vc2
        assert hash(vc1) == hash(vc2)

    def test_repr(self):
        vc = VectorClock({"a": 1})
        assert "VectorClock" in repr(vc)
