"""Tests for event_filter.py — Event stream filtering.

Run: python3 -m pytest tests/test_event_filter.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.event_filter import EventFilter


class TestEventFilter:
    def test_create(self):
        f = EventFilter()
        assert f.condition_count() == 0
        assert f.matches({}) is True

    def test_field_equals(self):
        f = EventFilter()
        f.add_field_equals("level", "error")
        assert f.matches({"level": "error"}) is True
        assert f.matches({"level": "info"}) is False

    def test_field_contains(self):
        f = EventFilter()
        f.add_field_contains("msg", "fail")
        assert f.matches({"msg": "connection failed"}) is True
        assert f.matches({"msg": "success"}) is False

    def test_field_exists(self):
        f = EventFilter()
        f.add_field_exists("trace_id")
        assert f.matches({"trace_id": "abc"}) is True
        assert f.matches({"other": "x"}) is False

    def test_field_greater(self):
        f = EventFilter()
        f.add_field_greater("latency", 100.0)
        assert f.matches({"latency": 150}) is True
        assert f.matches({"latency": 50}) is False
        assert f.matches({"latency": "not_a_number"}) is False

    def test_or_mode(self):
        f = EventFilter(mode="or")
        f.add_field_equals("a", 1)
        f.add_field_equals("b", 2)
        assert f.matches({"a": 1}) is True
        assert f.matches({"b": 2}) is True
        assert f.matches({"a": 1, "b": 2}) is True
        assert f.matches({"a": 99}) is False

    def test_filter_batch(self):
        f = EventFilter()
        f.add_field_equals("level", "error")
        events = [
            {"level": "error", "msg": "boom"},
            {"level": "info", "msg": "ok"},
            {"level": "error", "msg": "crash"},
        ]
        result = f.filter_batch(events)
        assert len(result) == 2
        assert all(e["level"] == "error" for e in result)

    def test_stats(self):
        f = EventFilter()
        f.add_field_equals("x", 1)
        f.matches({"x": 1})
        f.matches({"x": 1})
        f.matches({"x": 2})
        stats = f.stats()
        assert stats["conditions"] == 1
        assert stats["matches"] == 2

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            EventFilter(mode="invalid")

    def test_repr(self):
        f = EventFilter()
        f.add_field_equals("x", 1)
        assert "EventFilter" in repr(f)
