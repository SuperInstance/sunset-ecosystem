"""Tests for exception_tracker.py — Centralized exception tracking.

Run: python3 -m pytest tests/test_exception_tracker.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.exception_tracker import ExceptionTracker


class TestExceptionTracker:
    def test_create(self):
        tracker = ExceptionTracker()
        assert tracker.count() == 0

    def test_record_exception(self):
        tracker = ExceptionTracker()
        try:
            raise ValueError("test error")
        except Exception:
            tracker.record("test_service")
        assert tracker.count() == 1

    def test_record_with_context(self):
        tracker = ExceptionTracker()
        tracker.record("svc", exc_type="ValueError", exc_message="boom", context={"user_id": "abc"})
        entry = tracker.get(0)
        assert entry["service"] == "svc"
        assert entry["context"]["user_id"] == "abc"

    def test_capacity_eviction(self):
        tracker = ExceptionTracker(capacity=2)
        tracker.record("svc", exc_type="E", exc_message="1")
        tracker.record("svc", exc_type="E", exc_message="2")
        tracker.record("svc", exc_type="E", exc_message="3")
        assert tracker.count() == 2

    def test_filter_by_service(self):
        tracker = ExceptionTracker()
        tracker.record("svc_a", exc_type="E", exc_message="a")
        tracker.record("svc_b", exc_type="E", exc_message="b")
        results = tracker.filter(service="svc_a")
        assert len(results) == 1

    def test_filter_by_type(self):
        tracker = ExceptionTracker()
        try:
            raise ValueError("x")
        except Exception:
            tracker.record("svc")
        try:
            raise RuntimeError("y")
        except Exception:
            tracker.record("svc")
        results = tracker.filter(exc_type="ValueError")
        assert len(results) == 1

    def test_recent(self):
        tracker = ExceptionTracker()
        tracker.record("svc", exc_type="E", exc_message="1")
        tracker.record("svc", exc_type="E", exc_message="2")
        recent = tracker.recent(limit=1)
        assert len(recent) == 1

    def test_clear(self):
        tracker = ExceptionTracker()
        tracker.record("svc")
        tracker.clear()
        assert tracker.count() == 0

    def test_repr(self):
        tracker = ExceptionTracker()
        assert "ExceptionTracker" in repr(tracker)
