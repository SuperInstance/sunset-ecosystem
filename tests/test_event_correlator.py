"""Tests for event_correlator.py — Pattern matching across event streams.

Run: python3 -m pytest tests/test_event_correlator.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.event_correlator import EventCorrelator


class TestEventCorrelator:
    def test_create(self):
        corr = EventCorrelator()
        assert corr.stats()["rules"] == 0

    def test_add_rule(self):
        corr = EventCorrelator()
        corr.add_rule("alert", pattern=["A", "B"], window_sec=60)
        assert corr.stats()["rules"] == 1

    def test_match_sequence(self):
        corr = EventCorrelator()
        corr.add_rule("alert", pattern=["A", "B"], window_sec=60)
        corr.add_event("A", {"source": "node-1"})
        corr.add_event("B", {"source": "node-1"})
        matches = corr.matches()
        assert len(matches) == 1
        assert matches[0]["rule"] == "alert"

    def test_no_match_wrong_order(self):
        corr = EventCorrelator()
        corr.add_rule("alert", pattern=["A", "B"], window_sec=60)
        corr.add_event("B", {"source": "node-1"})
        corr.add_event("A", {"source": "node-1"})
        assert len(corr.matches()) == 0

    def test_no_match_window(self):
        corr = EventCorrelator(clock=lambda: 0)
        corr.add_rule("alert", pattern=["A", "B"], window_sec=10)
        corr.add_event("A", {"source": "node-1"})
        # Move clock past window
        corr._clock = lambda: 100
        corr.add_event("B", {"source": "node-1"})
        assert len(corr.matches()) == 0

    def test_match_fields(self):
        corr = EventCorrelator()
        corr.add_rule("alert", pattern=["A", "B"], window_sec=60, match_fields=["source"])
        corr.add_event("A", {"source": "node-1"})
        corr.add_event("B", {"source": "node-2"})  # Different source
        assert len(corr.matches()) == 0
        corr.add_event("B", {"source": "node-1"})  # Same source
        assert len(corr.matches()) == 1

    def test_clear(self):
        corr = EventCorrelator()
        corr.add_rule("alert", pattern=["A", "B"], window_sec=60)
        corr.add_event("A", {"source": "node-1"})
        corr.add_event("B", {"source": "node-1"})
        corr.clear()
        assert corr.stats()["events"] == 0
        assert corr.stats()["matches"] == 0

    def test_recent_matches(self):
        corr = EventCorrelator()
        corr.add_rule("alert", pattern=["A", "B"], window_sec=60)
        corr.add_event("A", {"source": "node-1"})
        corr.add_event("B", {"source": "node-1"})
        assert len(corr.recent_matches(limit=1)) == 1

    def test_repr(self):
        corr = EventCorrelator()
        assert "EventCorrelator" in repr(corr)
