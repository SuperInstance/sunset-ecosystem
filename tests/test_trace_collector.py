"""Tests for trace_collector.py — Distributed trace collection and correlation.

Run: python3 -m pytest tests/test_trace_collector.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.trace_collector import TraceCollector


class TestTraceCollector:
    def test_create(self):
        collector = TraceCollector(clock=lambda: 0)
        assert collector.stats()["traces"] == 0

    def test_start_span(self):
        collector = TraceCollector(clock=lambda: 0)
        sid = collector.start_span("request", trace_id="trace-1")
        assert sid.startswith("span-")
        assert collector.span_count("trace-1") == 1

    def test_start_span_custom_id(self):
        collector = TraceCollector(clock=lambda: 0)
        sid = collector.start_span("request", trace_id="trace-1", span_id="my-span")
        assert sid == "my-span"
        assert collector.get_span("my-span")["name"] == "request"

    def test_finish_span(self):
        collector = TraceCollector(clock=lambda: 0)
        sid = collector.start_span("request", trace_id="trace-1")
        assert collector.finish_span(sid) is True
        span = collector.get_span(sid)
        assert span["duration"] == 0  # start and end at same clock

    def test_finish_span_missing(self):
        collector = TraceCollector(clock=lambda: 0)
        assert collector.finish_span("missing") is False

    def test_finish_span_already_finished(self):
        collector = TraceCollector(clock=lambda: 0)
        sid = collector.start_span("request", trace_id="trace-1")
        collector.finish_span(sid)
        assert collector.finish_span(sid) is False

    def test_annotate(self):
        collector = TraceCollector(clock=lambda: 0)
        sid = collector.start_span("request", trace_id="trace-1")
        assert collector.annotate(sid, "http.status", 200) is True
        assert collector.get_span(sid)["annotations"]["http.status"] == 200
        assert collector.annotate("missing", "key", "value") is False

    def test_set_baggage(self):
        collector = TraceCollector(clock=lambda: 0)
        sid = collector.start_span("request", trace_id="trace-1")
        assert collector.set_baggage(sid, "user-id", "123") is True
        assert collector.get_span(sid)["baggage"]["user-id"] == "123"
        assert collector.set_baggage("missing", "key", "value") is False

    def test_get_trace(self):
        collector = TraceCollector(clock=lambda: 0)
        sid1 = collector.start_span("request", trace_id="trace-1")
        sid2 = collector.start_span("db-query", trace_id="trace-1", parent_id=sid1)
        trace = collector.get_trace("trace-1")
        assert trace["trace_id"] == "trace-1"
        assert trace["span_count"] == 2

    def test_get_trace_missing(self):
        collector = TraceCollector(clock=lambda: 0)
        assert collector.get_trace("missing") is None

    def test_trace_ids(self):
        collector = TraceCollector(clock=lambda: 0)
        collector.start_span("a", trace_id="trace-1")
        collector.start_span("b", trace_id="trace-2")
        assert sorted(collector.trace_ids()) == ["trace-1", "trace-2"]

    def test_remove_trace(self):
        collector = TraceCollector(clock=lambda: 0)
        collector.start_span("a", trace_id="trace-1")
        assert collector.remove_trace("trace-1") is True
        assert collector.get_trace("trace-1") is None
        assert collector.remove_trace("missing") is False

    def test_clear(self):
        collector = TraceCollector(clock=lambda: 0)
        collector.start_span("a", trace_id="trace-1")
        collector.clear()
        assert collector.stats()["traces"] == 0

    def test_stats(self):
        collector = TraceCollector(clock=lambda: 0)
        sid = collector.start_span("a", trace_id="trace-1")
        collector.finish_span(sid)
        collector.start_span("b", trace_id="trace-2")
        stats = collector.stats()
        assert stats["traces"] == 2
        assert stats["spans"] == 2
        assert stats["finished"] == 1
        assert stats["in_flight"] == 1

    def test_repr(self):
        collector = TraceCollector()
        assert "TraceCollector" in repr(collector)
