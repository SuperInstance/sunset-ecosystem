"""Tests for request_tracer.py — Distributed request tracing.

Run: python3 -m pytest tests/test_request_tracer.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.request_tracer import RequestTracer


class TestRequestTracer:
    def test_create(self):
        tracer = RequestTracer()
        assert tracer.trace_count() == 0

    def test_start_trace(self):
        tracer = RequestTracer()
        tid, sid = tracer.start_trace()
        assert len(tid) == 16
        assert len(sid) == 16
        assert tracer.trace_count() == 1

    def test_start_span(self):
        tracer = RequestTracer()
        tid, root_sid = tracer.start_trace(name="root")
        child_sid = tracer.start_span("child", tid, root_sid)
        assert child_sid != root_sid
        trace = tracer.get_trace(tid)
        assert len(trace.spans) == 2

    def test_finish_span(self):
        tracer = RequestTracer()
        tid, sid = tracer.start_trace()
        time.sleep(0.01)
        span = tracer.finish_span(sid)
        assert span is not None
        assert span.duration_ms > 0

    def test_span_context_manager(self):
        tracer = RequestTracer()
        with tracer.span("operation") as span:
            assert span.name == "operation"
            time.sleep(0.01)
        assert span.duration_ms > 0

    def test_nested_spans(self):
        tracer = RequestTracer()
        with tracer.span("parent") as parent:
            with tracer.span("child", trace_id=parent.trace_id, parent_span_id=parent.span_id) as child:
                assert child.parent_id == parent.span_id

    def test_log(self):
        tracer = RequestTracer()
        tid, sid = tracer.start_trace()
        tracer.log(sid, "started processing")
        span = tracer.get_span(sid)
        assert len(span.logs) == 1
        assert span.logs[0]["message"] == "started processing"

    def test_tag(self):
        tracer = RequestTracer()
        tid, sid = tracer.start_trace()
        tracer.tag(sid, "user_id", "123")
        span = tracer.get_span(sid)
        assert span.tags["user_id"] == "123"

    def test_get_trace(self):
        tracer = RequestTracer()
        tid, _ = tracer.start_trace()
        trace = tracer.get_trace(tid)
        assert trace is not None
        assert trace.trace_id == tid
        assert tracer.get_trace("missing") is None

    def test_trace_ids(self):
        tracer = RequestTracer()
        tid1, _ = tracer.start_trace()
        tid2, _ = tracer.start_trace()
        assert sorted(tracer.trace_ids()) == sorted([tid1, tid2])

    def test_span_count(self):
        tracer = RequestTracer()
        tid, root = tracer.start_trace()
        tracer.start_span("a", tid, root)
        tracer.start_span("b", tid, root)
        assert tracer.span_count(tid) == 3

    def test_total_duration(self):
        tracer = RequestTracer()
        tid, root = tracer.start_trace()
        time.sleep(0.01)
        tracer.finish_span(root)
        assert tracer.total_duration_ms(tid) > 0

    def test_max_traces_eviction(self):
        tracer = RequestTracer(max_traces=2)
        tid1, _ = tracer.start_trace()
        tid2, _ = tracer.start_trace()
        tid3, _ = tracer.start_trace()
        assert tracer.trace_count() == 2
        assert tid1 not in tracer.trace_ids()  # Oldest evicted

    def test_stats(self):
        tracer = RequestTracer()
        tid, sid = tracer.start_trace()
        tracer.start_span("child", tid, sid)
        stats = tracer.stats()
        assert stats["traces"] == 1
        assert stats["active_spans"] == 2

    def test_repr(self):
        tracer = RequestTracer()
        assert "RequestTracer" in repr(tracer)