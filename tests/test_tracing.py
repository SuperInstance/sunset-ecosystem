import time
import pytest
from fleet.tracing import Span, Tracer


class TestSpan:
    def test_to_dict(self):
        s = Span(
            trace_id="t1",
            span_id="s1",
            parent_id=None,
            name="test",
            start_time=0.0,
        )
        d = s.to_dict()
        assert d["trace_id"] == "t1"
        assert d["name"] == "test"

    def test_finish(self):
        s = Span(
            trace_id="t1",
            span_id="s1",
            parent_id=None,
            name="test",
            start_time=time.time(),
        )
        s.finish(tags={"key": "value"})
        assert s.end_time is not None
        assert s.tags["key"] == "value"
        assert s.duration_ms() > 0

    def test_log(self):
        s = Span(
            trace_id="t1",
            span_id="s1",
            parent_id=None,
            name="test",
            start_time=0.0,
        )
        s.log("msg", extra=1)
        assert len(s.logs) == 1
        assert s.logs[0]["message"] == "msg"


class TestTracer:
    def test_init(self):
        t = Tracer()
        assert t.fleet_node_id == "default"

    def test_start_span(self):
        t = Tracer()
        span = t.start_span("test")
        assert span.name == "test"
        assert span.span_id is not None
        assert span.trace_id is not None

    def test_start_span_with_trace_id(self):
        t = Tracer()
        span = t.start_span("test", trace_id="abc")
        assert span.trace_id == "abc"

    def test_start_span_with_parent(self):
        t = Tracer()
        parent = t.start_span("parent")
        child = t.start_span("child", parent_id=parent.span_id)
        assert child.parent_id == parent.span_id

    def test_finish_span(self):
        t = Tracer()
        span = t.start_span("test")
        t.finish_span(span.span_id, tags={"status": "ok"})
        assert span.end_time is not None
        assert span.tags["status"] == "ok"

    def test_get_trace(self):
        t = Tracer()
        span1 = t.start_span("a")
        span2 = t.start_span("b", trace_id=span1.trace_id)
        trace = t.get_trace(span1.trace_id)
        assert len(trace) == 2

    def test_get_span(self):
        t = Tracer()
        span = t.start_span("test")
        assert t.get_span(span.span_id) == span

    def test_get_stats(self):
        t = Tracer()
        t.start_span("a")
        span2 = t.start_span("b")
        t.finish_span(span2.span_id)
        stats = t.get_stats()
        assert stats["total_spans"] == 2
        assert stats["completed_spans"] == 1

    def test_export_json(self):
        t = Tracer()
        span = t.start_span("test")
        t.finish_span(span.span_id)
        j = t.export_json()
        assert "test" in j

    def test_to_dict(self):
        t = Tracer()
        t.start_span("test")
        d = t.to_dict()
        assert "stats" in d
