"""request_tracer.py — Distributed request tracing.

Provides:
1. Trace ID generation and propagation
2. Span creation with timing
3. Parent-child span relationships
4. Trace export/filtering
5. Performance metrics per span

Usage:
    tracer = RequestTracer()
    with tracer.span("breed_request", trace_id="abc123") as span:
        do_breeding()
        # span auto-closes with duration
    trace = tracer.get_trace("abc123")
"""
from __future__ import annotations

__all__ = [
    "RequestTracer",
    "Span",
    "Trace",
]

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """A single span in a trace."""
    span_id: str
    name: str
    trace_id: str
    parent_id: str | None
    start_time: float
    end_time: float | None = None
    duration_ms: float = 0.0
    tags: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Trace:
    """A complete trace with all spans."""
    trace_id: str
    spans: list[Span] = field(default_factory=list)
    root_span: Span | None = None


class RequestTracer:
    """Distributed request tracing with spans."""

    def __init__(self, max_traces: int = 1000) -> None:
        self._max_traces = max_traces
        self._traces: dict[str, Trace] = {}
        self._active_spans: dict[str, Span] = {}  # span_id -> Span

    def start_trace(self, trace_id: str | None = None, name: str = "root") -> tuple[str, str]:
        """Start a new trace. Returns (trace_id, span_id)."""
        tid = trace_id or uuid.uuid4().hex[:16]
        sid = uuid.uuid4().hex[:16]
        span = Span(
            span_id=sid,
            name=name,
            trace_id=tid,
            parent_id=None,
            start_time=time.time(),
        )
        trace = Trace(trace_id=tid, spans=[span], root_span=span)
        self._traces[tid] = trace
        self._active_spans[sid] = span
        self._evict_if_needed()
        return tid, sid

    def start_span(
        self,
        name: str,
        trace_id: str,
        parent_span_id: str | None = None,
    ) -> str:
        """Start a child span. Returns span_id."""
        sid = uuid.uuid4().hex[:16]
        span = Span(
            span_id=sid,
            name=name,
            trace_id=trace_id,
            parent_id=parent_span_id,
            start_time=time.time(),
        )
        trace = self._traces.get(trace_id)
        if trace:
            trace.spans.append(span)
        self._active_spans[sid] = span
        return sid

    def finish_span(self, span_id: str) -> Span | None:
        """Finish a span and record duration."""
        span = self._active_spans.pop(span_id, None)
        if span:
            span.end_time = time.time()
            span.duration_ms = (span.end_time - span.start_time) * 1000
            logger.debug(f"Span {span.name} finished in {span.duration_ms:.2f}ms")
        return span

    @contextmanager
    def span(
        self,
        name: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> Generator[Span, None, None]:
        """Context manager for a span."""
        if trace_id is None:
            trace_id, parent_span_id = self.start_trace(name=name)

        sid = self.start_span(name, trace_id, parent_span_id)
        span = self._active_spans[sid]
        try:
            yield span
        finally:
            self.finish_span(sid)

    def log(self, span_id: str, message: str, **kwargs: Any) -> None:
        """Add a log entry to a span."""
        span = self._active_spans.get(span_id)
        if span:
            span.logs.append({
                "timestamp": time.time(),
                "message": message,
                **kwargs,
            })

    def tag(self, span_id: str, key: str, value: Any) -> None:
        """Add a tag to a span."""
        span = self._active_spans.get(span_id)
        if span:
            span.tags[key] = value

    def get_trace(self, trace_id: str) -> Trace | None:
        """Get a complete trace."""
        return self._traces.get(trace_id)

    def get_span(self, span_id: str) -> Span | None:
        """Get a span by ID."""
        return self._active_spans.get(span_id)

    def trace_ids(self) -> list[str]:
        """List all trace IDs."""
        return list(self._traces.keys())

    def trace_count(self) -> int:
        """Count stored traces."""
        return len(self._traces)

    def span_count(self, trace_id: str) -> int:
        """Count spans in a trace."""
        trace = self._traces.get(trace_id)
        return len(trace.spans) if trace else 0

    def total_duration_ms(self, trace_id: str) -> float:
        """Total duration of all spans in a trace."""
        trace = self._traces.get(trace_id)
        if not trace:
            return 0.0
        return sum(s.duration_ms for s in trace.spans if s.end_time is not None)

    def _evict_if_needed(self) -> None:
        """Evict oldest traces if over limit."""
        if len(self._traces) > self._max_traces:
            # Sort by first span start time, evict oldest
            sorted_traces = sorted(
                self._traces.items(),
                key=lambda x: x[1].spans[0].start_time if x[1].spans else 0,
            )
            to_remove = len(self._traces) - self._max_traces
            for tid, _ in sorted_traces[:to_remove]:
                del self._traces[tid]

    def stats(self) -> dict[str, Any]:
        return {
            "traces": len(self._traces),
            "active_spans": len(self._active_spans),
            "max_traces": self._max_traces,
        }

    def __repr__(self) -> str:
        return f"RequestTracer(traces={len(self._traces)}, active_spans={len(self._active_spans)})"
