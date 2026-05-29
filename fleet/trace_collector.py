"""Distributed trace collection and correlation.

Collects distributed trace spans with parent-child relationships and
trace ID correlation. Supports span timing, annotations, and baggage.
Used for fleet request tracing, latency analysis, and dependency mapping.

Usage:
    collector = TraceCollector()
    span = collector.start_span("request", trace_id="abc-123")
    collector.finish_span(span)
    trace = collector.get_trace("abc-123")
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class TraceCollector:
    """
    Distributed trace collector with span correlation.

    :param clock: Optional clock function for testing.
    """

    def __init__(self, clock: Optional[callable] = None):
        self._clock = clock or time.time
        self._spans: Dict[str, Dict[str, Any]] = {}
        self._traces: Dict[str, List[str]] = {}  # trace_id -> span_ids

    # ------------------------------------------------------------------
    # Span management
    # ------------------------------------------------------------------

    def start_span(
        self,
        name: str,
        trace_id: str,
        parent_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> str:
        """
        Start a new span.

        :param name: Span name.
        :param trace_id: Trace identifier.
        :param parent_id: Parent span ID (optional).
        :param span_id: Span ID (auto-generated if None).
        :returns: Span ID.
        """
        sid = span_id or f"span-{len(self._spans)}-{self._clock()}"
        self._spans[sid] = {
            "name": name,
            "trace_id": trace_id,
            "parent_id": parent_id,
            "start_time": self._clock(),
            "end_time": None,
            "duration": None,
            "annotations": {},
            "baggage": {},
        }
        if trace_id not in self._traces:
            self._traces[trace_id] = []
        self._traces[trace_id].append(sid)
        return sid

    def finish_span(self, span_id: str) -> bool:
        """
        Finish a span and record duration.

        :param span_id: Span ID.
        :returns: True if span found and finished.
        """
        span = self._spans.get(span_id)
        if not span:
            return False
        if span["end_time"] is not None:
            return False
        span["end_time"] = self._clock()
        span["duration"] = span["end_time"] - span["start_time"]
        return True

    def annotate(self, span_id: str, key: str, value: Any) -> bool:
        """Add an annotation to a span."""
        span = self._spans.get(span_id)
        if not span:
            return False
        span["annotations"][key] = value
        return True

    def set_baggage(self, span_id: str, key: str, value: Any) -> bool:
        """Set baggage on a span."""
        span = self._spans.get(span_id)
        if not span:
            return False
        span["baggage"][key] = value
        return True

    # ------------------------------------------------------------------
    # Trace queries
    # ------------------------------------------------------------------

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get all spans for a trace.

        :param trace_id: Trace identifier.
        :returns: Dict with trace_id and spans list, or None.
        """
        span_ids = self._traces.get(trace_id)
        if not span_ids:
            return None
        spans = [self._spans[sid] for sid in span_ids if sid in self._spans]
        return {
            "trace_id": trace_id,
            "spans": spans,
            "span_count": len(spans),
        }

    def get_span(self, span_id: str) -> Optional[Dict[str, Any]]:
        """Get a single span by ID."""
        return self._spans.get(span_id)

    def trace_ids(self) -> List[str]:
        """List all trace IDs."""
        return list(self._traces.keys())

    def span_count(self, trace_id: str) -> int:
        """Get span count for a trace."""
        return len(self._traces.get(trace_id, []))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_trace(self, trace_id: str) -> bool:
        """Remove all spans for a trace."""
        span_ids = self._traces.pop(trace_id, [])
        for sid in span_ids:
            self._spans.pop(sid, None)
        return len(span_ids) > 0

    def clear(self) -> None:
        """Clear all traces and spans."""
        self._spans.clear()
        self._traces.clear()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        finished = sum(1 for s in self._spans.values() if s["end_time"] is not None)
        return {
            "traces": len(self._traces),
            "spans": len(self._spans),
            "finished": finished,
            "in_flight": len(self._spans) - finished,
        }

    def __repr__(self) -> str:
        return f"<TraceCollector traces={len(self._traces)} spans={len(self._spans)}>"
