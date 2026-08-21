from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Span:
    """A trace span."""

    trace_id: str
    span_id: str
    parent_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)

    def finish(self, tags: Optional[Dict[str, Any]] = None):
        """Finish the span."""
        self.end_time = time.time()
        if tags:
            self.tags.update(tags)

    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def log(self, message: str, **kwargs):
        """Add a log entry."""
        self.logs.append(
            {
                "timestamp": time.time(),
                "message": message,
                **kwargs,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms(),
            "tags": self.tags,
            "logs": self.logs,
        }


class Tracer:
    """
    Distributed tracing for fleet request flows.

    Traces requests across multiple services and operations.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self._spans: Dict[str, Span] = {}
        self._traces: Dict[str, List[str]] = {}  # trace_id -> [span_id]

    def start_span(
        self, name: str, trace_id: Optional[str] = None, parent_id: Optional[str] = None
    ) -> Span:
        """Start a new span."""
        import uuid

        span_id = str(uuid.uuid4())[:8]
        trace_id = trace_id or str(uuid.uuid4())[:8]

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            name=name,
            start_time=time.time(),
        )
        self._spans[span_id] = span
        self._traces.setdefault(trace_id, []).append(span_id)
        return span

    def finish_span(self, span_id: str, tags: Optional[Dict[str, Any]] = None):
        """Finish a span."""
        if span_id in self._spans:
            self._spans[span_id].finish(tags)

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        span_ids = self._traces.get(trace_id, [])
        return [self._spans[sid] for sid in span_ids if sid in self._spans]

    def get_span(self, span_id: str) -> Optional[Span]:
        """Get a specific span."""
        return self._spans.get(span_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        completed = [s for s in self._spans.values() if s.end_time is not None]
        durations = [s.duration_ms() for s in completed]
        return {
            "total_spans": len(self._spans),
            "completed_spans": len(completed),
            "active_spans": len(self._spans) - len(completed),
            "traces": len(self._traces),
            "avg_duration_ms": np.mean(durations) if durations else 0.0,
            "max_duration_ms": max(durations) if durations else 0.0,
        }

    def export_json(self) -> str:
        """Export all traces as JSON."""
        return json.dumps(
            {
                "node": self.fleet_node_id,
                "traces": {
                    trace_id: [self._spans[sid].to_dict() for sid in span_ids]
                    for trace_id, span_ids in self._traces.items()
                },
            },
            indent=2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
        }
