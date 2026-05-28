"""log_aggregator.py — Centralized log collection and querying.

Provides:
1. Structured log ingestion
2. Full-text search on logs
3. Log level filtering
4. Time-range queries
5. Log summarization (counts by level/source)

Usage:
    logs = LogAggregator()
    logs.ingest({"level": "ERROR", "message": "Connection failed", "source": "agent-1"})
    errors = logs.query(level="ERROR", since_minutes=5)
"""
from __future__ import annotations

__all__ = [
    "LogAggregator",
    "LogEntry",
]

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """A structured log entry."""
    timestamp: float
    level: str
    message: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class LogAggregator:
    """Centralized log collection and querying."""

    def __init__(self, max_entries: int = 10_000) -> None:
        self._max_entries = max_entries
        self._entries: list[LogEntry] = []
        self._sources: set[str] = set()

    def ingest(self, data: dict[str, Any]) -> LogEntry:
        """Ingest a log entry."""
        entry = LogEntry(
            timestamp=data.get("timestamp", time.time()),
            level=data.get("level", "INFO").upper(),
            message=data.get("message", ""),
            source=data.get("source", "unknown"),
            metadata=data.get("metadata", {}),
        )
        self._entries.append(entry)
        self._sources.add(entry.source)

        # Evict oldest if over limit
        if len(self._entries) > self._max_entries:
            removed = self._entries[:len(self._entries) - self._max_entries]
            self._entries = self._entries[-self._max_entries:]
            # Update sources
            for r in removed:
                if not any(e.source == r.source for e in self._entries):
                    self._sources.discard(r.source)

        return entry

    def query(
        self,
        level: str | None = None,
        source: str | None = None,
        message_contains: str | None = None,
        since_minutes: float | None = None,
        limit: int | None = None,
    ) -> list[LogEntry]:
        """Query logs with filters."""
        now = time.time()
        result = self._entries

        if level:
            result = [e for e in result if e.level == level.upper()]
        if source:
            result = [e for e in result if e.source == source]
        if message_contains:
            result = [e for e in result if message_contains in e.message]
        if since_minutes is not None:
            cutoff = now - since_minutes * 60
            result = [e for e in result if e.timestamp >= cutoff]
        if limit:
            result = result[-limit:]

        return result

    def levels(self) -> dict[str, int]:
        """Count entries per log level."""
        counts: dict[str, int] = {}
        for e in self._entries:
            counts[e.level] = counts.get(e.level, 0) + 1
        return counts

    def sources(self) -> list[str]:
        """List all unique sources."""
        return sorted(self._sources)

    def count(self) -> int:
        """Total number of stored entries."""
        return len(self._entries)

    def latest(self, source: str | None = None) -> LogEntry | None:
        """Get the most recent entry."""
        entries = self._entries if source is None else [e for e in self._entries if e.source == source]
        return entries[-1] if entries else None

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        self._sources.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "total_entries": len(self._entries),
            "sources": len(self._sources),
            "levels": self.levels(),
        }

    def __repr__(self) -> str:
        return f"LogAggregator(entries={len(self._entries)}, sources={len(self._sources)})"
