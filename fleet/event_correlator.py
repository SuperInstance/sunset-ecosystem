"""Event correlator for pattern matching across event streams.

Correlates events from multiple sources by pattern, window, and
sequence matching. Used for fleet anomaly detection, causal tracing,
and composite alerting.

Usage:
    corr = EventCorrelator()
    corr.add_rule("alert", pattern=["A", "B"], window_sec=60)
    corr.add_event("A", {"source": "node-1"})
    corr.add_event("B", {"source": "node-1"})
    matches = corr.matches()
    assert len(matches) == 1
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional


class EventCorrelator:
    """
    Event correlator with pattern matching.

    :param clock: Optional clock function for testing.
    """

    def __init__(self, clock: Optional[callable] = None):
        self._clock = clock or time.time
        self._rules: Dict[str, Dict[str, Any]] = {}
        self._events: List[Dict[str, Any]] = []
        self._matches: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Rule definition
    # ------------------------------------------------------------------

    def add_rule(
        self,
        name: str,
        pattern: List[str],
        window_sec: float,
        match_fields: Optional[List[str]] = None,
    ) -> None:
        """
        Add a correlation rule.

        :param name: Rule name.
        :param pattern: Ordered event type sequence to match.
        :param window_sec: Max seconds between first and last event.
        :param match_fields: Fields that must match across events.
        """
        self._rules[name] = {
            "pattern": pattern,
            "window_sec": window_sec,
            "match_fields": match_fields or [],
        }

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------

    def add_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Add an event for correlation."""
        self._events.append(
            {
                "timestamp": self._clock(),
                "type": event_type,
                "data": data,
            }
        )
        self._check_rules()

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    def _check_rules(self) -> None:
        """Check all rules against current events."""
        for rule_name, rule in self._rules.items():
            pattern = rule["pattern"]
            window = rule["window_sec"]
            match_fields = rule["match_fields"]
            # Find sequences matching pattern
            matches = self._find_sequences(pattern, window, match_fields)
            for match in matches:
                self._matches.append(
                    {
                        "rule": rule_name,
                        "events": match,
                        "timestamp": self._clock(),
                    }
                )

    def _find_sequences(
        self,
        pattern: List[str],
        window: float,
        match_fields: List[str],
    ) -> List[List[Dict[str, Any]]]:
        """Find event sequences matching a pattern."""
        # Simple implementation: find events in order with matching fields
        results: List[List[Dict[str, Any]]] = []
        # Filter events by type in order
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for e in self._events:
            by_type[e["type"]].append(e)
        # Find first event of pattern
        if not pattern:
            return results
        first_events = by_type.get(pattern[0], [])
        for first in first_events:
            sequence = [first]
            valid = True
            for i, p_type in enumerate(pattern[1:], 1):
                candidates = by_type.get(p_type, [])
                found = False
                for candidate in candidates:
                    if candidate["timestamp"] < sequence[-1]["timestamp"]:
                        continue
                    if candidate["timestamp"] - first["timestamp"] > window:
                        continue
                    # Check match fields
                    field_match = True
                    for field in match_fields:
                        if candidate["data"].get(field) != first["data"].get(field):
                            field_match = False
                            break
                    if not field_match:
                        continue
                    sequence.append(candidate)
                    found = True
                    break
                if not found:
                    valid = False
                    break
            if valid and len(sequence) == len(pattern):
                results.append(sequence)
        return results

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def matches(self) -> List[Dict[str, Any]]:
        """Get all matched sequences."""
        return list(self._matches)

    def recent_matches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent matches."""
        return self._matches[-limit:]

    def clear(self) -> None:
        """Clear all events and matches."""
        self._events.clear()
        self._matches.clear()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "rules": len(self._rules),
            "events": len(self._events),
            "matches": len(self._matches),
        }

    def __repr__(self) -> str:
        return f"<EventCorrelator rules={len(self._rules)} events={len(self._events)}>"
