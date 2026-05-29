"""Event stream filtering with predicate logic.

Filters events using boolean predicates, field matchers, and composite
conditions (AND/OR/NOT). Used for fleet log filtering, alert routing,
and event bus subscribers.

Usage:
    f = EventFilter()
    f.add_condition(lambda e: e["level"] == "error")
    f.add_condition(lambda e: e["service"] == "api")
    assert f.matches({"level": "error", "service": "api"})
    assert not f.matches({"level": "info", "service": "api"})
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class EventFilter:
    """
    Composite event filter with boolean logic.

    Default mode: AND (all conditions must match).
    """

    def __init__(self, mode: str = "and"):
        if mode not in ("and", "or"):
            raise ValueError("mode must be 'and' or 'or'")
        self._mode = mode
        self._conditions: List[Callable[[Dict[str, Any]], bool]] = []
        self._match_count = 0

    # ------------------------------------------------------------------
    # Conditions
    # ------------------------------------------------------------------

    def add_condition(
        self, fn: Callable[[Dict[str, Any]], bool]
    ) -> "EventFilter":
        """Add a condition function."""
        self._conditions.append(fn)
        return self

    def add_field_equals(self, field: str, value: Any) -> "EventFilter":
        """Add a field equality condition."""
        self._conditions.append(lambda e: e.get(field) == value)
        return self

    def add_field_contains(self, field: str, value: Any) -> "EventFilter":
        """Add a field containment condition."""
        self._conditions.append(
            lambda e: value in str(e.get(field, ""))
        )
        return self

    def add_field_exists(self, field: str) -> "EventFilter":
        """Add a field existence condition."""
        self._conditions.append(lambda e: field in e)
        return self

    def add_field_greater(self, field: str, value: float) -> "EventFilter":
        """Add a numeric greater-than condition."""
        self._conditions.append(
            lambda e: isinstance(e.get(field), (int, float))
            and e[field] > value
        )
        return self

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def matches(self, event: Dict[str, Any]) -> bool:
        """Check if an event matches all/any conditions."""
        if not self._conditions:
            return True
        if self._mode == "and":
            result = all(c(event) for c in self._conditions)
        else:
            result = any(c(event) for c in self._conditions)
        if result:
            self._match_count += 1
        return result

    def filter_batch(
        self, events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter a batch of events."""
        return [e for e in events if self.matches(e)]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def condition_count(self) -> int:
        return len(self._conditions)

    def stats(self) -> Dict[str, int]:
        return {"conditions": len(self._conditions), "matches": self._match_count}

    def __repr__(self) -> str:
        return f"<EventFilter mode={self._mode} conditions={len(self._conditions)}>"
