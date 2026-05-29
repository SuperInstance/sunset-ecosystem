"""Fleet message router with pattern matching.

Routes messages to handlers based on routing keys with wildcard support.
Uses AMQP-style topic matching (*.words, #.paths). Used for fleet-wide
message routing between nodes and services.

Usage:
    router = FleetRouter()
    router.add_route("fleet.breed.*", handler)
    router.route("fleet.breed.alpha", {"data": 1})
    # handler called with {"data": 1}
"""
from __future__ import annotations

import fnmatch
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class FleetRouter:
    """
    Message router with pattern-based routing.

    Supports exact match, * single-word wildcard, and # multi-word wildcard.
    Multiple handlers can match the same routing key.
    """

    def __init__(self):
        self._handlers: Dict[str, List[Callable[[Any], None]]] = {}

    # ------------------------------------------------------------------
    # Route registration
    # ------------------------------------------------------------------

    def add_route(self, pattern: str, handler: Callable[[Any], None]) -> None:
        """Register a handler for a routing pattern."""
        if pattern not in self._handlers:
            self._handlers[pattern] = []
        self._handlers[pattern].append(handler)

    def remove_route(self, pattern: str, handler: Callable[[Any], None]) -> bool:
        """Remove a handler from a pattern."""
        handlers = self._handlers.get(pattern)
        if handlers and handler in handlers:
            handlers.remove(handler)
            return True
        return False

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, routing_key: str, message: Any) -> int:
        """
        Route a message to all matching handlers.

        :returns: Number of handlers invoked.
        """
        matched = 0
        for pattern, handlers in self._handlers.items():
            if self._match(pattern, routing_key):
                for handler in handlers:
                    try:
                        handler(message)
                        matched += 1
                    except Exception as e:
                        logger.error(f"Route {routing_key} handler error: {e}")
        return matched

    def route_one(self, routing_key: str, message: Any) -> bool:
        """Route to first matching handler only."""
        for pattern, handlers in self._handlers.items():
            if self._match(pattern, routing_key):
                if handlers:
                    try:
                        handlers[0](message)
                        return True
                    except Exception as e:
                        logger.error(f"Route {routing_key} handler error: {e}")
        return False

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    @staticmethod
    def _match(pattern: str, routing_key: str) -> bool:
        """AMQP-style pattern matching."""
        # Convert AMQP wildcards to fnmatch
        converted = pattern.replace(".", "/")
        key = routing_key.replace(".", "/")
        # # matches zero or more words
        converted = converted.replace("#", "*")
        # * matches exactly one word
        converted = converted.replace("*/", "[^/]*/").replace("/*", "/[^/]*")
        # Need to handle exact word matching with fnmatch
        # Simpler: split and compare word by word
        return FleetRouter._match_words(pattern.split("."), routing_key.split("."))

    @staticmethod
    def _match_words(pattern_parts: List[str], key_parts: List[str]) -> bool:
        """Word-by-word AMQP matching."""
        pi, ki = 0, 0
        while pi < len(pattern_parts) and ki < len(key_parts):
            p = pattern_parts[pi]
            k = key_parts[ki]
            if p == "#":
                # # matches zero or more remaining parts
                # Try matching the rest of the pattern against all suffixes
                remaining_pattern = pattern_parts[pi + 1:]
                for j in range(ki, len(key_parts) + 1):
                    if FleetRouter._match_words(remaining_pattern, key_parts[j:]):
                        return True
                return False
            elif p == "*":
                # * matches exactly one part
                pi += 1
                ki += 1
            elif p == k:
                pi += 1
                ki += 1
            else:
                return False
        # Handle trailing #
        while pi < len(pattern_parts) and pattern_parts[pi] == "#":
            pi += 1
        return pi == len(pattern_parts) and ki == len(key_parts)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def patterns(self) -> List[str]:
        return list(self._handlers.keys())

    def handler_count(self, pattern: str) -> int:
        return len(self._handlers.get(pattern, []))

    def __repr__(self) -> str:
        total = sum(len(h) for h in self._handlers.values())
        return f"<FleetRouter patterns={len(self._handlers)} handlers={total}>"
