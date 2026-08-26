"""Signal/callback system for event-driven fleet architecture.

Decoupled publish-subscribe signal system. Agents and services can
emit signals without knowing who listens. Used for loose coupling
between fleet subsystems.

Usage:
    signals = SignalHandler()
    signals.connect("breed_complete", lambda ctx: print(ctx))
    signals.emit("breed_complete", {"agent": "alpha"})
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SignalHandler:
    """
    Signal/callback dispatcher.

    :param logger: Optional logger for signal errors.
    """

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, signal: str, handler: Callable[[Any], None]) -> None:
        """Register a handler for a signal."""
        if signal not in self._handlers:
            self._handlers[signal] = []
        self._handlers[signal].append(handler)

    def disconnect(self, signal: str, handler: Callable[[Any], None]) -> bool:
        """Remove a handler from a signal."""
        handlers = self._handlers.get(signal)
        if handlers and handler in handlers:
            handlers.remove(handler)
            return True
        return False

    def disconnect_all(self, signal: str) -> None:
        """Remove all handlers for a signal."""
        self._handlers.pop(signal, None)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def emit(self, signal: str, context: Any = None) -> None:
        """Emit a signal to all registered handlers."""
        handlers = self._handlers.get(signal, [])
        for handler in handlers:
            try:
                handler(context)
            except Exception as e:
                logger.error(f"Signal {signal} handler error: {e}")

    def emit_one(self, signal: str, context: Any = None) -> bool:
        """Emit to the first handler only. Returns True if handled."""
        handlers = self._handlers.get(signal, [])
        if not handlers:
            return False
        try:
            handlers[0](context)
            return True
        except Exception as e:
            logger.error(f"Signal {signal} handler error: {e}")
            return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_handlers(self, signal: str) -> bool:
        return bool(self._handlers.get(signal))

    def handler_count(self, signal: str) -> int:
        return len(self._handlers.get(signal, []))

    def signals(self) -> List[str]:
        return list(self._handlers.keys())

    def __repr__(self) -> str:
        total = sum(len(h) for h in self._handlers.values())
        return f"<SignalHandler signals={len(self._handlers)} handlers={total}>"
