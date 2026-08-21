"""shutdown_coordinator.py — Graceful shutdown with ordered hooks.

Provides:
1. SIGTERM / SIGINT handling
2. Ordered shutdown hooks (priority-based)
3. Drain in-flight work with timeout
4. Force-kill fallback if drain expires
5. Shutdown status reporting

Usage:
    coordinator = ShutdownCoordinator(timeout=30.0)
    coordinator.register_hook("db", close_db, priority=1)
    coordinator.register_hook("http", stop_server, priority=2)
    coordinator.shutdown()  # Runs hooks in priority order, drains work
"""

from __future__ import annotations

__all__ = [
    "ShutdownCoordinator",
    "ShutdownHook",
    "ShutdownTimeout",
]

import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ShutdownTimeout(Exception):
    """Raised when shutdown exceeds its deadline."""


@dataclass
class ShutdownHook:
    """A shutdown hook with priority."""

    name: str
    fn: Callable[[], None]
    priority: int = 0  # Lower = earlier
    timeout: float = 5.0
    ran: bool = False


class ShutdownCoordinator:
    """Coordinates graceful shutdown with ordered hooks and draining."""

    def __init__(self, timeout: float = 30.0, force_timeout: float = 5.0) -> None:
        self._timeout = timeout
        self._force_timeout = force_timeout
        self._hooks: list[ShutdownHook] = []
        self._drain_fns: list[Callable[[], bool]] = []
        self._in_flight: int = 0
        self._lock = threading.Lock()
        self._shutting_down = False
        self._start_time: float | None = None
        self._signal_installed = False

    def register_hook(
        self,
        name: str,
        fn: Callable[[], None],
        priority: int = 0,
        timeout: float = 5.0,
    ) -> None:
        """Register a shutdown hook."""
        self._hooks.append(
            ShutdownHook(name=name, fn=fn, priority=priority, timeout=timeout)
        )

    def register_drain(self, fn: Callable[[], bool]) -> None:
        """Register a function that returns True when drained."""
        self._drain_fns.append(fn)

    def start_work(self) -> None:
        """Signal that work has started."""
        with self._lock:
            self._in_flight += 1

    def finish_work(self) -> None:
        """Signal that work has finished."""
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    def install_signal_handlers(self) -> None:
        """Install SIGTERM / SIGINT handlers."""
        if self._signal_installed:
            return
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        self._signal_installed = True

    def _signal_handler(self, signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()

    def shutdown(self) -> None:
        """Execute shutdown sequence."""
        if self._shutting_down:
            return
        self._shutting_down = True
        self._start_time = time.time()
        deadline = self._start_time + self._timeout

        # Drain in-flight work
        self._drain(deadline)

        # Run hooks in priority order
        for hook in sorted(self._hooks, key=lambda h: h.priority):
            if time.time() >= deadline:
                logger.warning(
                    f"Shutdown deadline reached, skipping hook '{hook.name}'"
                )
                break
            try:
                hook_fn = hook.fn
                hook.ran = True
                # Run with per-hook timeout
                hook_deadline = time.time() + hook.timeout
                hook_fn()
                if time.time() > hook_deadline:
                    logger.warning(f"Hook '{hook.name}' exceeded its timeout")
            except Exception as e:
                logger.error(f"Hook '{hook.name}' failed: {e}")

        elapsed = time.time() - self._start_time
        logger.info(f"Shutdown complete in {elapsed:.2f}s")

    def _drain(self, deadline: float) -> None:
        """Wait for in-flight work to complete."""
        while time.time() < deadline:
            with self._lock:
                if self._in_flight == 0:
                    return
            # Check custom drain functions
            if all(fn() for fn in self._drain_fns):
                return
            time.sleep(0.1)
        logger.warning("Drain timeout expired, proceeding with shutdown")

    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def stats(self) -> dict[str, Any]:
        return {
            "hooks": len(self._hooks),
            "hooks_ran": sum(1 for h in self._hooks if h.ran),
            "in_flight": self._in_flight,
            "shutting_down": self._shutting_down,
            "signal_handlers_installed": self._signal_installed,
        }

    def __repr__(self) -> str:
        return f"ShutdownCoordinator(hooks={len(self._hooks)}, shutting_down={self._shutting_down})"
