"""Superinstance Runtime — COLLECT → SELECT → COMPILE event bus with plugin system.

See Task 2 in docs/STRATEGIC-ARCHITECTURE.md.
"""

from __future__ import annotations

__all__ = [
    "EventBus",
    "EventResult",
    "Plugin",
    "CollectorPlugin",
    "SelectorPlugin",
    "CompilerPlugin",
]

from superinstance.runtime import (
    CompilerPlugin,
    CollectorPlugin,
    EventBus,
    EventResult,
    Plugin,
    SelectorPlugin,
)
