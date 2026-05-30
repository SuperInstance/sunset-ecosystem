"""A2A identity — re-exported from a2a.identity for backward compatibility."""

from __future__ import annotations

from a2a.identity import (
    AgentIdentity,
    AgentRegistry,
    TaskHandle,
    TaskState,
    ValidationError,
    NegotiationError,
)

__all__ = [
    "AgentIdentity",
    "AgentRegistry",
    "TaskHandle",
    "TaskState",
    "ValidationError",
    "NegotiationError",
]
