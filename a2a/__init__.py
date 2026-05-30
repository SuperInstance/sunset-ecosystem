"""A2A (Agent-to-Agent) communication modules."""

from __future__ import annotations

from a2a.server import A2AServer
from a2a.identity import (
    AgentCard,
    AgentRegistry,
    AgentIdentity,
    TaskHandle,
    TaskState,
    NegotiationError,
    ValidationError,
)
from a2a.protocol import (
    A2AAgentCard,
    A2ATask,
    A2AClient,
    A2AProtocolAdapter,
    TaskStatus,
    A2AError,
)

__all__ = [
    "A2AServer",
    "AgentCard",
    "AgentRegistry",
    "AgentIdentity",
    "TaskHandle",
    "TaskState",
    "NegotiationError",
    "ValidationError",
    "A2AAgentCard",
    "A2ATask",
    "A2AClient",
    "A2AProtocolAdapter",
    "TaskStatus",
    "A2AError",
]
