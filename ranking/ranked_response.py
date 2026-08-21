"""RankedResponse — A single response ranked by the user."""

from __future__ import annotations

__all__ = ["RankedResponse"]

import time
from dataclasses import dataclass, field


@dataclass
class RankedResponse:
    """A single response with its ranking metadata.

    Attributes:
        response: The response text.
        source: Where it came from (model name, "distilled_v3", "nerve_compiled").
        rank: User's ranking (1 = best).
        hint_level: How many hints from the big model were used.
        seed: Random seed used to generate this response.
        temperature: Temperature setting.
        latency_ms: Time to generate this response.
        timestamp: When this response was generated.
    """

    response: str
    source: str = "unknown"
    rank: int = 0
    hint_level: int = 0
    seed: int = 0
    temperature: float = 0.7
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"RankedResponse(source={self.source!r}, rank={self.rank}, "
            f"hints={self.hint_level}, latency={self.latency_ms:.0f}ms)"
        )

    @property
    def is_distilled(self) -> bool:
        """Whether this response came from the distilled swarm."""
        return "distilled" in self.source.lower() or "nerve" in self.source.lower()

    @property
    def is_big_model(self) -> bool:
        """Whether this response came from the big model."""
        return not self.is_distilled
