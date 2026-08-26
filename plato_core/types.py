"""plato_core.types — minimal stub for sunset-ecosystem compatibility.

This is a lightweight stand-in for the full plato_core.types module.
When the real plato_core package is installed, it will take precedence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, List, Optional


class LamportClock:
    def __init__(self, node_id: int = 0) -> None:
        self._tick = 0
        self.node_id = node_id

    def tick(self) -> int:
        self._tick += 1
        return self._tick

    def update(self, other: int) -> int:
        self._tick = max(self._tick, other) + 1
        return self._tick


@dataclass
class LifecycleEvent:
    from_state: Any = None
    to_state: Any = None
    reason: str = ""
    lamport: int = 0


class TileLifecycle:
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class TileType:
    CHECKPOINT = "checkpoint"
    PREDICTION = "prediction"
    EVALUATION = "evaluation"
    METRICS = "metrics"
    DECISION = "decision"
    EPISTEME = "episteme"
    HYBRID = "hybrid"
    BIRTH = "birth"
    SEED = "seed"
    REFINEMENT = "refinement"
    INTEGRATION = "integration"


@dataclass
class TrainingTile:
    tile_id: str = ""
    tile_type: str = TileType.METRICS
    room: str = ""
    description: str = ""
    state: str = TileLifecycle.ACTIVE
    lamport: int = 0
    lifecycle_events: List[LifecycleEvent] = field(default_factory=list)
    content_hash: str = ""
    signature: str = ""
    name: str = ""
    _payload: dict = field(default_factory=dict)
    _extra: dict = field(default_factory=dict)

    def is_active(self) -> bool:
        return self.state == TileLifecycle.ACTIVE

    def transition(self, new_state: str, reason: str = "", lamport: int = 0) -> None:
        self.state = new_state
        self.lifecycle_events.append(
            LifecycleEvent(
                from_state=self.state,
                to_state=new_state,
                reason=reason,
                lamport=lamport,
            )
        )

    def to_dict(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "tile_type": self.tile_type,
            "room": self.room,
            "description": self.description,
            "state": self.state,
            "lamport": self.lamport,
            "lifecycle_events": [
                {
                    "from_state": e.from_state,
                    "to_state": e.to_state,
                    "reason": e.reason,
                    "lamport": e.lamport,
                }
                for e in self.lifecycle_events
            ],
            "content_hash": self.content_hash,
            "signature": self.signature,
            "name": self.name,
            "_payload": self._payload,
            **self._extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingTile":
        events = [
            LifecycleEvent(
                from_state=e.get("from_state"),
                to_state=e.get("to_state"),
                reason=e.get("reason", ""),
                lamport=e.get("lamport", 0),
            )
            for e in d.get("lifecycle_events", [])
        ]
        kwargs = {k: v for k, v in d.items() if k != "lifecycle_events"}
        kwargs["lifecycle_events"] = events
        return cls(**kwargs)


def content_hash(data: str | bytes) -> str:
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()[:16]
    return hashlib.sha256(data.encode()).hexdigest()[:16]
