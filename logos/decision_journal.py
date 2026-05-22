"""Decision Journal — FLAME format (Fleet Log of Agent Memory and Explanation).

Structured log of human-fleet interactions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "Decision",
    "DecisionJournal",
]


@dataclass
class Decision:
    """A single logged human-fleet decision."""

    timestamp: float
    why: str        # human intent
    what: str       # action taken
    expected: str   # expected outcome
    actual: str     # actual outcome (filled later)
    confidence: float  # 0-1
    scope: str      # which agents affected
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "why": self.why,
            "what": self.what,
            "expected": self.expected,
            "actual": self.actual,
            "confidence": round(self.confidence, 4),
            "scope": self.scope,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Decision":
        return cls(
            timestamp=data["timestamp"],
            why=data["why"],
            what=data["what"],
            expected=data["expected"],
            actual=data.get("actual", ""),
            confidence=data.get("confidence", 0.0),
            scope=data["scope"],
            metadata=data.get("metadata", {}),
        )


class DecisionJournal:
    """Structured log of human-fleet interactions in FLAME format.

    Format per record:
    {
        'timestamp': float,
        'why': str,        # human intent
        'what': str,       # action taken
        'expected': str,   # expected outcome
        'actual': str,     # actual outcome (filled later)
        'confidence': float, # 0-1
        'scope': str,      # which agents affected
    }
    """

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._entries: List[Decision] = []
        self._store_path = Path(store_path) if store_path else None
        if self._store_path and self._store_path.exists():
            self._load()

    def __repr__(self) -> str:
        return f"DecisionJournal(entries={len(self._entries)}, store={self._store_path})"

    def record(
        self,
        why: str,
        what: str,
        expected: str,
        actual: str = "",
        confidence: float = 0.0,
        scope: str = "",
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Decision:
        """Create and store a new decision record."""
        if timestamp is None:
            timestamp = time.time()

        decision = Decision(
            timestamp=timestamp,
            why=why,
            what=what,
            expected=expected,
            actual=actual,
            confidence=confidence,
            scope=scope,
            metadata=metadata or {},
        )
        self._entries.append(decision)
        self._save()
        return decision

    def update_actual(self, index: int, actual: str) -> bool:
        """Update the actual outcome of a decision by index."""
        if 0 <= index < len(self._entries):
            self._entries[index].actual = actual
            self._save()
            return True
        return False

    def recent(self, n: int = 10) -> List[Decision]:
        """Return the n most recent decisions, newest first."""
        return list(reversed(self._entries[-n:]))

    def all_entries(self) -> List[Decision]:
        """Return all decisions in chronological order."""
        return list(self._entries)

    def _save(self) -> None:
        if self._store_path is None:
            return
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        data = [entry.to_dict() for entry in self._entries]
        tmp = self._store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._store_path)

    def _load(self) -> None:
        try:
            raw = self._store_path.read_text()  # type: ignore[union-attr]
            data = json.loads(raw)
            self._entries = [Decision.from_dict(d) for d in data]
        except (json.JSONDecodeError, OSError, KeyError):
            self._entries = []
