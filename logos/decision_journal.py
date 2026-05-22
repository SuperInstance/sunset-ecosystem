"""Decision Journal — FLAME format (Fleet Log of Agent Memory and Explanation).

Structured log of human-fleet interactions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "Decision",
    "DecisionJournal",
    "log_spawn",
    "log_sunset",
    "log_breed",
    "log_human_command",
    "get_decision_history",
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


def _resolve_journal_path(journal_path: Optional[str]) -> Path:
    """Return the concrete JSONL file path.

    If *journal_path* is omitted, use ``data/decisions/YYYY-MM-DD.jsonl``.
    If *journal_path* points to an existing directory (or has no ``.jsonl``
    suffix), treat it as a directory and append the daily filename.
    Otherwise use it as a direct file path.
    """
    if journal_path is None:
        return _default_journal_path()
    p = Path(journal_path)
    if p.suffix != ".jsonl" or p.is_dir():
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return p / f"{today}.jsonl"
    return p


def _append_jsonl(record: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_spawn(
    agent_id: int,
    parents: Optional[tuple[Optional[int], Optional[int]]] = None,
    generation: int = 0,
    reason: str = "",
    journal_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Log an agent spawn decision to the daily JSONL journal.

    Returns the record that was written.
    """
    path = _resolve_journal_path(journal_path)
    record: Dict[str, Any] = {
        "timestamp": time.time(),
        "operation": "spawn",
        "agent_id": agent_id,
        "generation": generation,
        "parents": list(filter(None, parents or ())),  # type: ignore[arg-type]
        "reason": reason or "fleet spawn",
        "why": f"spawn agent {agent_id}",
        "what": "EGG → INCUBATE",
        "expected": "active agent",
        "actual": "",
        "confidence": 1.0,
        "scope": str(agent_id),
    }
    _append_jsonl(record, path)
    return record


def log_sunset(
    agent_id: int,
    reason: str,
    generation: int = 0,
    journal_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Log an agent sunset decision to the daily JSONL journal.

    Returns the record that was written.
    """
    path = _resolve_journal_path(journal_path)
    record: Dict[str, Any] = {
        "timestamp": time.time(),
        "operation": "sunset",
        "agent_id": agent_id,
        "generation": generation,
        "reason": reason,
        "why": f"sunset agent {agent_id}",
        "what": "→ SUNSET",
        "expected": "resources freed",
        "actual": "",
        "confidence": 1.0,
        "scope": str(agent_id),
    }
    _append_jsonl(record, path)
    return record


def log_breed(
    parent_a: int,
    parent_b: Optional[int],
    child_id: int,
    generation: int = 0,
    journal_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Log a breeding decision to the daily JSONL journal.

    Returns the record that was written.
    """
    path = _resolve_journal_path(journal_path)
    record: Dict[str, Any] = {
        "timestamp": time.time(),
        "operation": "breed",
        "agent_id": child_id,
        "generation": generation,
        "parents": [parent_a, parent_b] if parent_b is not None else [parent_a],
        "why": f"breed child {child_id} from {parent_a}" + (f" × {parent_b}" if parent_b else ""),
        "what": "BREED queue → step",
        "expected": "viable child",
        "actual": "",
        "confidence": 1.0,
        "scope": f"{parent_a},{parent_b or 'solo'}",
    }
    _append_jsonl(record, path)
    return record


def log_human_command(
    intent: Any,
    confirmed: bool,
    scope: str,
    journal_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Log a human fleet command to the daily JSONL journal.

    *intent* is an Intent dataclass (or any object with ``action``,
    ``raw_command``, and ``is_destructive`` attributes).

    Returns the record that was written.
    """
    path = _resolve_journal_path(journal_path)
    raw_command = getattr(intent, "raw_command", "")
    action = getattr(intent, "action", "unknown")
    is_destructive = bool(getattr(intent, "is_destructive", lambda: False)())
    record: Dict[str, Any] = {
        "timestamp": time.time(),
        "operation": "human_command",
        "agent_id": None,
        "generation": 0,
        "parents": [],
        "why": raw_command,
        "what": f"{action} → {scope}",
        "expected": "fleet-wide action" if scope == "all" else "scoped action",
        "actual": "confirmed" if confirmed else "pending",
        "confidence": 1.0 if confirmed else 0.5,
        "scope": scope,
        "metadata": {
            "destructive": is_destructive,
            "confirmed": confirmed,
        },
    }
    _append_jsonl(record, path)
    return record


def get_decision_history(
    agent_id: Optional[int] = None,
    operation: Optional[str] = None,
    journal_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query the daily JSONL journal by agent_id and/or operation.

    Returns matching records in chronological order.
    """
    path = _resolve_journal_path(journal_path)
    if not path.exists():
        return []

    matches: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if agent_id is not None and record.get("agent_id") != agent_id:
                continue
            if operation is not None and record.get("operation") != operation:
                continue
            matches.append(record)

    return matches
