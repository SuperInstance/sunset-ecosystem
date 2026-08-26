"""Record and query architectural decisions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence


__all__ = ["DecisionRecord", "DecisionRecords", "DecisionLog"]


class DecisionType(str, Enum):
    """Categories of architectural decisions."""

    ARCHITECTURE = "architecture"
    TECHNOLOGY = "technology"
    PATTERN = "pattern"
    API = "api"
    DATA_MODEL = "data_model"
    PROCESS = "process"
    DEPRECATION = "deprecation"
    OTHER = "other"


@dataclass
class DecisionRecord:
    """A single architectural decision."""

    id: str
    title: str
    decision_type: DecisionType
    decided_at: datetime
    decided_by: str
    context: str
    decision: str
    rationale: str
    alternatives: List[str] = field(default_factory=list)
    outcome: Optional[str] = None
    components: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    superseded_by: Optional[str] = None

    def __repr__(self) -> str:
        return f"DecisionRecord(id={self.id!r}, title={self.title!r})"

    def to_dict(self) -> Dict:
        d = {
            "id": self.id,
            "title": self.title,
            "decision_type": self.decision_type.value,
            "decided_at": self.decided_at.isoformat(),
            "decided_by": self.decided_by,
            "context": self.context,
            "decision": self.decision,
            "rationale": self.rationale,
            "alternatives": self.alternatives,
            "outcome": self.outcome,
            "components": self.components,
            "tags": self.tags,
            "superseded_by": self.superseded_by,
        }
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "DecisionRecord":
        return cls(
            id=data["id"],
            title=data["title"],
            decision_type=DecisionType(data["decision_type"]),
            decided_at=datetime.fromisoformat(data["decided_at"]),
            decided_by=data["decided_by"],
            context=data["context"],
            decision=data["decision"],
            rationale=data.get("rationale", ""),
            alternatives=data.get("alternatives", []),
            outcome=data.get("outcome"),
            components=data.get("components", []),
            tags=data.get("tags", []),
            superseded_by=data.get("superseded_by"),
        )


@dataclass
class DecisionRecords:
    """A collection of decision records with query results."""

    records: List[DecisionRecord] = field(default_factory=list)
    query: Optional[str] = None
    total: int = 0

    def __repr__(self) -> str:
        return f"DecisionRecords(total={self.total}, query={self.query!r})"


class DecisionLog:
    """Manages architectural decision records.

    Records can be stored on disk as JSON and queried by topic,
    component, decision type, or full-text search.
    """

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._records: Dict[str, DecisionRecord] = {}
        self._store_path = Path(store_path) if store_path else None
        if self._store_path and self._store_path.exists():
            self._load()

    def __repr__(self) -> str:
        return f"DecisionLog(records={len(self._records)}, store={self._store_path})"

    def record(
        self,
        title: str,
        decision_type: DecisionType,
        decided_by: str,
        context: str,
        decision: str,
        rationale: str = "",
        alternatives: Optional[List[str]] = None,
        components: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        decided_at: Optional[datetime] = None,
        record_id: Optional[str] = None,
    ) -> DecisionRecord:
        """Create and store a new decision record."""
        if decided_at is None:
            decided_at = datetime.now(timezone.utc)
        if record_id is None:
            record_id = f"ADR-{len(self._records) + 1:04d}"

        rec = DecisionRecord(
            id=record_id,
            title=title,
            decision_type=decision_type,
            decided_at=decided_at,
            decided_by=decided_by,
            context=context,
            decision=decision,
            rationale=rationale,
            alternatives=alternatives or [],
            components=components or [],
            tags=tags or [],
        )
        self._records[rec.id] = rec
        self._save()
        return rec

    def get(self, record_id: str) -> Optional[DecisionRecord]:
        """Retrieve a decision by ID."""
        return self._records.get(record_id)

    def supersede(self, old_id: str, new_id: str) -> bool:
        """Mark an old decision as superseded by a new one."""
        old = self._records.get(old_id)
        if old is None:
            return False
        old.superseded_by = new_id
        self._save()
        return True

    def set_outcome(self, record_id: str, outcome: str) -> bool:
        """Set the outcome of a decision."""
        rec = self._records.get(record_id)
        if rec is None:
            return False
        rec.outcome = outcome
        self._save()
        return True

    def query(
        self,
        topic: Optional[str] = None,
        component: Optional[str] = None,
        decision_type: Optional[DecisionType] = None,
        tag: Optional[str] = None,
        text_search: Optional[str] = None,
        include_superseded: bool = False,
    ) -> DecisionRecords:
        """Query decisions by various criteria."""
        results: List[DecisionRecord] = []
        text_lower = text_search.lower() if text_search else None
        topic_lower = topic.lower() if topic else None

        for rec in self._records.values():
            if not include_superseded and rec.superseded_by:
                continue

            if decision_type and rec.decision_type != decision_type:
                continue
            if component and component not in rec.components:
                continue
            if tag and tag not in rec.tags:
                continue

            if topic_lower:
                searchable = " ".join(
                    [
                        rec.title,
                        rec.context,
                        rec.decision,
                        rec.rationale,
                        " ".join(rec.tags),
                        " ".join(rec.components),
                    ]
                ).lower()
                if topic_lower not in searchable:
                    continue

            if text_lower:
                searchable = " ".join(
                    [
                        rec.title,
                        rec.context,
                        rec.decision,
                        rec.rationale,
                        " ".join(rec.alternatives),
                    ]
                ).lower()
                if text_lower not in searchable:
                    continue

            results.append(rec)

        results.sort(key=lambda r: r.decided_at, reverse=True)

        return DecisionRecords(
            records=results,
            query=topic or text_search,
            total=len(results),
        )

    def all_records(self, include_superseded: bool = False) -> DecisionRecords:
        """Return all records."""
        recs = [
            r
            for r in self._records.values()
            if include_superseded or not r.superseded_by
        ]
        recs.sort(key=lambda r: r.decided_at, reverse=True)
        return DecisionRecords(records=recs, total=len(recs))

    def _save(self) -> None:
        if self._store_path is None:
            return
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {rid: rec.to_dict() for rid, rec in self._records.items()}
        tmp = self._store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._store_path)

    def _load(self) -> None:
        try:
            data = json.loads(self._store_path.read_text())  # type: ignore[union-attr]
            for rid, rdata in data.items():
                self._records[rid] = DecisionRecord.from_dict(rdata)
        except (json.JSONDecodeError, OSError):
            pass
