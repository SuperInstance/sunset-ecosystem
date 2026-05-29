"""Immutable audit trail with tamper detection and append-only WAL-style logging.

Records structured audit events with HMAC signatures for integrity verification.
Used for compliance, debugging, and post-incident forensics.

Usage:
    logger = AuditLogger(secret=b"fleet-audit-key")
    logger.record("breed", agent="breeder-1", room="trap-alpha", score=0.95)
    for event in logger.tail(10):
        print(event.timestamp, event.payload)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditTamperError(Exception):
    pass


@dataclass
class AuditEvent:
    """A single audit record."""

    seq: int
    timestamp: float
    category: str
    payload: Dict[str, Any]
    signature: str = ""
    prev_hash: str = ""


class AuditLogger:
    """
    Append-only audit log with HMAC chain integrity.

    Each event stores a hash of the previous event's full record,
    creating an immutable chain. Tampering with any past event breaks
    the chain and is detected on verification.

    :param secret: HMAC secret for signing.
    :param max_events: In-memory retention limit (oldest discarded).
    """

    def __init__(
        self,
        secret: bytes,
        max_events: int = 10000,
    ):
        self._secret = secret
        self._max_events = max_events
        self._events: List[AuditEvent] = []
        self._seq = 0
        self._last_hash = ""

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        category: str,
        **kwargs: Any,
    ) -> AuditEvent:
        """Append a new audit event."""
        self._seq += 1
        event = AuditEvent(
            seq=self._seq,
            timestamp=time.time(),
            category=category,
            payload=kwargs,
            prev_hash=self._last_hash,
        )
        event.signature = self._sign(event)
        self._last_hash = self._hash_event(event)
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events.pop(0)
        return event

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def tail(self, n: int = 10) -> List[AuditEvent]:
        """Return last N events."""
        return self._events[-n:]

    def by_category(self, category: str) -> List[AuditEvent]:
        """Return all events matching category."""
        return [e for e in self._events if e.category == category]

    def since(self, timestamp: float) -> List[AuditEvent]:
        """Return events with timestamp >= value."""
        return [e for e in self._events if e.timestamp >= timestamp]

    def all(self) -> List[AuditEvent]:
        return list(self._events)

    def count(self) -> int:
        return len(self._events)

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    def verify(self) -> None:
        """Verify chain integrity. Raises AuditTamperError on first broken link."""
        prev_hash = ""
        for event in self._events:
            if event.prev_hash != prev_hash:
                raise AuditTamperError(
                    f"Chain broken at seq={event.seq}: expected prev_hash={prev_hash!r}, got {event.prev_hash!r}"
                )
            expected_sig = self._sign(event)
            if not hmac.compare_digest(expected_sig, event.signature):
                raise AuditTamperError(f"Signature mismatch at seq={event.seq}")
            prev_hash = self._hash_event(event)

    def verify_range(self, start_seq: int, end_seq: int) -> bool:
        """Verify integrity of a specific sequence range."""
        events = [e for e in self._events if start_seq <= e.seq <= end_seq]
        if not events:
            return True
        prev = events[0].prev_hash
        for event in events:
            if event.prev_hash != prev:
                return False
            expected = self._sign(event)
            if not hmac.compare_digest(expected, event.signature):
                return False
            prev = self._hash_event(event)
        return True

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def export_json(self) -> str:
        """Export all events as JSON."""
        return json.dumps(
            [
                {
                    "seq": e.seq,
                    "timestamp": e.timestamp,
                    "category": e.category,
                    "payload": e.payload,
                    "signature": e.signature,
                    "prev_hash": e.prev_hash,
                }
                for e in self._events
            ]
        )

    def import_json(self, data: str) -> None:
        """Replace current log with imported events."""
        raw = json.loads(data)
        self._events = []
        self._seq = 0
        self._last_hash = ""
        for item in raw:
            event = AuditEvent(
                seq=item["seq"],
                timestamp=item["timestamp"],
                category=item["category"],
                payload=item["payload"],
                signature=item["signature"],
                prev_hash=item["prev_hash"],
            )
            self._events.append(event)
            self._seq = max(self._seq, event.seq)
            self._last_hash = self._hash_event(event)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sign(self, event: AuditEvent) -> str:
        data = f"{event.seq}:{event.timestamp}:{event.category}:{json.dumps(event.payload, sort_keys=True)}:{event.prev_hash}"
        return hmac.new(
            self._secret, data.encode("utf-8"), hashlib.sha256
        ).hexdigest()[:32]

    def _hash_event(self, event: AuditEvent) -> str:
        data = f"{event.seq}:{event.signature}:{event.prev_hash}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()[:32]

    def __repr__(self) -> str:
        return f"<AuditLogger events={len(self._events)}>"
