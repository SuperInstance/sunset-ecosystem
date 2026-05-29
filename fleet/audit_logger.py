"""Structured audit logger with tamper detection.

Records auditable actions with HMAC integrity verification. Used for
fleet security audit trails, compliance, and forensic analysis.

Usage:
    logger = AuditLogger(secret="hmac-key")
    logger.record("user.login", user="alice", ip="10.0.0.1")
    entries = logger.query(action="user.login")
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Dict, List, Optional


class AuditLogger:
    """
    Append-only audit log with HMAC integrity.

    :param secret: HMAC key for tamper detection.
    :param capacity: Maximum entries before eviction.
    """

    def __init__(self, secret: str, capacity: int = 10000):
        self._secret = secret.encode("utf-8")
        self._capacity = capacity
        self._entries: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an auditable action.

        :param action: Action identifier.
        :param context: Additional structured data.
        """
        entry = {
            "timestamp": time.time(),
            "action": action,
            "context": context or {},
        }
        entry["hmac"] = self._hmac(entry)
        self._entries.append(entry)
        if len(self._entries) > self._capacity:
            self._entries.pop(0)

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    def _hmac(self, entry: Dict[str, Any]) -> str:
        """Compute HMAC for an entry."""
        payload = f"{entry['timestamp']}:{entry['action']}:{entry['context']}"
        return hmac.new(self._secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify(self, entry: Dict[str, Any]) -> bool:
        """Verify an entry's HMAC."""
        stored = entry.pop("hmac", None)
        computed = self._hmac(entry)
        if stored:
            entry["hmac"] = stored
        return hmac.compare_digest(stored or "", computed)

    def verify_all(self) -> List[Dict[str, Any]]:
        """Return all entries that fail verification."""
        bad: List[Dict[str, Any]] = []
        for entry in self._entries:
            if not self.verify(entry):
                bad.append(entry)
        return bad

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query(
        self,
        action: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Query audit entries."""
        results: List[Dict[str, Any]] = []
        for entry in self._entries:
            if action and entry["action"] != action:
                continue
            if since and entry["timestamp"] < since:
                continue
            if until and entry["timestamp"] > until:
                continue
            results.append(entry)
        return results

    def count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        by_action: Dict[str, int] = {}
        for entry in self._entries:
            a = entry["action"]
            by_action[a] = by_action.get(a, 0) + 1
        return {
            "total": len(self._entries),
            "by_action": by_action,
        }

    def __repr__(self) -> str:
        return f"<AuditLogger entries={len(self._entries)}>"
