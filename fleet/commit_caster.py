"""Commit-Caster I2I Router — Broadcasts commit events to the fleet mesh."""

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable


@dataclass
class CommitEvent:
    """A commit event from any SuperInstance repository."""

    repo: str
    commit: str
    author: str
    message: str
    branch: str
    timestamp: str
    files: List[str] = field(default_factory=list)
    received_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "CommitEvent":
        return cls(
            repo=d.get("repo", ""),
            commit=d.get("commit", ""),
            author=d.get("author", ""),
            message=d.get("message", ""),
            branch=d.get("branch", ""),
            timestamp=d.get("timestamp", ""),
            files=d.get("files", []),
            received_at=d.get("received_at", time.time()),
        )

    def fingerprint(self) -> str:
        """Unique deduplication key."""
        return f"{self.repo}:{self.commit}"


class CommitCaster:
    """Receives signed commit webhooks and broadcasts to the fleet."""

    def __init__(
        self,
        secret: str,
        mesh_broadcast: Optional[Callable] = None,
        window_sec: float = 60.0,
    ):
        self.secret = secret.encode() if isinstance(secret, str) else secret
        self.mesh_broadcast = mesh_broadcast
        self._seen: Dict[str, float] = {}
        self._window_sec = window_sec
        self._queue: List[CommitEvent] = []
        self._stats = {"received": 0, "accepted": 0, "rejected": 0, "queued": 0}

    def validate(self, payload: bytes, signature: str) -> bool:
        """Validate HMAC-SHA256 signature."""
        if not signature.startswith("sha256="):
            return False
        expected = hmac.new(self.secret, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature[7:])

    def receive(self, payload: bytes, signature: str) -> Optional[CommitEvent]:
        """Process an incoming webhook. Returns event or None if rejected."""
        self._stats["received"] += 1

        if not self.validate(payload, signature):
            self._stats["rejected"] += 1
            return None

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            self._stats["rejected"] += 1
            return None

        event = CommitEvent.from_dict(data)
        fp = event.fingerprint()

        now = time.time()
        if fp in self._seen and (now - self._seen[fp]) < self._window_sec:
            self._stats["rejected"] += 1
            return None

        self._seen[fp] = now
        self._cleanup_old(now)

        if self.mesh_broadcast is None:
            self._queue.append(event)
            self._stats["queued"] += 1
        else:
            try:
                self.mesh_broadcast(event.to_dict())
            except Exception:
                self._queue.append(event)
                self._stats["queued"] += 1
                return event

        self._stats["accepted"] += 1
        return event

    def flush_queue(self) -> int:
        """Retry broadcasting queued events. Returns count of successfully sent."""
        if self.mesh_broadcast is None:
            return 0
        sent = 0
        still_queued: List[CommitEvent] = []
        for event in self._queue:
            try:
                self.mesh_broadcast(event.to_dict())
                sent += 1
            except Exception:
                still_queued.append(event)
        self._queue = still_queued
        return sent

    def get_queue(self) -> List[CommitEvent]:
        return list(self._queue)

    def get_stats(self) -> Dict:
        return dict(self._stats, queue_size=len(self._queue))

    def _cleanup_old(self, now: float) -> None:
        cutoff = now - self._window_sec
        self._seen = {k: v for k, v in self._seen.items() if v > cutoff}

    def to_dict(self) -> Dict:
        return {
            "secret_set": bool(self.secret),
            "broadcast_set": self.mesh_broadcast is not None,
            "window_sec": self._window_sec,
            "stats": self.get_stats(),
        }
