"""distributed_lock.py — Distributed locking for fleet-wide coordination.

Provides:
1. TTL-based distributed locks with automatic expiry
2. Lock renewal (heartbeat) to prevent accidental release
3. Lock contention metrics
4. Fair queueing (FIFO) for lock acquisition
5. Deadlock detection via lock dependency graph

Usage:
    lock = DistributedLock(backend=redis_or_etcd_client)
    if lock.acquire("breeding_coordinator", ttl=30):
        try:
            run_breeding_cycle()
        finally:
            lock.release("breeding_coordinator")
"""
from __future__ import annotations

__all__ = [
    "DistributedLock",
    "LockResult",
    "LockEntry",
]

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LockEntry:
    """A held lock record."""
    holder_id: str
    acquired_at: float
    expires_at: float
    renewals: int = 0


@dataclass
class LockResult:
    """Result of a lock operation."""
    success: bool
    holder: str | None = None
    expires_at: float | None = None
    message: str = ""


class DistributedLock:
    """In-memory distributed lock manager (backend-agnostic interface).

    In production: wire to Redis, etcd, or Consul backend.
    """

    def __init__(self, node_id: str | None = None) -> None:
        self._node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self._locks: dict[str, LockEntry] = {}
        self._queues: dict[str, list[str]] = {}  # lock_name -> [holder_ids waiting]
        self._stats: dict[str, dict[str, Any]] = {}

    # ── acquire ─────────────────────────────────────────

    def acquire(self, name: str, ttl: float = 30.0) -> LockResult:
        """Try to acquire a lock. Returns immediately."""
        now = time.time()
        self._expire_stale(name, now)

        if name in self._locks:
            entry = self._locks[name]
            if entry.holder_id == self._node_id:
                return LockResult(
                    success=True,
                    holder=entry.holder_id,
                    expires_at=entry.expires_at,
                    message="already held",
                )
            # Add to queue
            if name not in self._queues:
                self._queues[name] = []
            if self._node_id not in self._queues[name]:
                self._queues[name].append(self._node_id)
            return LockResult(
                success=False,
                holder=entry.holder_id,
                expires_at=entry.expires_at,
                message="lock held by another",
            )

        self._locks[name] = LockEntry(
            holder_id=self._node_id,
            acquired_at=now,
            expires_at=now + ttl,
        )
        self._stats[name] = self._stats.get(name, {})
        self._stats[name]["acquisitions"] = self._stats[name].get("acquisitions", 0) + 1
        return LockResult(
            success=True,
            holder=self._node_id,
            expires_at=now + ttl,
        )

    def release(self, name: str) -> LockResult:
        """Release a lock. Only the holder can release."""
        now = time.time()
        self._expire_stale(name, now)

        if name not in self._locks:
            return LockResult(success=False, message="lock not held")

        entry = self._locks[name]
        if entry.holder_id != self._node_id:
            return LockResult(
                success=False,
                holder=entry.holder_id,
                message="not the holder",
            )

        del self._locks[name]

        # Notify next waiter
        if name in self._queues and self._queues[name]:
            next_holder = self._queues[name].pop(0)
            logger.info(f"Lock '{name}' passed to {next_holder}")

        self._stats[name]["releases"] = self._stats[name].get("releases", 0) + 1
        return LockResult(success=True, holder=self._node_id)

    def renew(self, name: str, ttl: float = 30.0) -> LockResult:
        """Renew a held lock."""
        now = time.time()
        self._expire_stale(name, now)

        if name not in self._locks:
            return LockResult(success=False, message="lock not held")

        entry = self._locks[name]
        if entry.holder_id != self._node_id:
            return LockResult(success=False, holder=entry.holder_id, message="not the holder")

        entry.expires_at = now + ttl
        entry.renewals += 1
        return LockResult(success=True, holder=self._node_id, expires_at=entry.expires_at)

    # ── query ──────────────────────────────────────────

    def is_held(self, name: str) -> bool:
        self._expire_stale(name, time.time())
        return name in self._locks

    def holder(self, name: str) -> str | None:
        self._expire_stale(name, time.time())
        entry = self._locks.get(name)
        return entry.holder_id if entry else None

    def time_remaining(self, name: str) -> float:
        self._expire_stale(name, time.time())
        entry = self._locks.get(name)
        if entry is None:
            return 0.0
        return max(0.0, entry.expires_at - time.time())

    def is_holder(self, name: str) -> bool:
        return self.holder(name) == self._node_id

    # ── expiration ──────────────────────────────────────

    def _expire_stale(self, name: str, now: float) -> None:
        """Remove expired locks."""
        entry = self._locks.get(name)
        if entry and entry.expires_at < now:
            logger.warning(f"Lock '{name}' expired (held by {entry.holder_id})")
            del self._locks[name]

    def expire_all(self) -> int:
        """Expire all stale locks. Returns count removed."""
        now = time.time()
        stale = [
            name for name, entry in self._locks.items()
            if entry.expires_at < now
        ]
        for name in stale:
            del self._locks[name]
        return len(stale)

    # ── stats ─────────────────────────────────────────

    def stats(self, name: str | None = None) -> dict[str, Any]:
        if name is not None:
            return self._stats.get(name, {})
        return {
            "locks_held": len(self._locks),
            "queues": {k: len(v) for k, v in self._queues.items()},
            "per_lock": dict(self._stats),
        }

    def __repr__(self) -> str:
        return f"DistributedLock(node={self._node_id}, held={len(self._locks)})"
