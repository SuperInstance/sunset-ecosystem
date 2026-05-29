"""Distributed lock with TTL and renewal.

Implements a distributed mutual exclusion lock with time-to-live and
automatic renewal. Used for fleet-wide resource coordination, singleton
tasks, and critical section protection.

Usage:
    lock = DistributedLock(lock_id="task-1", ttl_sec=30)
    lock.acquire("node-1")
    lock.renew()  # Reset TTL
    lock.release()
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional


class DistributedLock:
    """
    Distributed lock with TTL.

    :param lock_id: Unique lock identifier.
    :param ttl_sec: Lock time-to-live in seconds.
    :param clock: Optional clock function for testing.
    """

    def __init__(
        self,
        lock_id: str,
        ttl_sec: float = 30.0,
        clock: Optional[callable] = None,
    ):
        self.lock_id = lock_id
        self.ttl_sec = ttl_sec
        self._clock = clock or time.time
        self._owner: Optional[str] = None
        self._acquired_at: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def acquire(self, owner: str, blocking: bool = False) -> bool:
        """
        Acquire the lock.

        :param owner: Lock owner identifier.
        :param blocking: Wait for lock (not implemented, returns False if held).
        :returns: True if acquired.
        """
        if self.is_locked() and self._owner != owner:
            return False
        self._owner = owner
        self._acquired_at = self._clock()
        return True

    def release(self, owner: str) -> bool:
        """
        Release the lock.

        :param owner: Must match current owner.
        :returns: True if released.
        """
        if self._owner != owner:
            return False
        self._owner = None
        self._acquired_at = 0.0
        return True

    def renew(self, owner: str) -> bool:
        """
        Renew the lock TTL.

        :param owner: Must match current owner.
        :returns: True if renewed.
        """
        if self._owner != owner:
            return False
        self._acquired_at = self._clock()
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_locked(self) -> bool:
        """Check if lock is currently held (and not expired)."""
        if self._owner is None:
            return False
        elapsed = self._clock() - self._acquired_at
        return elapsed < self.ttl_sec

    def owner(self) -> Optional[str]:
        """Get current owner (or None if expired/unheld)."""
        if self.is_locked():
            return self._owner
        return None

    def time_remaining(self) -> float:
        """Get seconds until lock expires (0 if not held)."""
        if self._owner is None:
            return 0.0
        elapsed = self._clock() - self._acquired_at
        remaining = self.ttl_sec - elapsed
        return max(0.0, remaining)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "lock_id": self.lock_id,
            "ttl_sec": self.ttl_sec,
            "locked": self.is_locked(),
            "owner": self.owner(),
            "time_remaining": self.time_remaining(),
        }

    def __repr__(self) -> str:
        return f"<DistributedLock id={self.lock_id} owner={self._owner}>"
