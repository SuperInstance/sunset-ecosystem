"""Distributed lease manager with TTL and renewal.

Manages time-bound leases for resource coordination. Supports
renewal, expiration, and ownership tracking. Used for fleet leader
election, resource locking, and session management.

Usage:
    leases = LeaseManager()
    leases.acquire("leader", owner="node-1", ttl_sec=30)
    assert leases.is_owner("leader", "node-1")
    leases.renew("leader", ttl_sec=30)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Lease:
    """A lease record."""

    key: str
    owner: str
    expires_at: float
    metadata: Dict[str, Any]


class LeaseManager:
    """
    In-memory lease manager with TTL.

    :param clock: Optional clock function for testing.
    """

    def __init__(self, clock: Optional[callable] = None):
        self._leases: Dict[str, Lease] = {}
        self._clock = clock or time.time

    # ------------------------------------------------------------------
    # Lease operations
    # ------------------------------------------------------------------

    def acquire(
        self,
        key: str,
        owner: str,
        ttl_sec: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Acquire a lease if available or expired.

        :returns: True if acquired.
        """
        self._evict()
        existing = self._leases.get(key)
        if existing and existing.expires_at > self._clock():
            return False
        self._leases[key] = Lease(
            key=key,
            owner=owner,
            expires_at=self._clock() + ttl_sec,
            metadata=metadata or {},
        )
        return True

    def renew(self, key: str, owner: str, ttl_sec: float) -> bool:
        """Renew a lease if still owned."""
        lease = self._leases.get(key)
        if not lease or lease.owner != owner:
            return False
        if lease.expires_at <= self._clock():
            return False
        lease.expires_at = self._clock() + ttl_sec
        return True

    def release(self, key: str, owner: str) -> bool:
        """Release a lease."""
        lease = self._leases.get(key)
        if lease and lease.owner == owner:
            del self._leases[key]
            return True
        return False

    def force_release(self, key: str) -> bool:
        """Forcefully release a lease."""
        if key in self._leases:
            del self._leases[key]
            return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_owner(self, key: str, owner: str) -> bool:
        """Check if owner currently holds the lease."""
        lease = self._leases.get(key)
        if not lease:
            return False
        if lease.expires_at <= self._clock():
            return False
        return lease.owner == owner

    def get_owner(self, key: str) -> Optional[str]:
        """Get current owner of a lease."""
        lease = self._leases.get(key)
        if lease and lease.expires_at > self._clock():
            return lease.owner
        return None

    def ttl(self, key: str) -> Optional[float]:
        """Get remaining TTL in seconds."""
        lease = self._leases.get(key)
        if not lease:
            return None
        remaining = lease.expires_at - self._clock()
        return remaining if remaining > 0 else 0.0

    def list_leases(self) -> List[str]:
        """List all active lease keys."""
        self._evict()
        return list(self._leases.keys())

    def list_by_owner(self, owner: str) -> List[str]:
        """List all leases held by an owner."""
        self._evict()
        return [
            key for key, lease in self._leases.items() if lease.owner == owner
        ]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _evict(self) -> None:
        """Remove expired leases."""
        now = self._clock()
        expired = [
            key for key, lease in self._leases.items() if lease.expires_at <= now
        ]
        for key in expired:
            del self._leases[key]

    def purge(self) -> None:
        """Remove all leases."""
        self._leases.clear()

    def stats(self) -> Dict[str, int]:
        self._evict()
        return {"active": len(self._leases)}

    def __repr__(self) -> str:
        self._evict()
        return f"<LeaseManager active={len(self._leases)}>"
