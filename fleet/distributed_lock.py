from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class LockToken:
    """A distributed lock token."""
    resource: str
    holder: str
    timestamp: float
    ttl: float  # Time to live in seconds
    token_id: str

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource": self.resource,
            "holder": self.holder,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "token_id": self.token_id,
        }


class DistributedLock:
    """
    Distributed lock service for fleet coordination.

    Provides exclusive access to resources across fleet nodes.
    Locks have TTL and auto-expire if not renewed.
    """

    def __init__(self, fleet_node_id: str = "default", default_ttl: float = 30.0):
        self.fleet_node_id = fleet_node_id
        self.default_ttl = default_ttl
        self._locks: Dict[str, LockToken] = {}
        self._lock = threading.Lock()
        self._stats: Dict[str, int] = {"acquired": 0, "released": 0, "expired": 0, "failed": 0}

    def acquire(self, resource: str, holder: Optional[str] = None,
                ttl: Optional[float] = None) -> Optional[LockToken]:
        """
        Acquire a lock on a resource.
        Returns token if successful, None if already locked.
        """
        holder = holder or self.fleet_node_id
        ttl = ttl or self.default_ttl

        with self._lock:
            # Check if lock exists and is not expired
            if resource in self._locks:
                existing = self._locks[resource]
                if not existing.is_expired():
                    self._stats["failed"] += 1
                    return None
                else:
                    self._stats["expired"] += 1

            token = LockToken(
                resource=resource,
                holder=holder,
                timestamp=time.time(),
                ttl=ttl,
                token_id=hashlib.sha256(f"{resource}:{holder}:{time.time()}".encode()).hexdigest()[:16],
            )
            self._locks[resource] = token
            self._stats["acquired"] += 1
            return token

    def release(self, resource: str, token_id: str) -> bool:
        """
        Release a lock. Must provide correct token_id.
        Returns True if released, False if not found or mismatched.
        """
        with self._lock:
            if resource not in self._locks:
                return False
            existing = self._locks[resource]
            if existing.token_id != token_id:
                return False
            del self._locks[resource]
            self._stats["released"] += 1
            return True

    def renew(self, resource: str, token_id: str,
              extension: Optional[float] = None) -> bool:
        """Renew a lock's TTL."""
        with self._lock:
            if resource not in self._locks:
                return False
            existing = self._locks[resource]
            if existing.token_id != token_id:
                return False
            existing.ttl += extension or self.default_ttl
            existing.timestamp = time.time()
            return True

    def is_locked(self, resource: str) -> bool:
        """Check if a resource is currently locked."""
        with self._lock:
            if resource not in self._locks:
                return False
            return not self._locks[resource].is_expired()

    def get_holder(self, resource: str) -> Optional[str]:
        """Get the current holder of a lock."""
        with self._lock:
            if resource not in self._locks:
                return None
            token = self._locks[resource]
            if token.is_expired():
                return None
            return token.holder

    def list_locks(self) -> List[LockToken]:
        """List all active locks."""
        with self._lock:
            return [t for t in self._locks.values() if not t.is_expired()]

    def cleanup_expired(self) -> int:
        """Remove expired locks. Returns count removed."""
        with self._lock:
            expired = [r for r, t in self._locks.items() if t.is_expired()]
            for r in expired:
                del self._locks[r]
                self._stats["expired"] += 1
            return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """Get lock service statistics."""
        return {
            "active_locks": len(self.list_locks()),
            "total_locks": len(self._locks),
            **self._stats,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.fleet_node_id,
            "stats": self.get_stats(),
        }
