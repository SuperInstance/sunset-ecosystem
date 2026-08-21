"""Simple key-value store with TTL and transactions.

In-memory KV store with expiration, atomic batches, and nested key
namespaces. Used for fleet configuration, session state, and caching.

Usage:
    kv = KeyValueStore()
    kv.set("key", "value", ttl_sec=60)
    assert kv.get("key") == "value"
    kv.delete("key")
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple


class KeyValueStore:
    """
    Key-value store with TTL support.

    :param namespace: Optional prefix for all keys.
    """

    def __init__(self, namespace: str = ""):
        self._namespace = namespace
        self._data: Dict[str, Tuple[Any, Optional[float]]] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def _prefixed(self, key: str) -> str:
        return f"{self._namespace}:{key}" if self._namespace else key

    def set(self, key: str, value: Any, ttl_sec: Optional[float] = None) -> None:
        """Set a key with optional TTL."""
        expires = time.time() + ttl_sec if ttl_sec else None
        self._data[self._prefixed(key)] = (value, expires)

    def get(self, key: str) -> Optional[Any]:
        """Get a value, or None if expired/missing."""
        self._evict()
        entry = self._data.get(self._prefixed(key))
        if entry is None:
            return None
        value, expires = entry
        if expires and time.time() > expires:
            self.delete(key)
            return None
        return value

    def delete(self, key: str) -> bool:
        """Delete a key."""
        prefixed = self._prefixed(key)
        if prefixed in self._data:
            del self._data[prefixed]
            return True
        return False

    def has(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None

    def keys(self) -> List[str]:
        """Return all non-expired keys (without namespace prefix)."""
        self._evict()
        result: List[str] = []
        for k in self._data:
            if self._namespace:
                if k.startswith(f"{self._namespace}:"):
                    result.append(k[len(self._namespace) + 1 :])
            else:
                result.append(k)
        return sorted(result)

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def batch_set(self, items: Dict[str, Any], ttl_sec: Optional[float] = None) -> None:
        """Atomically set multiple keys."""
        expires = time.time() + ttl_sec if ttl_sec else None
        for key, value in items.items():
            self._data[self._prefixed(key)] = (value, expires)

    def batch_delete(self, keys: List[str]) -> int:
        """Delete multiple keys. Returns count deleted."""
        count = 0
        for key in keys:
            if self.delete(key):
                count += 1
        return count

    # ------------------------------------------------------------------
    # TTL management
    # ------------------------------------------------------------------

    def ttl(self, key: str) -> Optional[float]:
        """Get remaining TTL in seconds, or None if no TTL."""
        entry = self._data.get(self._prefixed(key))
        if not entry:
            return None
        _, expires = entry
        if not expires:
            return None
        remaining = expires - time.time()
        return remaining if remaining > 0 else 0.0

    def expire(self, key: str, ttl_sec: float) -> bool:
        """Set/reset TTL on an existing key."""
        prefixed = self._prefixed(key)
        entry = self._data.get(prefixed)
        if not entry:
            return False
        value, _ = entry
        self._data[prefixed] = (value, time.time() + ttl_sec)
        return True

    def _evict(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired = [k for k, (_, exp) in self._data.items() if exp and now > exp]
        for k in expired:
            del self._data[k]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def size(self) -> int:
        self._evict()
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()

    def __repr__(self) -> str:
        return f"<KeyValueStore namespace={self._namespace!r} keys={self.size()}>"
