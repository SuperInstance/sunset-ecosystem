"""Generic connection pool with health checking.

Manages a pool of connections with max size, timeout, and health
verification. Used for fleet database, API, and service connections.

Usage:
    pool = ConnectionPool(max_size=10, factory=make_conn, health=check_conn)
    conn = pool.acquire()
    pool.release(conn)
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional


class ConnectionPool:
    """
    Generic connection pool.

    :param max_size: Maximum connections in pool.
    :param factory: Callable returning a new connection.
    :param health: Callable(connection) -> bool for health check.
    :param idle_timeout: Seconds before idle connections are closed.
    """

    def __init__(
        self,
        max_size: int,
        factory: Callable[[], Any],
        health: Optional[Callable[[Any], bool]] = None,
        idle_timeout: Optional[float] = None,
    ):
        self._max_size = max_size
        self._factory = factory
        self._health = health
        self._idle_timeout = idle_timeout
        self._pool: List[Dict[str, Any]] = []
        self._in_use: set = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def acquire(self, timeout: Optional[float] = None) -> Any:
        """
        Acquire a connection.

        :param timeout: Max seconds to wait.
        :returns: Connection object.
        :raises RuntimeError: If pool exhausted and no connection available.
        """
        deadline = time.time() + timeout if timeout else None
        while True:
            with self._lock:
                # Check for available healthy connection
                now = time.time()
                for i, item in enumerate(self._pool):
                    if item["conn"] in self._in_use:
                        continue
                    if self._idle_timeout and now - item["last_used"] > self._idle_timeout:
                        continue
                    if self._health and not self._health(item["conn"]):
                        continue
                    self._in_use.add(item["conn"])
                    item["last_used"] = now
                    self._pool.pop(i)
                    return item["conn"]

                # Create new if under max
                if len(self._in_use) < self._max_size:
                    conn = self._factory()
                    self._in_use.add(conn)
                    return conn

            if deadline and time.time() >= deadline:
                raise RuntimeError("Connection pool exhausted")
            time.sleep(0.01)

    def release(self, conn: Any) -> None:
        """Release a connection back to the pool."""
        with self._lock:
            self._in_use.discard(conn)
            self._pool.append({"conn": conn, "last_used": time.time()})

    def close(self, conn: Any) -> None:
        """Remove a connection permanently."""
        with self._lock:
            self._in_use.discard(conn)
            self._pool = [item for item in self._pool if item["conn"] is not conn]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "available": len(self._pool),
                "in_use": len(self._in_use),
                "max": self._max_size,
            }

    def __repr__(self) -> str:
        with self._lock:
            return f"<ConnectionPool available={len(self._pool)} in_use={len(self._in_use)} max={self._max_size}>"
