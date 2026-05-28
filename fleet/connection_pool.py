"""connection_pool.py — Generic connection pooling for fleet services.

Provides:
1. Borrow/return connection lifecycle
2. Max connections per pool
3. Connection health checks
4. Idle timeout eviction
5. Wait-timeout for exhausted pools

Usage:
    pool = ConnectionPool(max_connections=10, factory=create_db_connection)
    conn = pool.borrow()
    try:
        conn.query("SELECT * FROM agents")
    finally:
        pool.return_(conn)
"""
from __future__ import annotations

__all__ = [
    "ConnectionPool",
    "PooledConnection",
    "PoolExhausted",
]

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


class PoolExhausted(Exception):
    """Raised when pool is exhausted and wait timeout expires."""


@dataclass
class PooledConnection:
    """A connection wrapper with pool metadata."""
    conn: Any
    created_at: float
    last_used: float
    use_count: int = 0
    healthy: bool = True


class ConnectionPool:
    """Generic connection pool with health checks and eviction."""

    def __init__(
        self,
        factory: Callable[[], Any],
        max_connections: int = 10,
        idle_timeout: float = 300.0,
        wait_timeout: float = 5.0,
        health_check: Callable[[Any], bool] | None = None,
    ) -> None:
        self._factory = factory
        self._max_connections = max_connections
        self._idle_timeout = idle_timeout
        self._wait_timeout = wait_timeout
        self._health_check = health_check
        self._pool: list[PooledConnection] = []
        self._in_use_ids: set[int] = set()  # Track by id() for unhashable objects
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._total_created = 0
        self._total_borrowed = 0
        self._total_returned = 0

    def borrow(self) -> Any:
        """Borrow a connection from the pool."""
        with self._condition:
            deadline = time.time() + self._wait_timeout
            while True:
                # Try to get an available connection
                now = time.time()
                available = [p for p in self._pool if id(p.conn) not in self._in_use_ids]
                # Remove stale connections
                stale = [p for p in available if now - p.last_used > self._idle_timeout]
                for p in stale:
                    self._pool.remove(p)
                    available.remove(p)

                # Find a healthy available connection
                for p in available:
                    if self._health_check is not None and not self._health_check(p.conn):
                        p.healthy = False
                        self._pool.remove(p)
                        continue
                    self._in_use_ids.add(id(p.conn))
                    p.last_used = now
                    p.use_count += 1
                    self._total_borrowed += 1
                    return p.conn

                # Create new connection if under limit
                if len(self._pool) < self._max_connections:
                    try:
                        raw = self._factory()
                        pc = PooledConnection(
                            conn=raw,
                            created_at=now,
                            last_used=now,
                            use_count=1,
                        )
                        self._pool.append(pc)
                        self._in_use_ids.add(id(raw))
                        self._total_created += 1
                        self._total_borrowed += 1
                        return raw
                    except Exception as e:
                        logger.error(f"Connection factory failed: {e}")
                        raise PoolExhausted(f"Failed to create connection: {e}") from e

                # Wait for a connection to be returned
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise PoolExhausted("Pool exhausted, wait timeout expired")
                self._condition.wait(timeout=remaining)

    def return_(self, conn: Any) -> None:
        """Return a connection to the pool."""
        with self._lock:
            conn_id = id(conn)
            if conn_id in self._in_use_ids:
                self._in_use_ids.remove(conn_id)
                self._total_returned += 1
                self._condition.notify()

    def stats(self) -> dict[str, Any]:
        """Pool statistics."""
        with self._lock:
            available = len([p for p in self._pool if id(p.conn) not in self._in_use_ids])
            return {
                "total_connections": len(self._pool),
                "in_use": len(self._in_use_ids),
                "available": available,
                "max_connections": self._max_connections,
                "total_created": self._total_created,
                "total_borrowed": self._total_borrowed,
                "total_returned": self._total_returned,
            }

    def __repr__(self) -> str:
        stats = self.stats()
        return f"ConnectionPool(total={stats['total_connections']}, in_use={stats['in_use']})"
