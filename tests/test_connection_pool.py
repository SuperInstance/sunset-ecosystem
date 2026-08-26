"""Tests for connection_pool.py — Generic connection pool with health checking.

Run: python3 -m pytest tests/test_connection_pool.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.connection_pool import ConnectionPool


class TestConnectionPool:
    def test_create(self):
        pool = ConnectionPool(max_size=5, factory=lambda: "conn")
        assert pool.stats()["max"] == 5

    def test_acquire_and_release(self):
        pool = ConnectionPool(max_size=2, factory=lambda: object())
        conn = pool.acquire()
        assert pool.stats()["in_use"] == 1
        pool.release(conn)
        assert pool.stats()["in_use"] == 0
        assert pool.stats()["available"] == 1

    def test_pool_reuses(self):
        pool = ConnectionPool(max_size=2, factory=lambda: object())
        conn1 = pool.acquire()
        pool.release(conn1)
        conn2 = pool.acquire()
        assert conn1 is conn2

    def test_max_size_blocks(self):
        pool = ConnectionPool(max_size=1, factory=lambda: object())
        conn = pool.acquire()
        with pytest.raises(RuntimeError):
            pool.acquire(timeout=0.01)

    def test_health_check(self):
        healthy = [True]
        pool = ConnectionPool(
            max_size=2,
            factory=lambda: object(),
            health=lambda c: healthy[0],
        )
        conn = pool.acquire()
        pool.release(conn)
        healthy[0] = False
        conn2 = pool.acquire()
        assert conn2 is not conn

    def test_close(self):
        pool = ConnectionPool(max_size=2, factory=lambda: object())
        conn = pool.acquire()
        pool.close(conn)
        assert pool.stats()["in_use"] == 0
        assert pool.stats()["available"] == 0

    def test_repr(self):
        pool = ConnectionPool(max_size=2, factory=lambda: object())
        assert "ConnectionPool" in repr(pool)
