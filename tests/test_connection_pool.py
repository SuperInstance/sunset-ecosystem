"""Tests for connection_pool.py — Generic connection pooling.

Run: python3 -m pytest tests/test_connection_pool.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.connection_pool import ConnectionPool, PoolExhausted


class TestConnectionPool:
    def test_create(self):
        pool = ConnectionPool(factory=lambda: {"conn": 1}, max_connections=5)
        assert pool.stats()["max_connections"] == 5

    def test_borrow(self):
        pool = ConnectionPool(factory=lambda: {"conn": 1})
        conn = pool.borrow()
        assert conn == {"conn": 1}
        assert pool.stats()["in_use"] == 1

    def test_return_(self):
        pool = ConnectionPool(factory=lambda: {"conn": 1})
        conn = pool.borrow()
        pool.return_(conn)
        assert pool.stats()["in_use"] == 0
        assert pool.stats()["available"] == 1

    def test_reuse(self):
        pool = ConnectionPool(factory=lambda: {"conn": 1}, max_connections=2)
        conn1 = pool.borrow()
        pool.return_(conn1)
        conn2 = pool.borrow()
        assert conn1 is conn2  # Reused

    def test_max_connections(self):
        pool = ConnectionPool(factory=lambda: {"conn": 1}, max_connections=2, wait_timeout=0.1)
        c1 = pool.borrow()
        c2 = pool.borrow()
        assert pool.stats()["total_connections"] == 2
        with pytest.raises(PoolExhausted):
            pool.borrow()

    def test_exhausted_timeout(self):
        pool = ConnectionPool(factory=lambda: {"conn": 1}, max_connections=1, wait_timeout=0.1)
        c1 = pool.borrow()
        with pytest.raises(PoolExhausted):
            pool.borrow()

    def test_wait_for_return(self):
        pool = ConnectionPool(factory=lambda: {"conn": 1}, max_connections=1, wait_timeout=1.0)
        c1 = pool.borrow()

        def return_later():
            time.sleep(0.1)
            pool.return_(c1)

        import threading
        t = threading.Thread(target=return_later)
        t.start()
        c2 = pool.borrow()
        assert c2 is c1
        t.join()

    def test_idle_eviction(self):
        pool = ConnectionPool(factory=lambda: {"conn": 1}, idle_timeout=0.1)
        c1 = pool.borrow()
        pool.return_(c1)
        time.sleep(0.15)
        c2 = pool.borrow()
        assert pool.stats()["total_connections"] == 1  # Old evicted, new created

    def test_health_check(self):
        healthy = [False]
        pool = ConnectionPool(
            factory=lambda: {"conn": 1},
            health_check=lambda c: healthy.pop(0),
        )
        c1 = pool.borrow()
        pool.return_(c1)
        # Second borrow: health check returns False, connection evicted, new created
        c2 = pool.borrow()
        assert pool.stats()["total_created"] == 2

    def test_stats(self):
        pool = ConnectionPool(factory=lambda: {"conn": 1}, max_connections=3)
        c1 = pool.borrow()
        c2 = pool.borrow()
        pool.return_(c1)
        stats = pool.stats()
        assert stats["total_connections"] == 2
        assert stats["in_use"] == 1
        assert stats["available"] == 1
        assert stats["total_borrowed"] == 2
        assert stats["total_returned"] == 1

    def test_repr(self):
        pool = ConnectionPool(factory=lambda: {"conn": 1})
        assert "ConnectionPool" in repr(pool)
