"""Tests for thread_pool.py — Thread pool with metrics.

Run: python3 -m pytest tests/test_thread_pool.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.thread_pool import ThreadPool, PoolShutdown, PoolFull, Future


class TestThreadPool:
    def test_create(self):
        pool = ThreadPool(max_workers=2)
        assert pool.stats()["max_workers"] == 2
        pool.shutdown()

    def test_submit_and_result(self):
        pool = ThreadPool(max_workers=2)
        future = pool.submit(lambda: 42)
        assert future.result(timeout=1.0) == 42
        pool.shutdown()

    def test_submit_exception(self):
        pool = ThreadPool(max_workers=2)
        future = pool.submit(lambda: (_ for _ in ()).throw(ValueError("boom")))
        with pytest.raises(ValueError):
            future.result(timeout=1.0)
        pool.shutdown()

    def test_done(self):
        pool = ThreadPool(max_workers=2)
        future = pool.submit(lambda: 42)
        future.result(timeout=1.0)
        assert future.done() is True
        pool.shutdown()

    def test_cancel(self):
        pool = ThreadPool(max_workers=2)
        # Submit a slow task, then cancel before it runs
        future = pool.submit(lambda: time.sleep(1.0))
        assert future.cancel() is True
        assert future.cancelled() is True
        pool.shutdown()

    def test_shutdown_reject(self):
        pool = ThreadPool(max_workers=2)
        pool.shutdown()
        with pytest.raises(PoolShutdown):
            pool.submit(lambda: 42)

    def test_queue_depth(self):
        pool = ThreadPool(max_workers=1, max_queue=2)
        pool.submit(lambda: time.sleep(0.1))
        pool.submit(lambda: time.sleep(0.1))
        # With 1 worker, at least one task should be in queue
        assert pool.queue_depth() >= 1
        pool.shutdown()

    def test_reject_policy_drop(self):
        pool = ThreadPool(max_workers=1, max_queue=1, reject_policy="drop")
        pool.submit(lambda: time.sleep(0.5))
        with pytest.raises(PoolFull):
            pool.submit(lambda: time.sleep(0.5))
        pool.shutdown()

    def test_reject_policy_block(self):
        pool = ThreadPool(max_workers=1, max_queue=1, reject_policy="block")
        pool.submit(lambda: time.sleep(0.1))
        future = pool.submit(lambda: 42, timeout=1.0)
        assert future.result(timeout=1.0) == 42
        pool.shutdown()

    def test_reject_policy_caller_runs(self):
        pool = ThreadPool(max_workers=1, max_queue=0, reject_policy="caller-runs")
        future = pool.submit(lambda: 42)
        assert future.result(timeout=1.0) == 42
        pool.shutdown()

    def test_active_count(self):
        pool = ThreadPool(max_workers=2)
        assert pool.active_count() == 2
        pool.shutdown()
        time.sleep(0.1)
        assert pool.active_count() == 0

    def test_stats(self):
        pool = ThreadPool(max_workers=2)
        pool.submit(lambda: 1)
        pool.submit(lambda: 2)
        pool.shutdown(wait=True)
        stats = pool.stats()
        assert stats["submitted"] == 2
        assert stats["completed"] == 2
        assert stats["shutdown"] is True

    def test_repr(self):
        pool = ThreadPool(max_workers=2)
        assert "ThreadPool" in repr(pool)
        pool.shutdown()
