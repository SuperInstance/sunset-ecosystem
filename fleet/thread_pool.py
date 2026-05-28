"""thread_pool.py — Thread pool with metrics and rejection policy.

Provides:
1. Fixed-size thread pool
2. Task submission with future-like results
3. Queue depth monitoring
4. Rejection policy (drop, caller-runs, block)
5. Thread naming and daemon control

Usage:
    pool = ThreadPool(max_workers=4, reject_policy="block")
    future = pool.submit(lambda: heavy_work())
    result = future.result(timeout=5.0)
    pool.shutdown()
"""
from __future__ import annotations

__all__ = [
    "ThreadPool",
    "Future",
    "PoolShutdown",
    "PoolFull",
]

import logging
import threading
import time
from concurrent.futures import TimeoutError
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PoolShutdown(Exception):
    """Raised when submitting to a shut down pool."""


class PoolFull(Exception):
    """Raised when pool queue is full and reject_policy=drop."""


class Future(Generic[T]):
    """A simple future for thread pool results."""

    def __init__(self) -> None:
        self._result: T | None = None
        self._exception: Exception | None = None
        self._done = threading.Event()
        self._cancelled = False

    def set_result(self, result: T) -> None:
        self._result = result
        self._done.set()

    def set_exception(self, exc: Exception) -> None:
        self._exception = exc
        self._done.set()

    def result(self, timeout: float | None = None) -> T:
        if not self._done.wait(timeout):
            raise TimeoutError("Future timed out")
        if self._exception:
            raise self._exception
        return self._result  # type: ignore

    def done(self) -> bool:
        return self._done.is_set()

    def cancel(self) -> bool:
        if self._done.is_set():
            return False
        self._cancelled = True
        self._done.set()
        return True

    def cancelled(self) -> bool:
        return self._cancelled


class ThreadPool:
    """Fixed-size thread pool with metrics and rejection policy."""

    def __init__(
        self,
        max_workers: int = 4,
        max_queue: int = 100,
        reject_policy: str = "block",
        thread_name_prefix: str = "pool",
        daemon: bool = True,
    ) -> None:
        self._max_workers = max_workers
        self._max_queue = max_queue
        self._reject_policy = reject_policy
        self._thread_name_prefix = thread_name_prefix
        self._daemon = daemon
        self._queue: list[tuple[Callable[[], Any], Future]] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._shutdown = False
        self._workers: list[threading.Thread] = []
        self._submitted = 0
        self._completed = 0
        self._rejected = 0
        self._start_workers()

    def _start_workers(self) -> None:
        for i in range(self._max_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"{self._thread_name_prefix}-{i}",
                daemon=self._daemon,
            )
            t.start()
            self._workers.append(t)

    def submit(self, fn: Callable[[], T], timeout: float | None = None) -> Future[T]:
        """Submit a task. Returns a Future."""
        if self._shutdown:
            raise PoolShutdown("Pool is shut down")

        future: Future[T] = Future()
        with self._lock:
            if len(self._queue) >= self._max_queue:
                if self._reject_policy == "drop":
                    self._rejected += 1
                    raise PoolFull("Queue is full")
                elif self._reject_policy == "caller-runs":
                    # Run in caller thread
                    self._lock.release()
                    try:
                        result = fn()
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                    finally:
                        self._lock.acquire()
                    self._completed += 1
                    return future
                else:  # block
                    if timeout is not None:
                        if not self._not_full.wait(timeout=timeout):
                            self._rejected += 1
                            raise PoolFull("Queue is full, block timeout expired")
                    else:
                        while len(self._queue) >= self._max_queue:
                            self._not_full.wait()

            self._queue.append((fn, future))
            self._submitted += 1
            self._not_empty.notify()
        return future

    def _worker_loop(self) -> None:
        while True:
            with self._lock:
                while not self._queue and not self._shutdown:
                    self._not_empty.wait(timeout=1.0)
                if self._shutdown and not self._queue:
                    return
                if not self._queue:
                    continue
                fn, future = self._queue.pop(0)
                self._not_full.notify()

            if future.cancelled():
                continue
            try:
                result = fn()
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                with self._lock:
                    self._completed += 1

    def shutdown(self, wait: bool = True, timeout: float | None = None) -> None:
        """Shutdown the pool."""
        with self._lock:
            self._shutdown = True
            self._not_empty.notify_all()

        if wait:
            for t in self._workers:
                t.join(timeout=timeout)

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._queue)

    def active_count(self) -> int:
        return sum(1 for t in self._workers if t.is_alive())

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_workers": self._max_workers,
                "max_queue": self._max_queue,
                "queue_depth": len(self._queue),
                "submitted": self._submitted,
                "completed": self._completed,
                "rejected": self._rejected,
                "active_workers": self.active_count(),
                "shutdown": self._shutdown,
            }

    def __repr__(self) -> str:
        stats = self.stats()
        return f"ThreadPool(workers={stats['active_workers']}, queue={stats['queue_depth']})"
