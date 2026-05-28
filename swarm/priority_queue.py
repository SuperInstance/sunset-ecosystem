"""priority_queue.py — Priority queue for breeding task scheduling.

Heap-based priority queue with:
1. Priority levels (int, lower = higher priority)
2. FIFO ordering within same priority
3. Capacity limits with backpressure
4. Task peek/drop by predicate
5. Bulk enqueue/dequeue
6. Statistics: size, wait time distribution

Usage:
    pq = BreedingPriorityQueue(capacity=1000)
    pq.enqueue(task, priority=1)
    task = pq.dequeue()  # highest priority task
"""
from __future__ import annotations

__all__ = [
    "BreedingPriorityQueue",
    "QueuedTask",
]

import heapq
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(order=True)
class QueuedTask:
    """A task in the priority queue.
    
    Ordering: priority (ascending), then sequence (ascending) for FIFO.
    """
    priority: int
    sequence: int
    task_id: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)
    enqueued_at: float = field(compare=False, default_factory=time.time)


class BreedingPriorityQueue:
    """Priority queue for breeding tasks."""

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self._queue: list[QueuedTask] = []  # min-heap
        self._sequence = 0
        self._dropped = 0
        self._dequeued = 0

    # ── enqueue ────────────────────────────────────────

    def enqueue(self, payload: dict[str, Any], priority: int = 5) -> QueuedTask | None:
        """Enqueue a task. Returns task or None if at capacity."""
        if len(self._queue) >= self.capacity:
            self._dropped += 1
            return None

        self._sequence += 1
        task = QueuedTask(
            priority=priority,
            sequence=self._sequence,
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            payload=payload,
        )
        heapq.heappush(self._queue, task)
        return task

    def enqueue_many(self, items: list[tuple[dict[str, Any], int]]) -> list[QueuedTask | None]:
        """Bulk enqueue. Returns list of enqueued tasks (None for dropped)."""
        return [self.enqueue(payload, priority) for payload, priority in items]

    # ── dequeue ────────────────────────────────────────

    def dequeue(self) -> QueuedTask | None:
        """Dequeue highest priority task."""
        if not self._queue:
            return None
        self._dequeued += 1
        return heapq.heappop(self._queue)

    def dequeue_many(self, n: int) -> list[QueuedTask]:
        """Dequeue up to n tasks."""
        result = []
        for _ in range(n):
            task = self.dequeue()
            if task is None:
                break
            result.append(task)
        return result

    def peek(self) -> QueuedTask | None:
        """Peek at highest priority task without removing."""
        return self._queue[0] if self._queue else None

    # ── query ──────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._queue)

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def is_full(self) -> bool:
        return len(self._queue) >= self.capacity

    def priorities(self) -> dict[int, int]:
        """Count tasks by priority level."""
        counts: dict[int, int] = {}
        for task in self._queue:
            counts[task.priority] = counts.get(task.priority, 0) + 1
        return counts

    def wait_times(self) -> list[float]:
        """Current wait times for all queued tasks."""
        now = time.time()
        return [now - task.enqueued_at for task in self._queue]

    def mean_wait(self) -> float:
        waits = self.wait_times()
        return sum(waits) / len(waits) if waits else 0.0

    def max_wait(self) -> float:
        waits = self.wait_times()
        return max(waits) if waits else 0.0

    # ── filtering ──────────────────────────────────────

    def drop_where(self, predicate: Callable[[QueuedTask], bool]) -> int:
        """Remove tasks matching predicate. Returns count removed."""
        before = len(self._queue)
        self._queue = [t for t in self._queue if not predicate(t)]
        heapq.heapify(self._queue)
        return before - len(self._queue)

    def find(self, predicate: Callable[[QueuedTask], bool]) -> list[QueuedTask]:
        """Find tasks matching predicate."""
        return [t for t in self._queue if predicate(t)]

    # ── stats ─────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._queue),
            "capacity": self.capacity,
            "dropped": self._dropped,
            "dequeued": self._dequeued,
            "mean_wait": self.mean_wait(),
            "max_wait": self.max_wait(),
            "priorities": self.priorities(),
        }

    def __repr__(self) -> str:
        return f"BreedingPriorityQueue(size={len(self._queue)}, capacity={self.capacity})"
