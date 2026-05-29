"""Priority task queue with delayed and scheduled tasks.

Implements a priority queue with support for delayed execution,
scheduled tasks, and task cancellation. Used for fleet job scheduling,
background processing, and deferred work.

Usage:
    queue = TaskQueue()
    queue.enqueue("task-1", priority=1, delay_sec=5)
    queue.enqueue("task-2", priority=0)
    task = queue.dequeue()  # task-2 first (higher priority)
"""
from __future__ import annotations

import heapq
import time
from typing import Any, Dict, List, Optional, Tuple


class TaskQueue:
    """
    Priority task queue with delayed scheduling.

    :param clock: Optional clock function for testing.
    """

    def __init__(self, clock: Optional[callable] = None):
        self._clock = clock or time.time
        self._queue: List[Tuple[float, int, str, Dict[str, Any]]] = []
        self._counter = 0
        self._cancelled: set = set()
        self._completed: List[str] = []
        self._max_size = 0

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue(
        self,
        task_id: str,
        priority: int = 0,
        delay_sec: float = 0.0,
        payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a task to the queue.

        :param task_id: Unique task identifier.
        :param priority: Lower number = higher priority (0 is highest).
        :param delay_sec: Delay before task becomes available.
        :param payload: Task payload data.
        :returns: True if enqueued, False if already exists.
        """
        # Check for duplicate
        for _, _, _, tid, _ in self._queue:
            if tid == task_id:
                return False

        available_at = self._clock() + delay_sec
        # Heap: (available_at, priority, counter, task_id, payload)
        # Lower available_at first, then lower priority first
        heapq.heappush(
            self._queue,
            (available_at, priority, self._counter, task_id, payload or {}),
        )
        self._counter += 1
        self._max_size = max(self._max_size, len(self._queue))
        return True

    # ------------------------------------------------------------------
    # Dequeue
    # ------------------------------------------------------------------

    def dequeue(self, block: bool = False, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Get the next available task.

        :param block: Whether to wait for a task (not implemented).
        :param timeout: Max seconds to wait if blocking.
        :returns: Task dict or None if no tasks available.
        """
        now = self._clock()

        while self._queue:
            available_at, priority, counter, task_id, payload = self._queue[0]
            if task_id in self._cancelled:
                heapq.heappop(self._queue)
                self._cancelled.discard(task_id)
                continue
            if available_at <= now:
                heapq.heappop(self._queue)
                self._completed.append(task_id)
                return {
                    "task_id": task_id,
                    "priority": priority,
                    "payload": payload,
                }
            return None

        return None

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task."""
        for _, _, _, tid, _ in self._queue:
            if tid == task_id:
                self._cancelled.add(task_id)
                return True
        return False

    def peek(self) -> Optional[Dict[str, Any]]:
        """Peek at the next available task without removing."""
        now = self._clock()
        for available_at, priority, counter, task_id, payload in self._queue:
            if task_id in self._cancelled:
                continue
            if available_at <= now:
                return {
                    "task_id": task_id,
                    "priority": priority,
                    "payload": payload,
                }
            return None
        return None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Get current queue size."""
        return len(self._queue) - len(self._cancelled)

    def is_empty(self) -> bool:
        return self.size() == 0

    def completed(self) -> List[str]:
        """List completed task IDs."""
        return list(self._completed)

    def pending(self) -> List[str]:
        """List pending task IDs."""
        result = []
        for _, _, _, task_id, _ in self._queue:
            if task_id not in self._cancelled:
                result.append(task_id)
        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "size": self.size(),
            "max_size": self._max_size,
            "completed": len(self._completed),
            "cancelled": len(self._cancelled),
        }

    def __repr__(self) -> str:
        return f"<TaskQueue size={self.size()} completed={len(self._completed)}>"