"""Priority work queue with visibility timeout and dead-letter support.

Tasks are enqueued with a priority; workers claim them with a visibility
timeout. If a worker doesn't ack within the timeout, the task becomes
visible again. After max retries, tasks move to a dead-letter queue.

Usage:
    queue = WorkQueue(visibility_timeout=30, max_retries=3)
    queue.enqueue("breed_task", payload={"room_id": 42}, priority=1)

    item = queue.claim(worker_id="worker-1")
    if item:
        ...  # do work
        queue.ack(item.id)
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class WorkItem:
    id: str
    kind: str
    payload: Dict[str, Any]
    priority: int
    created_at: float
    retries: int = 0
    claimed_by: Optional[str] = None
    claimed_at: Optional[float] = None
    visible_at: float = field(default_factory=lambda: 0.0)
    acked: bool = False


class WorkQueue:
    """
    In-memory priority work queue.

    :param visibility_timeout: Seconds before an un-acked claim becomes visible.
    :param max_retries: Times a task can be re-claimed before dead-letter.
    """

    def __init__(
        self,
        visibility_timeout: float = 30.0,
        max_retries: int = 3,
        clock: Optional[Callable[[], float]] = None,
    ):
        self._visibility = visibility_timeout
        self._max_retries = max_retries
        self._clock = clock or time.monotonic
        self._items: Dict[str, WorkItem] = {}
        self._heap: List[Tuple[int, float, str]] = []  # (priority, created_at, id)
        self._dead_letter: List[WorkItem] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        kind: str,
        payload: Dict[str, Any],
        priority: int = 0,
        item_id: Optional[str] = None,
    ) -> str:
        """Add a task to the queue. Lower priority number = higher priority."""
        now = self._clock()
        item = WorkItem(
            id=item_id or str(uuid.uuid4()),
            kind=kind,
            payload=payload,
            priority=priority,
            created_at=now,
            visible_at=now,
        )
        with self._lock:
            self._items[item.id] = item
            heapq.heappush(self._heap, (priority, item.created_at, item.id))
            self._condition.notify()
        return item.id

    def size(self) -> int:
        """Count of items currently in the queue (not including dead-letter)."""
        with self._lock:
            return sum(
                1
                for item in self._items.values()
                if not item.acked and item.visible_at <= self._clock()
            )

    def pending_count(self) -> int:
        """Count of items currently claimed by workers."""
        now = self._clock()
        with self._lock:
            return sum(
                1
                for item in self._items.values()
                if item.claimed_by is not None
                and item.claimed_at is not None
                and (item.claimed_at + self._visibility) > now
                and not item.acked
            )

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def claim(
        self, worker_id: str, timeout: Optional[float] = None
    ) -> Optional[WorkItem]:
        """
        Claim the highest-priority visible task.

        :param worker_id: Identifier for the claiming worker.
        :param timeout: Seconds to wait for a task; None = block forever.
        :returns: A WorkItem, or None if timeout expired.
        """
        deadline = None if timeout is None else self._clock() + timeout
        with self._lock:
            while True:
                now = self._clock()
                item = self._try_claim(now, worker_id)
                if item is not None:
                    return item
                if deadline is not None and now >= deadline:
                    return None
                wait_time = None if deadline is None else max(0, deadline - now)
                self._condition.wait(wait_time)

    def _try_claim(self, now: float, worker_id: str) -> Optional[WorkItem]:
        """Attempt to claim the highest-priority visible item."""
        deferred: list = []
        while self._heap:
            priority, created_at, item_id = heapq.heappop(self._heap)
            item = self._items.get(item_id)
            if item is None or item.acked:
                continue
            if item.visible_at > now:
                deferred.append((priority, created_at, item_id))
                continue
            if (
                item.claimed_by is not None
                and (item.claimed_at + self._visibility) > now
            ):
                deferred.append((priority, created_at, item_id))
                continue
            # Found one — claim it and restore deferred items
            item.claimed_by = worker_id
            item.claimed_at = now
            item.visible_at = now + self._visibility
            heapq.heappush(self._heap, (priority, created_at, item.id))
            for d in deferred:
                heapq.heappush(self._heap, d)
            return item
        # Nothing available — restore all deferred items
        for d in deferred:
            heapq.heappush(self._heap, d)
        return None

    def ack(self, item_id: str) -> bool:
        """Acknowledge successful completion. Removes from queue."""
        with self._lock:
            item = self._items.get(item_id)
            if item is None or item.acked:
                return False
            item.acked = True
            self._items.pop(item_id, None)
            return True

    def nack(self, item_id: str) -> bool:
        """
        Negative acknowledgement — task failed, make visible again.

        After max_retries, moves to dead-letter queue.
        """
        with self._lock:
            item = self._items.get(item_id)
            if item is None or item.acked:
                return False
            item.retries += 1
            item.claimed_by = None
            item.claimed_at = None
            if item.retries >= self._max_retries:
                self._items.pop(item_id, None)
                self._dead_letter.append(item)
                return False
            item.visible_at = self._clock()
            self._condition.notify()
            return True

    def release(self, item_id: str) -> bool:
        """Release a claim without incrementing retry count."""
        with self._lock:
            item = self._items.get(item_id)
            if item is None or item.acked:
                return False
            item.claimed_by = None
            item.claimed_at = None
            item.visible_at = self._clock()
            self._condition.notify()
            return True

    # ------------------------------------------------------------------
    # Dead letter
    # ------------------------------------------------------------------

    def dead_letter_size(self) -> int:
        return len(self._dead_letter)

    def peek_dead_letter(self, n: int = 10) -> List[WorkItem]:
        return self._dead_letter[:n]

    def purge_dead_letter(self) -> int:
        count = len(self._dead_letter)
        self._dead_letter.clear()
        return count

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        now = self._clock()
        with self._lock:
            visible = sum(
                1
                for item in self._items.values()
                if not item.acked and item.visible_at <= now and item.claimed_by is None
            )
            claimed = sum(
                1
                for item in self._items.values()
                if item.claimed_by is not None
                and (item.claimed_at + self._visibility) > now
                and not item.acked
            )
            return {
                "visible": visible,
                "claimed": claimed,
                "dead_letter": len(self._dead_letter),
                "total_enqueued": len(self._items) + len(self._dead_letter),
            }

    def __repr__(self) -> str:
        return f"<WorkQueue visible={self.size()} dead={self.dead_letter_size()}>"
