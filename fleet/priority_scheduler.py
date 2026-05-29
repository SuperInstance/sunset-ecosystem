"""Priority-based task scheduler with preemption support.

Schedules tasks by priority level, with optional preemption for higher-priority
work. Used for fleet job scheduling where critical tasks must interrupt
lower-priority background work.

Usage:
    sched = PriorityScheduler()
    sched.submit("cleanup", priority=1, fn=cleanup_task)
    sched.submit("alert", priority=10, fn=alert_task)
    task = sched.next()  # alert_task (priority 10)
"""
from __future__ import annotations

import heapq
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PrioritizedTask:
    """A task with priority for heap ordering."""

    priority: int
    seq: int = field(compare=True)
    task_id: str = field(compare=False)
    name: str = field(compare=False)
    fn: Callable = field(compare=False)
    args: tuple = field(compare=False, default_factory=tuple)
    kwargs: dict = field(compare=False, default_factory=dict)
    preemptible: bool = field(compare=False, default=True)


class PriorityScheduler:
    """
    Priority scheduler with preemption tracking.

    Higher priority values are executed first. Tasks with equal priority
    are FIFO by submission order.
    """

    def __init__(self):
        self._heap: List[PrioritizedTask] = []
        self._seq = 0
        self._running: Optional[PrioritizedTask] = None
        self._completed: int = 0

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(
        self,
        name: str,
        fn: Callable,
        priority: int = 0,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        preemptible: bool = True,
    ) -> str:
        """
        Submit a task.

        :param name: Human-readable task name.
        :param fn: Callable to execute.
        :param priority: Higher = more urgent.
        :param args: Positional args for fn.
        :param kwargs: Keyword args for fn.
        :param preemptible: Whether lower-priority tasks can be interrupted.
        :returns: task_id.
        """
        self._seq += 1
        task_id = f"task-{self._seq}-{uuid.uuid4().hex[:6]}"
        task = PrioritizedTask(
            priority=-priority,  # negate for max-heap via min-heap
            seq=self._seq,
            task_id=task_id,
            name=name,
            fn=fn,
            args=args,
            kwargs=kwargs or {},
            preemptible=preemptible,
        )
        heapq.heappush(self._heap, task)
        return task_id

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def next(self) -> Optional[PrioritizedTask]:
        """Pop and return the highest-priority ready task."""
        if not self._heap:
            return None
        return heapq.heappop(self._heap)

    def peek(self) -> Optional[PrioritizedTask]:
        """Return the highest-priority task without removing."""
        return self._heap[0] if self._heap else None

    def should_preempt(self, current_priority: int) -> bool:
        """
        Check if the running task should be preempted.

        :param current_priority: Priority of currently running task.
        """
        next_task = self.peek()
        if not next_task:
            return False
        # next_task.priority is negated
        return -next_task.priority > current_priority

    # ------------------------------------------------------------------
    # State tracking
    # ------------------------------------------------------------------

    def mark_running(self, task: PrioritizedTask) -> None:
        self._running = task

    def mark_completed(self, task: PrioritizedTask) -> None:
        if self._running and self._running.task_id == task.task_id:
            self._running = None
        self._completed += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def queue_size(self) -> int:
        return len(self._heap)

    def list_queue(self) -> List[Dict[str, Any]]:
        """Return snapshot of queued tasks."""
        return [
            {
                "task_id": t.task_id,
                "name": t.name,
                "priority": -t.priority,
                "preemptible": t.preemptible,
            }
            for t in sorted(self._heap)
        ]

    def stats(self) -> Dict[str, int]:
        return {
            "queued": len(self._heap),
            "completed": self._completed,
            "running": 1 if self._running else 0,
        }

    def __repr__(self) -> str:
        return f"<PriorityScheduler queued={len(self._heap)} completed={self._completed}>"
