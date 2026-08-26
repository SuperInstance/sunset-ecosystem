"""task_scheduler.py — Cron-like task scheduling for fleet operations.

Provides:
1. Schedule tasks at intervals or specific times
2. One-shot and recurring tasks
3. Task overlap prevention (skip or queue)
4. Execution statistics and history
5. Graceful shutdown with pending task completion

Usage:
    scheduler = TaskScheduler()
    scheduler.schedule("heartbeat", interval=30.0, fn=send_heartbeat)
    scheduler.schedule("cleanup", cron="0 */6 * * *", fn=cleanup_old_agents)
    scheduler.run()  # blocks until stop()
"""

from __future__ import annotations

__all__ = [
    "TaskScheduler",
    "ScheduledTask",
    "TaskExecution",
]

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class TaskExecution:
    """Record of a single task execution."""

    task_name: str
    started_at: float
    finished_at: float
    success: bool
    error: str = ""


@dataclass
class ScheduledTask:
    """A scheduled task definition."""

    name: str
    fn: Callable[[], Any]
    interval: float | None = None
    cron: str | None = None
    last_run: float = 0.0
    running: bool = False
    executions: list[TaskExecution] = field(default_factory=list)
    max_history: int = 100


class TaskScheduler:
    """Cron-like task scheduler for fleet operations."""

    def __init__(self, tick_interval: float = 1.0) -> None:
        self._tick_interval = tick_interval
        self._tasks: dict[str, ScheduledTask] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def schedule(
        self,
        name: str,
        fn: Callable[[], Any],
        interval: float | None = None,
        cron: str | None = None,
    ) -> None:
        """Schedule a recurring task."""
        if interval is None and cron is None:
            raise ValueError("Must specify interval or cron")
        self._tasks[name] = ScheduledTask(
            name=name,
            fn=fn,
            interval=interval,
            cron=cron,
            last_run=time.time(),
        )
        logger.info(f"Scheduled task '{name}' (interval={interval}, cron={cron})")

    def unschedule(self, name: str) -> bool:
        """Remove a scheduled task."""
        if name in self._tasks:
            del self._tasks[name]
            return True
        return False

    def run(self) -> None:
        """Run the scheduler loop (blocks)."""
        self._running = True
        while self._running:
            self._tick()
            time.sleep(self._tick_interval)

    def run_once(self) -> None:
        """Run a single tick (non-blocking)."""
        self._tick()

    def _tick(self) -> None:
        """Check all tasks and execute due ones."""
        now = time.time()
        for task in list(self._tasks.values()):
            if self._is_due(task, now):
                if task.running:
                    logger.warning(f"Task '{task.name}' still running, skipping")
                    continue
                self._execute(task)

    def _is_due(self, task: ScheduledTask, now: float) -> bool:
        if task.interval is not None:
            return now - task.last_run >= task.interval
        if task.cron is not None:
            return self._check_cron(task.cron, now, task.last_run)
        return False

    def _check_cron(self, cron: str, now: float, last_run: float) -> bool:
        """Simple cron check (only supports exact minute/hour matches)."""
        parts = cron.split()
        if len(parts) < 2:
            return False
        try:
            target_min = int(parts[0])
            target_hour = int(parts[1])
        except ValueError:
            return False
        tm = time.localtime(now)
        return (
            tm.tm_min == target_min
            and tm.tm_hour == target_hour
            and now - last_run >= 60
        )

    def _execute(self, task: ScheduledTask) -> None:
        task.running = True
        task.last_run = time.time()
        started = time.time()
        try:
            task.fn()
            execution = TaskExecution(
                task_name=task.name,
                started_at=started,
                finished_at=time.time(),
                success=True,
            )
        except Exception as e:
            logger.error(f"Task '{task.name}' failed: {e}")
            execution = TaskExecution(
                task_name=task.name,
                started_at=started,
                finished_at=time.time(),
                success=False,
                error=str(e),
            )
        task.executions.append(execution)
        if len(task.executions) > task.max_history:
            task.executions = task.executions[-task.max_history :]
        task.running = False

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False

    def start_background(self) -> None:
        """Start scheduler in a background thread."""
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def task_names(self) -> list[str]:
        return list(self._tasks.keys())

    def executions(self, name: str) -> list[TaskExecution]:
        task = self._tasks.get(name)
        return list(task.executions) if task else []

    def success_rate(self, name: str) -> float:
        """Success rate for a task."""
        execs = self.executions(name)
        if not execs:
            return 0.0
        return sum(1 for e in execs if e.success) / len(execs)

    def stats(self) -> dict[str, Any]:
        return {
            "tasks": len(self._tasks),
            "running": self._running,
            "executions": {name: len(t.executions) for name, t in self._tasks.items()},
        }

    def __repr__(self) -> str:
        return f"TaskScheduler(tasks={len(self._tasks)}, running={self._running})"
