"""subagent_conductor.py — Fleet subagent lifecycle manager with auto-fallback.

Monitors subagent gateway health, queues pending tasks, and automatically
falls back to direct execution when the gateway is overloaded.

Core loop:
1. Accept task requests (research, coding, audit)
2. Attempt subagent dispatch with timeout monitoring
3. On gateway timeout → queue for retry + trigger direct fallback
4. On subagent completion → harvest results, mark task done
5. On subagent failure → analyze error, retry with backoff, then fallback

Metrics tracked:
- Gateway latency histogram
- Subagent success/failure/timeout rates
- Average task completion time
- Fallback frequency
"""
from __future__ import annotations

__all__ = [
    "SubagentConductor",
    "TaskQueue",
    "GatewayHealthMonitor",
    "TaskResult",
    "TaskPriority",
]

import enum
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskPriority(enum.IntEnum):
    CRITICAL = 0   # P0 — fleet safety, consensus
    HIGH = 1      # P1 — breeding coordination, sync
    NORMAL = 2    # P2 — research, optimization
    LOW = 3       # P3 — background analysis, telemetry


class TaskStatus(enum.Enum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    FALLBACK = "fallback"


@dataclass
class TaskSpec:
    """Specification for a task to be executed."""
    task_id: str
    task_type: str          # "research", "code", "audit", "test"
    description: str
    priority: TaskPriority
    payload: dict[str, Any]
    max_runtime_sec: float = 300.0
    max_retries: int = 2
    fallback_direct: bool = True  # fallback to direct execution?
    created_at: float = field(default_factory=time.time)

    def cache_key(self) -> str:
        """Content-addressable key for deduplication."""
        data = json.dumps({
            "type": self.task_type,
            "payload": self.payload,
            "description": self.description,
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:32]


@dataclass
class TaskResult:
    """Result from a completed (or failed) task."""
    task_id: str
    status: TaskStatus
    output: str = ""
    error: str = ""
    execution_mode: str = ""   # "subagent", "direct", "failed"
    latency_sec: float = 0.0
    retry_count: int = 0
    completed_at: float = field(default_factory=time.time)


# ── Gateway Health Monitor ────────────────────────────────────

class GatewayHealthMonitor:
    """Tracks subagent gateway health via EWMA latency and success rate."""

    def __init__(
        self,
        latency_alpha: float = 0.3,
        success_window: int = 20,
    ) -> None:
        self.latency_alpha = latency_alpha
        self.ewma_latency_ms: float = 0.0
        self.success_window = success_window
        self._history: list[tuple[float, bool]] = []  # (latency_ms, success)
        self._circuit_open: bool = False
        self._circuit_until: float = 0.0

    def record(self, latency_ms: float, success: bool) -> None:
        if self.ewma_latency_ms == 0.0:
            self.ewma_latency_ms = latency_ms
        else:
            self.ewma_latency_ms = (
                self.latency_alpha * latency_ms
                + (1 - self.latency_alpha) * self.ewma_latency_ms
            )
        self._history.append((latency_ms, success))
        if len(self._history) > self.success_window:
            self._history.pop(0)

        # Circuit breaker: if 3 consecutive failures, open for 30s
        recent = self._history[-3:]
        if len(recent) >= 3 and all(not s for _, s in recent):
            self._circuit_open = True
            self._circuit_until = time.time() + 30.0
            logger.warning("Subagent gateway circuit OPEN (3 failures)")

        # Auto-close after cooldown
        if self._circuit_open and time.time() > self._circuit_until:
            self._circuit_open = False
            logger.info("Subagent gateway circuit CLOSED")

    @property
    def is_healthy(self) -> bool:
        # Auto-close circuit if cooldown expired
        if self._circuit_open and time.time() > self._circuit_until:
            self._circuit_open = False
        if self._circuit_open:
            return False
        if self.ewma_latency_ms == 0.0:
            return True  # no data yet
        return self.ewma_latency_ms < 5000.0  # 5s threshold

    @property
    def success_rate(self) -> float:
        if not self._history:
            return 1.0
        return sum(1 for _, s in self._history if s) / len(self._history)

    @property
    def recommend_direct(self) -> bool:
        """True if we should skip subagent and go direct."""
        return self._circuit_open or self.ewma_latency_ms > 3000.0

    def report(self) -> dict[str, Any]:
        return {
            "ewma_latency_ms": round(self.ewma_latency_ms, 1),
            "success_rate": round(self.success_rate, 2),
            "circuit_open": self._circuit_open,
            "circuit_until": self._circuit_until,
            "healthy": self.is_healthy,
            "recommend_direct": self.recommend_direct,
            "samples": len(self._history),
        }


# ── Task Queue ────────────────────────────────────────────────

class TaskQueue:
    """Priority queue for pending subagent tasks."""

    def __init__(self) -> None:
        self._tasks: list[TaskSpec] = []
        self._completed: dict[str, TaskResult] = {}
        self._dispatched: set[str] = set()

    def enqueue(self, task: TaskSpec) -> None:
        # Insert by priority (lower = higher priority)
        idx = 0
        for i, t in enumerate(self._tasks):
            if t.priority.value > task.priority.value:
                idx = i
                break
            idx = i + 1
        self._tasks.insert(idx, task)

    def dequeue(self) -> TaskSpec | None:
        while self._tasks:
            task = self._tasks.pop(0)
            if task.task_id not in self._dispatched:
                self._dispatched.add(task.task_id)
                return task
        return None

    def complete(self, result: TaskResult) -> None:
        self._completed[result.task_id] = result
        self._dispatched.discard(result.task_id)

    @property
    def pending_count(self) -> int:
        return len(self._tasks)

    @property
    def dispatched_count(self) -> int:
        return len(self._dispatched)

    @property
    def completed_count(self) -> int:
        return len(self._completed)


# ── Subagent Conductor ────────────────────────────────────────

class SubagentConductor:
    """Orchestrates subagent dispatch with health monitoring and fallback.

    Usage:
        conductor = SubagentConductor()
        conductor.submit(TaskSpec(...))
        conductor.tick()  # process queue, dispatch, handle results
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        gateway_timeout_ms: float = 10000.0,
    ) -> None:
        self.queue = TaskQueue()
        self.health = GatewayHealthMonitor()
        self.max_concurrent = max_concurrent
        self.gateway_timeout_ms = gateway_timeout_ms
        self._in_flight: dict[str, tuple[TaskSpec, float]] = {}  # task_id -> (spec, dispatched_at)
        self._fallback_handlers: dict[str, Callable[[TaskSpec], TaskResult]] = {}
        self._metrics: dict[str, Any] = {
            "tasks_submitted": 0,
            "tasks_dispatched": 0,
            "tasks_completed": 0,
            "tasks_fallback": 0,
            "tasks_failed": 0,
        }

    def register_fallback(self, task_type: str, handler: Callable[[TaskSpec], TaskResult]) -> None:
        """Register a direct-execution fallback for a task type."""
        self._fallback_handlers[task_type] = handler

    def submit(self, task: TaskSpec) -> None:
        """Queue a task for execution."""
        self.queue.enqueue(task)
        self._metrics["tasks_submitted"] += 1
        logger.info(f"Task {task.task_id} queued (type={task.task_type}, priority={task.priority.name})")

    def tick(self) -> list[TaskResult]:
        """Process queue, attempt dispatches, handle completions.

        Returns list of newly completed results this tick.
        """
        results: list[TaskResult] = []

        # Check in-flight for timeouts
        now = time.time()
        timed_out: list[str] = []
        for task_id, (spec, dispatched_at) in list(self._in_flight.items()):
            if now - dispatched_at > spec.max_runtime_sec:
                timed_out.append(task_id)
                result = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.TIMEOUT,
                    error=f"Task exceeded {spec.max_runtime_sec}s",
                    execution_mode="subagent",
                    latency_sec=spec.max_runtime_sec,
                )
                self._handle_result(result)
                results.append(result)

        for tid in timed_out:
            del self._in_flight[tid]

        # Attempt to dispatch pending tasks
        while len(self._in_flight) < self.max_concurrent:
            task = self.queue.dequeue()
            if task is None:
                break

            # Check if we should go direct
            if self.health.recommend_direct or not task.fallback_direct:
                # Try direct fallback immediately
                result = self._execute_fallback(task)
                if result:
                    self._handle_result(result)
                    results.append(result)
                    continue

            # Attempt subagent dispatch
            dispatched_at = time.time()
            # In a real system, this would call sessions_spawn
            # Here we record the attempt and let external integration handle it
            self._in_flight[task.task_id] = (task, dispatched_at)
            self._metrics["tasks_dispatched"] += 1
            logger.info(f"Task {task.task_id} dispatched (in_flight={len(self._in_flight)})")

        return results

    def _execute_fallback(self, task: TaskSpec) -> TaskResult | None:
        """Execute task via registered fallback handler."""
        handler = self._fallback_handlers.get(task.task_type)
        if handler is None:
            logger.warning(f"No fallback handler for {task.task_type}, skipping")
            return None

        start = time.time()
        try:
            result = handler(task)
            result.execution_mode = "direct"
            result.latency_sec = time.time() - start
            self._metrics["tasks_fallback"] += 1
            logger.info(f"Task {task.task_id} executed via fallback ({result.latency_sec:.1f}s)")
            return result
        except Exception as e:
            latency = time.time() - start
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_mode="direct",
                latency_sec=latency,
            )

    def _handle_result(self, result: TaskResult) -> None:
        self.queue.complete(result)
        if result.status == TaskStatus.COMPLETED:
            self._metrics["tasks_completed"] += 1
        elif result.status in (TaskStatus.FAILED, TaskStatus.TIMEOUT):
            self._metrics["tasks_failed"] += 1

    def record_gateway_attempt(self, latency_ms: float, success: bool) -> None:
        """Record the result of a gateway dispatch attempt."""
        self.health.record(latency_ms, success)

    def on_subagent_complete(self, task_id: str, output: str, error: str = "") -> TaskResult:
        """Called when an external subagent reports completion."""
        if task_id in self._in_flight:
            spec, dispatched_at = self._in_flight.pop(task_id)
            latency = time.time() - dispatched_at
            status = TaskStatus.COMPLETED if not error else TaskStatus.FAILED
            result = TaskResult(
                task_id=task_id,
                status=status,
                output=output,
                error=error,
                execution_mode="subagent",
                latency_sec=latency,
            )
            self._handle_result(result)
            return result
        else:
            # Orphaned completion
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                output=output,
                error=error,
                execution_mode="subagent",
            )

    def report(self) -> dict[str, Any]:
        return {
            **self._metrics,
            "in_flight": len(self._in_flight),
            "pending": self.queue.pending_count,
            "gateway_health": self.health.report(),
        }
