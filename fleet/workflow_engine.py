"""Simple DAG-based workflow execution engine for fleet operations.

Executes ordered steps with dependency resolution, parallel batching,
retry policies, and timeout enforcement. Used for breeding pipelines,
deployment rollouts, and fleet-wide maintenance tasks.

Usage:
    wf = WorkflowEngine()
    wf.add_step("build", lambda ctx: compile_agent(ctx))
    wf.add_step("test", lambda ctx: run_tests(ctx), depends_on=["build"])
    wf.add_step("deploy", lambda ctx: push_agent(ctx), depends_on=["test"])
    result = wf.run(context={"agent_id": 42})
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class WorkflowError(Exception):
    pass


class WorkflowTimeoutError(WorkflowError):
    pass


@dataclass
class StepResult:
    """Result of a single workflow step."""

    name: str
    success: bool
    value: Any = None
    error: Optional[str] = None
    duration_sec: float = 0.0
    skipped: bool = False


@dataclass
class WorkflowResult:
    """Result of an entire workflow run."""

    success: bool
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    total_duration_sec: float = 0.0


class WorkflowEngine:
    """
    DAG workflow executor with parallel batching.

    Steps are organized into parallel-executable batches based on
    their dependency graph. Each batch runs sequentially; steps within
    a batch can run concurrently (though this implementation runs them
    sequentially for determinism).
    """

    def __init__(self):
        self._steps: Dict[str, Callable[[Any], Any]] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._timeouts: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_step(
        self,
        name: str,
        fn: Callable[[Any], Any],
        depends_on: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Add a workflow step."""
        self._steps[name] = fn
        self._dependencies[name] = set(depends_on or [])
        if timeout is not None:
            self._timeouts[name] = timeout

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, context: Any = None, stop_on_failure: bool = True) -> WorkflowResult:
        """Execute all steps in dependency order."""
        start = time.perf_counter()
        batches = self._resolve_batches()
        results: Dict[str, StepResult] = {}
        any_failed = False

        for batch in batches:
            for name in batch:
                if any_failed and stop_on_failure:
                    results[name] = StepResult(name=name, success=False, skipped=True)
                    continue

                step_start = time.perf_counter()
                try:
                    fn = self._steps[name]
                    timeout = self._timeouts.get(name)
                    if timeout is not None:
                        value = self._run_with_timeout(fn, context, timeout)
                    else:
                        value = fn(context)
                    results[name] = StepResult(
                        name=name,
                        success=True,
                        value=value,
                        duration_sec=time.perf_counter() - step_start,
                    )
                except WorkflowTimeoutError:
                    any_failed = True
                    results[name] = StepResult(
                        name=name,
                        success=False,
                        error="Timeout exceeded",
                        duration_sec=time.perf_counter() - step_start,
                    )
                except Exception as exc:
                    any_failed = True
                    results[name] = StepResult(
                        name=name,
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                        duration_sec=time.perf_counter() - step_start,
                    )

        return WorkflowResult(
            success=not any_failed,
            step_results=results,
            total_duration_sec=time.perf_counter() - start,
        )

    def _resolve_batches(self) -> List[List[str]]:
        """Kahn's algorithm for parallel batch grouping."""
        # Check for missing dependencies first
        all_steps = set(self._steps.keys())
        for deps in self._dependencies.values():
            for dep in deps:
                if dep not in all_steps:
                    raise WorkflowError(f"Missing dependency: {dep}")

        in_degree: Dict[str, int] = {name: 0 for name in self._steps}
        reverse: Dict[str, Set[str]] = {name: set() for name in self._steps}
        for name, deps in self._dependencies.items():
            in_degree[name] = len(deps)
            for dep in deps:
                reverse[dep].add(name)

        queue = [name for name in self._steps if in_degree[name] == 0]
        batches: List[List[str]] = []
        current: List[str] = []
        next_queue: List[str] = []

        while queue:
            name = queue.pop(0)
            current.append(name)
            for dependent in reverse[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    next_queue.append(dependent)

            if not queue:
                if current:
                    batches.append(current)
                queue = next_queue
                next_queue = []
                current = []

        if sum(len(b) for b in batches) != len(self._steps):
            raise WorkflowError("Cycle detected in workflow dependencies")
        return batches

    def _run_with_timeout(
        self,
        fn: Callable[[Any], Any],
        context: Any,
        timeout: float,
    ) -> Any:
        """Run fn with timeout using signal alarm (POSIX only)."""
        import signal

        old_handler = signal.signal(signal.SIGALRM, lambda _signum, _frame: (_ for _ in ()).throw(WorkflowTimeoutError()))
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            return fn(context)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def step_count(self) -> int:
        return len(self._steps)

    def dependencies(self, name: str) -> Set[str]:
        return set(self._dependencies.get(name, set()))

    def __repr__(self) -> str:
        return f"<WorkflowEngine steps={len(self._steps)}>"
