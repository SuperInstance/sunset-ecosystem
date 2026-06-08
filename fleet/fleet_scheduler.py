"""FleetScheduler — cron-like task scheduler for automated fleet operations.

Schedules recurring tasks: fleet beats, health checks, report generation,
metric collection, and benchmark runs. Supports cron expressions, intervals,
and one-shot tasks with persistent state.

Reference
---------
- Inspired by APScheduler and cron patterns
- Uses threading.Timer for lightweight scheduling
- Persists state to JSON file for recovery across restarts

Usage
-----
    scheduler = FleetScheduler()
    scheduler.add_interval_job("beat", interval_seconds=60)
    scheduler.add_cron_job("report", cron="0 9 * * *")  # 9 AM daily
    scheduler.start()
"""

from __future__ import annotations

__all__ = [
    "FleetScheduler",
    "ScheduledJob",
    "JobResult",
    "JobType",
]

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

from fleet.fleet_cli import FleetCLI


class JobType(Enum):
    """Types of scheduled jobs."""

    INTERVAL = auto()
    CRON = auto()
    ONESHOT = auto()


@dataclass
class ScheduledJob:
    """A scheduled job definition."""

    job_id: str
    name: str
    job_type: JobType
    interval_seconds: int | None = None
    cron_expression: str | None = None
    next_run: float | None = None
    last_run: float | None = None
    run_count: int = 0
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "job_type": self.job_type.name,
            "interval_seconds": self.interval_seconds,
            "cron_expression": self.cron_expression,
            "next_run": self.next_run,
            "last_run": self.last_run,
            "run_count": self.run_count,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScheduledJob:
        return cls(
            job_id=d["job_id"],
            name=d["name"],
            job_type=JobType[d["job_type"]],
            interval_seconds=d.get("interval_seconds"),
            cron_expression=d.get("cron_expression"),
            next_run=d.get("next_run"),
            last_run=d.get("last_run"),
            run_count=d.get("run_count", 0),
            enabled=d.get("enabled", True),
            metadata=d.get("metadata", {}),
        )


@dataclass
class JobResult:
    """Result of a scheduled job execution."""

    job_id: str
    success: bool
    message: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "success": self.success,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


class FleetScheduler:
    """Fleet task scheduler.

    Parameters
    ----------
    workspace : str
        Path to sunset-ecosystem workspace.
    state_file : str
        Path to state persistence file.
    """

    def __init__(
        self,
        workspace: str = ".",
        state_file: str = "fleet_scheduler_state.json",
    ) -> None:
        self.workspace = Path(workspace)
        self.state_file = self.workspace / state_file
        self._cli = FleetCLI(workspace=str(self.workspace))
        self._jobs: dict[str, ScheduledJob] = {}
        self._timers: dict[str, threading.Timer] = {}
        self._results: list[JobResult] = []
        self._running = False
        self._lock = threading.RLock()
        self._load_state()

    # ── State Management ──────────────────────────────────────

    def _load_state(self) -> None:
        """Load persisted state."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                for job_data in data.get("jobs", []):
                    job = ScheduledJob.from_dict(job_data)
                    self._jobs[job.job_id] = job
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_state(self) -> None:
        """Persist state to file."""
        data = {
            "jobs": [j.to_dict() for j in self._jobs.values()],
            "results": [r.to_dict() for r in self._results[-100:]],  # Last 100
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    # ── Job Registration ─────────────────────────────────────

    def add_interval_job(
        self,
        name: str,
        interval_seconds: int,
        job_id: str | None = None,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> ScheduledJob:
        """Add an interval-based job.

        Parameters
        ----------
        name : str
            Job name (must be a valid CLI command).
        interval_seconds : int
            Interval between runs.
        job_id : str | None
            Optional job ID (auto-generated if not provided).
        enabled : bool
            Whether job is enabled.
        metadata : dict | None
            Additional metadata.

        Returns
        -------
        ScheduledJob
            The registered job.
        """
        job_id = job_id or f"{name}_{int(time.time())}"
        job = ScheduledJob(
            job_id=job_id,
            name=name,
            job_type=JobType.INTERVAL,
            interval_seconds=interval_seconds,
            next_run=time.time() + interval_seconds if enabled else None,
            enabled=enabled,
            metadata=metadata or {},
        )
        with self._lock:
            self._jobs[job_id] = job
        self._save_state()
        return job

    def add_cron_job(
        self,
        name: str,
        cron_expression: str,
        job_id: str | None = None,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> ScheduledJob:
        """Add a cron-based job.

        Parameters
        ----------
        name : str
            Job name.
        cron_expression : str
            Cron expression (simplified: "min hour day month dow").
        job_id : str | None
            Optional job ID.
        enabled : bool
            Whether job is enabled.
        metadata : dict | None
            Additional metadata.

        Returns
        -------
        ScheduledJob
            The registered job.
        """
        job_id = job_id or f"{name}_cron_{int(time.time())}"
        # Simple cron: calculate next run from expression
        next_run = self._parse_cron_next(cron_expression)
        job = ScheduledJob(
            job_id=job_id,
            name=name,
            job_type=JobType.CRON,
            cron_expression=cron_expression,
            next_run=next_run if enabled else None,
            enabled=enabled,
            metadata=metadata or {},
        )
        with self._lock:
            self._jobs[job_id] = job
        self._save_state()
        return job

    def add_oneshot_job(
        self,
        name: str,
        run_at: float,
        job_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ScheduledJob:
        """Add a one-shot job.

        Parameters
        ----------
        name : str
            Job name.
        run_at : float
            Unix timestamp to run at.
        job_id : str | None
            Optional job ID.
        metadata : dict | None
            Additional metadata.

        Returns
        -------
        ScheduledJob
            The registered job.
        """
        job_id = job_id or f"{name}_oneshot_{int(time.time())}"
        job = ScheduledJob(
            job_id=job_id,
            name=name,
            job_type=JobType.ONESHOT,
            next_run=run_at,
            enabled=True,
            metadata=metadata or {},
        )
        with self._lock:
            self._jobs[job_id] = job
        self._save_state()
        return job

    def remove_job(self, job_id: str) -> bool:
        """Remove a job.

        Parameters
        ----------
        job_id : str
            Job ID to remove.

        Returns
        -------
        bool
            True if removed, False if not found.
        """
        with self._lock:
            if job_id in self._jobs:
                # Cancel timer if running
                if job_id in self._timers:
                    self._timers[job_id].cancel()
                    del self._timers[job_id]
                del self._jobs[job_id]
                self._save_state()
                return True
            return False

    def enable_job(self, job_id: str) -> bool:
        """Enable a job."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].enabled = True
                self._save_state()
                return True
            return False

    def disable_job(self, job_id: str) -> bool:
        """Disable a job."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].enabled = False
                # Cancel timer
                if job_id in self._timers:
                    self._timers[job_id].cancel()
                    del self._timers[job_id]
                self._save_state()
                return True
            return False

    # ── Cron Parsing ──────────────────────────────────────────

    def _parse_cron_next(self, cron: str) -> float:
        """Parse simplified cron and return next run timestamp.

        Supports: "min hour day month dow" (5 fields)
        """
        try:
            parts = cron.split()
            if len(parts) != 5:
                return time.time() + 3600  # Default: 1 hour

            minute, hour, day, month, dow = parts
            now = time.localtime()

            # Simple implementation: if minute is a number, schedule for next occurrence
            if minute.isdigit():
                target_min = int(minute)
                target_hour = int(hour) if hour.isdigit() else now.tm_hour

                # Calculate next occurrence
                next_time = time.mktime(
                    (now.tm_year, now.tm_mon, now.tm_mday, target_hour, target_min, 0, 0, 0, -1)
                )
                if next_time <= time.time():
                    next_time += 86400  # Add 1 day
                return next_time

            return time.time() + 3600  # Default: 1 hour
        except (ValueError, IndexError):
            return time.time() + 3600

    # ── Execution ───────────────────────────────────────────────

    def _execute_job(self, job_id: str) -> None:
        """Execute a scheduled job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or not job.enabled:
                return

        start = time.time()
        try:
            # Map job names to CLI methods
            command_map: dict[str, Callable[[], Any]] = {
                "beat": lambda: self._cli.beat(),
                "health": lambda: self._cli.health(),
                "status": lambda: self._cli.status(),
                "modules": lambda: self._cli.modules(),
                "metrics": lambda: self._cli.metrics(collect=True),
                "report": lambda: self._cli.report(),
            }

            cmd = command_map.get(job.name)
            if cmd:
                result = cmd()
                job_result = JobResult(
                    job_id=job_id,
                    success=result.success,
                    message=result.message,
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                job_result = JobResult(
                    job_id=job_id,
                    success=False,
                    message=f"Unknown command: {job.name}",
                    duration_ms=(time.time() - start) * 1000,
                )

            with self._lock:
                self._results.append(job_result)
                job.last_run = time.time()
                job.run_count += 1

                # Schedule next run
                if job.job_type == JobType.INTERVAL and job.interval_seconds:
                    job.next_run = time.time() + job.interval_seconds
                elif job.job_type == JobType.CRON and job.cron_expression:
                    job.next_run = self._parse_cron_next(job.cron_expression)
                elif job.job_type == JobType.ONESHOT:
                    job.enabled = False
                    job.next_run = None

                self._save_state()

                # Reschedule if still enabled
                if job.enabled and job.next_run:
                    self._schedule_job(job_id)

        except Exception as e:
            job_result = JobResult(
                job_id=job_id,
                success=False,
                message=str(e),
                duration_ms=(time.time() - start) * 1000,
            )
            with self._lock:
                self._results.append(job_result)
                self._save_state()

    def _schedule_job(self, job_id: str) -> None:
        """Schedule a job timer."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or not job.enabled or not job.next_run:
                return

            delay = max(0, job.next_run - time.time())
            timer = threading.Timer(delay, self._execute_job, args=[job_id])
            timer.daemon = True
            self._timers[job_id] = timer
            timer.start()

    # ── Lifecycle ───────────────────────────────────────────────

    def start(self) -> None:
        """Start the scheduler."""
        self._running = True
        with self._lock:
            for job_id, job in self._jobs.items():
                if job.enabled and job.next_run:
                    self._schedule_job(job_id)

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    # ── Query ───────────────────────────────────────────────────

    def list_jobs(self) -> list[ScheduledJob]:
        """List all jobs."""
        with self._lock:
            return list(self._jobs.values())

    def get_job(self, job_id: str) -> ScheduledJob | None:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_results(self, job_id: str | None = None, limit: int = 100) -> list[JobResult]:
        """Get job results.

        Parameters
        ----------
        job_id : str | None
            Filter by job ID. If None, returns all results.
        limit : int
            Maximum results to return.

        Returns
        -------
        list[JobResult]
            Job results.
        """
        with self._lock:
            results = self._results
            if job_id:
                results = [r for r in results if r.job_id == job_id]
            return results[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        with self._lock:
            total_jobs = len(self._jobs)
            enabled_jobs = sum(1 for j in self._jobs.values() if j.enabled)
            total_runs = sum(j.run_count for j in self._jobs.values())
            total_results = len(self._results)
            success_count = sum(1 for r in self._results if r.success)

        return {
            "total_jobs": total_jobs,
            "enabled_jobs": enabled_jobs,
            "total_runs": total_runs,
            "total_results": total_results,
            "success_rate": success_count / total_results if total_results > 0 else 1.0,
            "running": self._running,
        }
