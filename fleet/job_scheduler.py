"""Cron-like job scheduling with intervals and one-off jobs.

Implements a lightweight job scheduler supporting one-off jobs, interval
scheduling, and named recurring jobs. Used for fleet maintenance tasks,
periodic health checks, and timed breeding operations.

Usage:
    scheduler = JobScheduler()
    scheduler.schedule("backup", interval_sec=3600, fn=lambda: run_backup())
    scheduler.schedule_once("alert", delay_sec=300, fn=lambda: send_alert())
    due = scheduler.tick()  # Returns list of jobs ready to run
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


class JobScheduler:
    """
    Lightweight job scheduler.

    :param clock: Optional clock function for testing.
    """

    def __init__(self, clock: Optional[callable] = None):
        self._clock = clock or time.time
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._once_jobs: List[Dict[str, Any]] = []
        self._executed: List[str] = []
        self._skipped: List[str] = []

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule(
        self,
        name: str,
        interval_sec: float,
        fn: Callable[[], Any],
        start_at: Optional[float] = None,
        max_runs: Optional[int] = None,
    ) -> bool:
        """
        Schedule a recurring job.

        :param name: Job identifier.
        :param interval_sec: Seconds between runs.
        :param fn: Function to execute.
        :param start_at: Absolute start time (now if None).
        :param max_runs: Maximum number of executions (unlimited if None).
        :returns: True if scheduled, False if name already exists.
        """
        if name in self._jobs:
            return False
        self._jobs[name] = {
            "interval": interval_sec,
            "fn": fn,
            "next_run": start_at or self._clock(),
            "runs": 0,
            "max_runs": max_runs,
        }
        return True

    def schedule_once(
        self,
        name: str,
        delay_sec: float,
        fn: Callable[[], Any],
    ) -> None:
        """Schedule a one-off job."""
        self._once_jobs.append({
            "name": name,
            "run_at": self._clock() + delay_sec,
            "fn": fn,
        })

    def unschedule(self, name: str) -> bool:
        """Remove a recurring job."""
        if name in self._jobs:
            del self._jobs[name]
            return True
        # Check one-off jobs
        before = len(self._once_jobs)
        self._once_jobs = [j for j in self._once_jobs if j["name"] != name]
        return len(self._once_jobs) < before

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def tick(self) -> List[Dict[str, Any]]:
        """
        Check for due jobs and execute them.

        :returns: List of execution results.
        """
        now = self._clock()
        results = []

        # Recurring jobs
        for name, job in list(self._jobs.items()):
            if job["next_run"] <= now:
                if job["max_runs"] is not None and job["runs"] >= job["max_runs"]:
                    continue
                try:
                    result = job["fn"]()
                    job["runs"] += 1
                    job["next_run"] = now + job["interval"]
                    self._executed.append(name)
                    results.append({"name": name, "type": "recurring", "result": result})
                except Exception as e:
                    self._skipped.append(name)
                    results.append({"name": name, "type": "recurring", "error": str(e)})

        # One-off jobs
        for job in list(self._once_jobs):
            if job["run_at"] <= now:
                try:
                    result = job["fn"]()
                    self._executed.append(job["name"])
                    results.append({"name": job["name"], "type": "once", "result": result})
                except Exception as e:
                    self._skipped.append(job["name"])
                    results.append({"name": job["name"], "type": "once", "error": str(e)})
                self._once_jobs.remove(job)

        return results

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def due_jobs(self) -> List[str]:
        """List jobs that are due now."""
        now = self._clock()
        due = []
        for name, job in self._jobs.items():
            if job["next_run"] <= now:
                if job["max_runs"] is None or job["runs"] < job["max_runs"]:
                    due.append(name)
        for job in self._once_jobs:
            if job["run_at"] <= now:
                due.append(job["name"])
        return due

    def job_names(self) -> List[str]:
        """List all scheduled job names."""
        return list(self._jobs.keys()) + [j["name"] for j in self._once_jobs]

    def next_run(self, name: str) -> Optional[float]:
        """Get next scheduled run time."""
        job = self._jobs.get(name)
        if job:
            return job["next_run"]
        for j in self._once_jobs:
            if j["name"] == name:
                return j["run_at"]
        return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "recurring": len(self._jobs),
            "one_off": len(self._once_jobs),
            "executed": len(self._executed),
            "skipped": len(self._skipped),
        }

    def __repr__(self) -> str:
        return f"<JobScheduler recurring={len(self._jobs)} one_off={len(self._once_jobs)}>"
