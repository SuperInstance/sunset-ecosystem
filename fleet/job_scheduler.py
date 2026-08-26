from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """A scheduled job."""

    job_id: str
    name: str
    func: Callable
    args: tuple
    kwargs: dict
    scheduled_time: float
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    priority: int = 0
    retries: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "scheduled_time": self.scheduled_time,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "priority": self.priority,
            "retries": self.retries,
        }


class JobScheduler:
    """
    Job scheduler for breeding campaigns and fleet tasks.

    Supports priority queues, retries, and cron-like scheduling.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self.jobs: Dict[str, Job] = {}
        self._queue: List[str] = []
        self._completed: List[str] = []
        self._failed: List[str] = []

    def schedule(
        self,
        name: str,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        delay_seconds: float = 0.0,
        priority: int = 0,
    ) -> Job:
        """Schedule a job to run after a delay."""
        job_id = f"{name}_{int(time.time() * 1000000)}"
        job = Job(
            job_id=job_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs or {},
            scheduled_time=time.time() + delay_seconds,
            priority=priority,
        )
        self.jobs[job_id] = job
        self._queue.append(job_id)
        self._sort_queue()
        return job

    def schedule_immediate(
        self, name: str, func: Callable, args: tuple = (), kwargs: Optional[dict] = None
    ) -> Any:
        """Schedule and immediately execute a job."""
        job = self.schedule(name, func, args, kwargs, delay_seconds=0)
        return self.run_job(job.job_id)

    def _sort_queue(self):
        """Sort queue by scheduled time then priority."""
        self._queue.sort(
            key=lambda jid: (
                self.jobs[jid].scheduled_time,
                -self.jobs[jid].priority,
            )
        )

    def run_job(self, job_id: str) -> Any:
        """Execute a specific job."""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        job.status = JobStatus.RUNNING
        try:
            result = job.func(*job.args, **job.kwargs)
            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            self._completed.append(job_id)
            if job_id in self._queue:
                self._queue.remove(job_id)
            return result
        except Exception as e:
            job.error = str(e)
            job.retries += 1
            if job.retries >= job.max_retries:
                job.status = JobStatus.FAILED
                if job_id not in self._failed:
                    self._failed.append(job_id)
                if job_id in self._queue:
                    self._queue.remove(job_id)
                raise
            else:
                # Reschedule with backoff
                job.scheduled_time = time.time() + (2**job.retries)
                job.status = JobStatus.PENDING
                self._sort_queue()
                raise

    def run_pending(self) -> List[Dict[str, Any]]:
        """Run all pending jobs whose scheduled time has passed."""
        now = time.time()
        results = []
        ready = [
            jid
            for jid in self._queue
            if self.jobs[jid].status == JobStatus.PENDING
            and self.jobs[jid].scheduled_time <= now
        ]

        for job_id in ready:
            try:
                result = self.run_job(job_id)
                results.append(
                    {"job_id": job_id, "status": "completed", "result": result}
                )
            except Exception as e:
                results.append({"job_id": job_id, "status": "error", "error": str(e)})

        return results

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job."""
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.PENDING:
            return False
        job.status = JobStatus.CANCELLED
        if job_id in self._queue:
            self._queue.remove(job_id)
        return True

    def get_pending(self) -> List[Job]:
        """Get all pending jobs."""
        return [
            self.jobs[jid]
            for jid in self._queue
            if self.jobs[jid].status == JobStatus.PENDING
        ]

    def get_completed(self) -> List[Job]:
        """Get all completed jobs."""
        return [self.jobs[jid] for jid in self._completed]

    def get_failed(self) -> List[Job]:
        """Get all failed jobs."""
        return [self.jobs[jid] for jid in self._failed]

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "total": len(self.jobs),
            "pending": len(self.get_pending()),
            "completed": len(self._completed),
            "failed": len(self._failed),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.fleet_node_id,
            "stats": self.get_stats(),
        }
