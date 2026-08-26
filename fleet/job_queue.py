"""job_queue.py — Distributed job queue with workers.

Provides:
1. FIFO job queue with priorities
2. Job status tracking (pending, running, completed, failed)
3. Retry with exponential backoff
4. Worker pool simulation
5. Job result storage

Usage:
    queue = JobQueue()
    job_id = queue.submit({"task": "breed", "params": {...}}, priority=1)
    job = queue.claim(worker_id="w1")
    queue.complete(job_id, result={"agents": ["a1", "a2"]})
"""

from __future__ import annotations

__all__ = [
    "JobQueue",
    "Job",
    "JobStatus",
]

import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Job:
    """A job in the queue."""

    job_id: str
    payload: dict[str, Any]
    priority: int
    status: str = JobStatus.PENDING
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    worker_id: str | None = None
    result: Any = None
    error: str = ""
    retries: int = 0
    max_retries: int = 3
    retry_at: float = 0.0


class JobQueue:
    """Distributed job queue with retry and result tracking."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        # Priority queue: (priority, created_at, job_id)
        self._pending: list[tuple[int, float, str]] = []
        self._counter = 0

    def submit(
        self,
        payload: dict[str, Any],
        priority: int = 0,
        max_retries: int = 3,
    ) -> str:
        """Submit a job to the queue."""
        self._counter += 1
        job_id = f"job-{self._counter}"
        job = Job(
            job_id=job_id,
            payload=payload,
            priority=priority,
            created_at=time.time(),
            max_retries=max_retries,
        )
        self._jobs[job_id] = job
        heapq.heappush(self._pending, (priority, job.created_at, job_id))
        return job_id

    def claim(self, worker_id: str) -> Job | None:
        """Claim the highest-priority pending job."""
        now = time.time()
        # Find next available job (respect retry_at)
        while self._pending:
            priority, created_at, job_id = heapq.heappop(self._pending)
            job = self._jobs.get(job_id)
            if job is None or job.status not in (JobStatus.PENDING, JobStatus.RETRYING):
                continue
            if now < job.retry_at:
                # Put it back, not ready yet
                heapq.heappush(self._pending, (priority, created_at, job_id))
                return None
            job.status = JobStatus.RUNNING
            job.started_at = now
            job.worker_id = worker_id
            return job
        return None

    def complete(self, job_id: str, result: Any = None) -> bool:
        """Mark a job as completed."""
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.RUNNING:
            return False
        job.status = JobStatus.COMPLETED
        job.finished_at = time.time()
        job.result = result
        return True

    def fail(self, job_id: str, error: str = "") -> bool:
        """Mark a job as failed (with retry logic)."""
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.RUNNING:
            return False

        job.retries += 1
        if job.retries <= job.max_retries:
            # Schedule retry with exponential backoff
            backoff = 2 ** (job.retries - 1)
            job.retry_at = time.time() + backoff
            job.status = JobStatus.RETRYING
            job.error = error
            heapq.heappush(self._pending, (job.priority, job.created_at, job_id))
            logger.info(f"Job {job_id} retrying in {backoff}s (attempt {job.retries})")
        else:
            job.status = JobStatus.FAILED
            job.finished_at = time.time()
            job.error = error
            logger.error(f"Job {job_id} failed permanently: {error}")
        return True

    def get(self, job_id: str) -> Job | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self, status: str | None = None) -> list[Job]:
        """List jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs

    def pending_count(self) -> int:
        """Count pending/retriable jobs."""
        return sum(
            1
            for j in self._jobs.values()
            if j.status in (JobStatus.PENDING, JobStatus.RETRYING)
        )

    def stats(self) -> dict[str, Any]:
        """Queue statistics."""
        statuses: dict[str, int] = {}
        for j in self._jobs.values():
            statuses[j.status] = statuses.get(j.status, 0) + 1
        return {
            "total_jobs": len(self._jobs),
            "pending": self.pending_count(),
            "status_breakdown": statuses,
        }

    def __repr__(self) -> str:
        return f"JobQueue(jobs={len(self._jobs)}, pending={self.pending_count()})"
