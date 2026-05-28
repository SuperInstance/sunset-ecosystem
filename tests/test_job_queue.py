"""Tests for job_queue.py — Distributed job queue.

Run: python3 -m pytest tests/test_job_queue.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.job_queue import JobQueue, JobStatus


class TestJobQueue:
    def test_create(self):
        queue = JobQueue()
        assert queue.pending_count() == 0

    def test_submit(self):
        queue = JobQueue()
        job_id = queue.submit({"task": "breed"})
        assert job_id.startswith("job-")
        assert queue.pending_count() == 1

    def test_claim(self):
        queue = JobQueue()
        queue.submit({"task": "breed"})
        job = queue.claim("worker-1")
        assert job is not None
        assert job.status == JobStatus.RUNNING
        assert job.worker_id == "worker-1"

    def test_claim_empty(self):
        queue = JobQueue()
        job = queue.claim("worker-1")
        assert job is None

    def test_complete(self):
        queue = JobQueue()
        job_id = queue.submit({"task": "breed"})
        queue.claim("worker-1")
        assert queue.complete(job_id, result={"agents": ["a1"]}) is True
        job = queue.get(job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.result == {"agents": ["a1"]}

    def test_complete_not_running(self):
        queue = JobQueue()
        job_id = queue.submit({"task": "breed"})
        # Not claimed yet
        assert queue.complete(job_id) is False

    def test_fail_with_retry(self):
        queue = JobQueue()
        job_id = queue.submit({"task": "breed"}, max_retries=3)
        queue.claim("worker-1")
        assert queue.fail(job_id, error="timeout") is True
        job = queue.get(job_id)
        assert job.status == JobStatus.RETRYING
        assert job.retries == 1

    def test_fail_permanent(self):
        queue = JobQueue()
        job_id = queue.submit({"task": "breed"}, max_retries=0)
        queue.claim("worker-1")
        assert queue.fail(job_id, error="boom") is True
        job = queue.get(job_id)
        assert job.status == JobStatus.FAILED

    def test_retry_backoff(self):
        queue = JobQueue()
        job_id = queue.submit({"task": "breed"}, max_retries=2)
        queue.claim("worker-1")
        queue.fail(job_id, error="timeout")
        job = queue.get(job_id)
        assert job.retry_at > time.time()
        # Can't claim immediately due to backoff
        assert queue.claim("worker-2") is None

    def test_priority(self):
        queue = JobQueue()
        queue.submit({"task": "low"}, priority=10)
        queue.submit({"task": "high"}, priority=1)
        job = queue.claim("worker-1")
        assert job.payload["task"] == "high"

    def test_list_jobs(self):
        queue = JobQueue()
        queue.submit({"task": "a"})
        queue.submit({"task": "b"})
        jobs = queue.list_jobs()
        assert len(jobs) == 2

    def test_list_jobs_by_status(self):
        queue = JobQueue()
        j1 = queue.submit({"task": "a"})
        j2 = queue.submit({"task": "b"})
        queue.claim("worker-1")
        queue.complete(j1)
        pending = queue.list_jobs(status=JobStatus.PENDING)
        completed = queue.list_jobs(status=JobStatus.COMPLETED)
        assert len(pending) == 1
        assert len(completed) == 1

    def test_get(self):
        queue = JobQueue()
        job_id = queue.submit({"task": "breed"})
        job = queue.get(job_id)
        assert job is not None
        assert job.payload["task"] == "breed"

    def test_get_missing(self):
        queue = JobQueue()
        assert queue.get("missing") is None

    def test_stats(self):
        queue = JobQueue()
        queue.submit({"task": "a"})
        queue.submit({"task": "b"})
        stats = queue.stats()
        assert stats["total_jobs"] == 2
        assert stats["pending"] == 2

    def test_repr(self):
        queue = JobQueue()
        assert "JobQueue" in repr(queue)
