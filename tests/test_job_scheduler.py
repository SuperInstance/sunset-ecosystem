import time
import pytest
from fleet.job_scheduler import Job, JobScheduler, JobStatus


class TestJob:
    def test_to_dict(self):
        j = Job(
            job_id="j1",
            name="test",
            func=lambda: None,
            args=(),
            kwargs={},
            scheduled_time=0.0,
        )
        d = j.to_dict()
        assert d["job_id"] == "j1"
        assert d["status"] == "pending"


class TestJobScheduler:
    def test_init(self):
        js = JobScheduler()
        assert js.fleet_node_id == "default"
        assert js.jobs == {}

    def test_schedule(self):
        js = JobScheduler()
        job = js.schedule("test", lambda: 42, delay_seconds=10)
        assert job.name == "test"
        assert job.status == JobStatus.PENDING
        assert job.scheduled_time > time.time()

    def test_schedule_immediate(self):
        js = JobScheduler()
        result = js.schedule_immediate("test", lambda: 42)
        assert result == 42
        assert len(js._completed) == 1

    def test_run_job(self):
        js = JobScheduler()
        job = js.schedule("test", lambda: 42, delay_seconds=0)
        result = js.run_job(job.job_id)
        assert result == 42
        assert job.status == JobStatus.COMPLETED

    def test_run_job_failure(self):
        js = JobScheduler()
        job = js.schedule("test", lambda: (_ for _ in ()).throw(ValueError("boom")), delay_seconds=0)
        with pytest.raises(ValueError):
            js.run_job(job.job_id)
        assert job.retries == 1
        assert job.status == JobStatus.PENDING  # Retried

    def test_run_job_max_retries(self):
        js = JobScheduler()
        job = js.schedule(
            "test",
            lambda: (_ for _ in ()).throw(ValueError("boom")),
            delay_seconds=0,
        )
        job.max_retries = 1
        with pytest.raises(ValueError):
            js.run_job(job.job_id)
        with pytest.raises(ValueError):
            js.run_job(job.job_id)
        assert job.status == JobStatus.FAILED

    def test_run_pending(self):
        js = JobScheduler()
        js.schedule("a", lambda: 1, delay_seconds=0)
        js.schedule("b", lambda: 2, delay_seconds=0)
        results = js.run_pending()
        assert len(results) == 2
        assert all(r["status"] == "completed" for r in results)

    def test_run_pending_empty(self):
        js = JobScheduler()
        results = js.run_pending()
        assert results == []

    def test_cancel(self):
        js = JobScheduler()
        job = js.schedule("test", lambda: 42, delay_seconds=10)
        assert js.cancel(job.job_id) is True
        assert job.status == JobStatus.CANCELLED

    def test_cancel_missing(self):
        js = JobScheduler()
        assert js.cancel("missing") is False

    def test_get_pending(self):
        js = JobScheduler()
        job = js.schedule("test", lambda: 42, delay_seconds=10)
        pending = js.get_pending()
        assert len(pending) == 1

    def test_get_completed(self):
        js = JobScheduler()
        js.schedule_immediate("test", lambda: 42)
        completed = js.get_completed()
        assert len(completed) == 1

    def test_get_failed(self):
        js = JobScheduler()
        job = js.schedule(
            "test",
            lambda: (_ for _ in ()).throw(ValueError("boom")),
            delay_seconds=0,
        )
        job.max_retries = 1
        try:
            js.run_job(job.job_id)
        except:
            pass
        try:
            js.run_job(job.job_id)
        except:
            pass
        failed = js.get_failed()
        assert len(failed) == 1

    def test_get_stats(self):
        js = JobScheduler()
        js.schedule_immediate("a", lambda: 1)
        js.schedule("b", lambda: 2, delay_seconds=10)
        stats = js.get_stats()
        assert stats["total"] == 2
        assert stats["completed"] == 1
        assert stats["pending"] == 1

    def test_to_dict(self):
        js = JobScheduler()
        js.schedule_immediate("a", lambda: 1)
        d = js.to_dict()
        assert d["stats"]["completed"] == 1
