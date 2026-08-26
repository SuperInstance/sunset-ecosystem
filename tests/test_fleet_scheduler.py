"""Tests for FleetScheduler — cron-like task scheduler for fleet operations.

Reference: fleet/fleet_scheduler.py
"""

from __future__ import annotations

import time

import pytest

from fleet.fleet_scheduler import (
    FleetScheduler,
    JobResult,
    JobType,
    ScheduledJob,
)


class TestScheduledJob:
    def test_to_dict(self) -> None:
        job = ScheduledJob(
            job_id="test_1",
            name="beat",
            job_type=JobType.INTERVAL,
            interval_seconds=60,
            next_run=1000.0,
            last_run=500.0,
            run_count=5,
            enabled=True,
            metadata={"key": "val"},
        )
        d = job.to_dict()
        assert d["job_id"] == "test_1"
        assert d["name"] == "beat"
        assert d["job_type"] == "INTERVAL"
        assert d["interval_seconds"] == 60
        assert d["next_run"] == 1000.0
        assert d["last_run"] == 500.0
        assert d["run_count"] == 5
        assert d["enabled"] is True
        assert d["metadata"] == {"key": "val"}

    def test_from_dict(self) -> None:
        d = {
            "job_id": "test_2",
            "name": "health",
            "job_type": "CRON",
            "interval_seconds": None,
            "cron_expression": "0 9 * * *",
            "next_run": 2000.0,
            "last_run": None,
            "run_count": 0,
            "enabled": True,
            "metadata": {},
        }
        job = ScheduledJob.from_dict(d)
        assert job.job_id == "test_2"
        assert job.name == "health"
        assert job.job_type == JobType.CRON
        assert job.cron_expression == "0 9 * * *"
        assert job.next_run == 2000.0


class TestJobResult:
    def test_fields(self) -> None:
        result = JobResult(
            job_id="test",
            success=True,
            message="ok",
            duration_ms=100.0,
            timestamp=1000.0,
        )
        assert result.job_id == "test"
        assert result.success is True
        assert result.duration_ms == 100.0

    def test_to_dict(self) -> None:
        result = JobResult(
            job_id="test",
            success=True,
            message="ok",
            duration_ms=100.0,
            timestamp=1000.0,
        )
        d = result.to_dict()
        assert d["job_id"] == "test"
        assert d["success"] is True
        assert d["timestamp"] == 1000.0


class TestFleetScheduler:
    def test_init(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        assert scheduler.workspace.exists()
        assert not scheduler.is_running()

    def test_init_state_file(self, tmp_path) -> None:
        scheduler = FleetScheduler(
            workspace=str(tmp_path), state_file="custom_state.json"
        )
        assert scheduler.state_file.name == "custom_state.json"

    def test_add_interval_job(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_interval_job("beat", interval_seconds=60)
        assert job.name == "beat"
        assert job.job_type == JobType.INTERVAL
        assert job.interval_seconds == 60
        assert job.enabled is True
        assert job.next_run > time.time()

    def test_add_interval_job_custom_id(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_interval_job(
            "health", interval_seconds=30, job_id="my_health"
        )
        assert job.job_id == "my_health"

    def test_add_interval_job_disabled(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_interval_job("beat", interval_seconds=60, enabled=False)
        assert job.enabled is False
        assert job.next_run is None

    def test_add_cron_job(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_cron_job("report", cron_expression="0 9 * * *")
        assert job.name == "report"
        assert job.job_type == JobType.CRON
        assert job.cron_expression == "0 9 * * *"
        assert job.next_run > time.time()

    def test_add_oneshot_job(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        run_at = time.time() + 3600
        job = scheduler.add_oneshot_job("status", run_at=run_at)
        assert job.name == "status"
        assert job.job_type == JobType.ONESHOT
        assert job.next_run == run_at
        assert job.enabled is True

    def test_remove_job(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_interval_job("beat", interval_seconds=60)
        assert scheduler.remove_job(job.job_id) is True
        assert scheduler.get_job(job.job_id) is None

    def test_remove_job_not_found(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        assert scheduler.remove_job("nonexistent") is False

    def test_enable_disable_job(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_interval_job("beat", interval_seconds=60, enabled=False)
        assert job.enabled is False
        assert scheduler.enable_job(job.job_id) is True
        assert scheduler.get_job(job.job_id).enabled is True
        assert scheduler.disable_job(job.job_id) is True
        assert scheduler.get_job(job.job_id).enabled is False

    def test_list_jobs(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job1 = scheduler.add_interval_job("beat", interval_seconds=60)
        job2 = scheduler.add_interval_job("health", interval_seconds=30)
        jobs = scheduler.list_jobs()
        assert len(jobs) == 2
        assert any(j.name == "beat" for j in jobs)
        assert any(j.name == "health" for j in jobs)

    def test_get_job(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_interval_job("beat", interval_seconds=60)
        found = scheduler.get_job(job.job_id)
        assert found is not None
        assert found.name == "beat"

    def test_get_job_not_found(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        assert scheduler.get_job("nonexistent") is None

    def test_get_results(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        result = JobResult(
            job_id="test",
            success=True,
            message="ok",
            duration_ms=100.0,
        )
        scheduler._results.append(result)
        results = scheduler.get_results()
        assert len(results) == 1
        assert results[0].job_id == "test"

    def test_get_results_filtered(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        scheduler._results.append(
            JobResult(job_id="job1", success=True, message="ok", duration_ms=10.0)
        )
        scheduler._results.append(
            JobResult(job_id="job2", success=True, message="ok", duration_ms=20.0)
        )
        results = scheduler.get_results(job_id="job1")
        assert len(results) == 1
        assert results[0].job_id == "job1"

    def test_get_results_limit(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        for i in range(150):
            scheduler._results.append(
                JobResult(
                    job_id="test", success=True, message=f"run {i}", duration_ms=1.0
                )
            )
        results = scheduler.get_results(limit=10)
        assert len(results) == 10

    def test_get_stats(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        scheduler.add_interval_job("beat", interval_seconds=60)
        scheduler.add_interval_job("health", interval_seconds=30, enabled=False)
        stats = scheduler.get_stats()
        assert stats["total_jobs"] == 2
        assert stats["enabled_jobs"] == 1
        assert stats["total_runs"] == 0
        assert stats["running"] is False

    def test_parse_cron_next(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        next_run = scheduler._parse_cron_next("0 9 * * *")
        assert next_run > time.time()

    def test_parse_cron_next_invalid(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        next_run = scheduler._parse_cron_next("invalid")
        assert next_run > time.time()
        assert next_run <= time.time() + 3600 + 1

    def test_start_stop(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        scheduler.add_interval_job("beat", interval_seconds=3600)
        scheduler.start()
        assert scheduler.is_running()
        scheduler.stop()
        assert not scheduler.is_running()

    def test_state_persistence(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_interval_job(
            "beat", interval_seconds=60, metadata={"key": "val"}
        )
        job_id = job.job_id

        # Create new scheduler with same state file
        scheduler2 = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        loaded_job = scheduler2.get_job(job_id)
        assert loaded_job is not None
        assert loaded_job.name == "beat"
        assert loaded_job.interval_seconds == 60
        assert loaded_job.metadata == {"key": "val"}

    def test_execute_job_beat(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_interval_job("beat", interval_seconds=60)
        scheduler._execute_job(job.job_id)
        loaded = scheduler.get_job(job.job_id)
        assert loaded.run_count == 1
        assert loaded.last_run > 0

    def test_execute_job_health(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_interval_job("health", interval_seconds=60)
        scheduler._execute_job(job.job_id)
        loaded = scheduler.get_job(job.job_id)
        assert loaded.run_count == 1

    def test_execute_job_status(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_interval_job("status", interval_seconds=60)
        scheduler._execute_job(job.job_id)
        loaded = scheduler.get_job(job.job_id)
        assert loaded.run_count == 1

    def test_execute_job_modules(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_interval_job("modules", interval_seconds=60)
        scheduler._execute_job(job.job_id)
        loaded = scheduler.get_job(job.job_id)
        assert loaded.run_count == 1

    def test_execute_job_unknown(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_interval_job("unknown_cmd", interval_seconds=60)
        scheduler._execute_job(job.job_id)
        results = scheduler.get_results(job_id=job.job_id)
        assert len(results) == 1
        assert results[0].success is False
        assert "Unknown" in results[0].message

    def test_oneshot_job_executes_once(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_oneshot_job("beat", run_at=time.time() - 1)
        scheduler._execute_job(job.job_id)
        loaded = scheduler.get_job(job.job_id)
        assert loaded.run_count == 1
        assert loaded.enabled is False
        assert loaded.next_run is None

    def test_interval_job_reschedules(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_interval_job("beat", interval_seconds=60)
        original_next = job.next_run
        scheduler._execute_job(job.job_id)
        loaded = scheduler.get_job(job.job_id)
        assert loaded.next_run > original_next

    def test_job_result_duration(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_interval_job("beat", interval_seconds=60)
        scheduler._execute_job(job.job_id)
        results = scheduler.get_results(job_id=job.job_id)
        assert len(results) == 1
        assert results[0].duration_ms >= 0

    def test_disable_job_cancels_timer(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        job = scheduler.add_interval_job("beat", interval_seconds=60)
        scheduler.start()
        assert scheduler.disable_job(job.job_id) is True
        scheduler.stop()

    def test_multiple_jobs(self, tmp_path) -> None:
        scheduler = FleetScheduler(workspace=str(tmp_path), state_file="state.json")
        for name in ["beat", "health", "status", "modules"]:
            scheduler.add_interval_job(name, interval_seconds=60)
        assert len(scheduler.list_jobs()) == 4
