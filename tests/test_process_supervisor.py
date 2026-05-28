"""Tests for process_supervisor.py — Process lifecycle management.

Run: python3 -m pytest tests/test_process_supervisor.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.process_supervisor import ProcessSupervisor, ProcessNotFound, ManagedProcess


class TestProcessSupervisor:
    def test_create(self):
        ps = ProcessSupervisor()
        assert ps.stats()["total"] == 0

    def test_start_and_stop(self):
        ps = ProcessSupervisor()
        proc = ps.start("sleep", cmd=["sleep", "10"])
        assert proc.status == "running"
        assert proc.process is not None
        assert ps.stop("sleep") is True
        assert ps.status("sleep")["status"] == "stopped"

    def test_stop_not_found(self):
        ps = ProcessSupervisor()
        with pytest.raises(ProcessNotFound):
            ps.stop("missing")

    def test_restart(self):
        ps = ProcessSupervisor()
        ps.start("sleep", cmd=["sleep", "10"])
        old_pid = ps.status("sleep")["pid"]
        ps.restart("sleep")
        time.sleep(0.2)
        new_pid = ps.status("sleep")["pid"]
        assert new_pid != old_pid

    def test_status(self):
        ps = ProcessSupervisor()
        ps.start("sleep", cmd=["sleep", "10"])
        status = ps.status("sleep")
        assert status["name"] == "sleep"
        assert status["status"] == "running"
        assert status["pid"] is not None
        ps.stop("sleep")

    def test_list_processes(self):
        ps = ProcessSupervisor()
        ps.start("a", cmd=["sleep", "10"])
        ps.start("b", cmd=["sleep", "10"])
        assert sorted(ps.list_processes()) == ["a", "b"]
        ps.stop("a")
        ps.stop("b")

    def test_auto_restart(self):
        ps = ProcessSupervisor(poll_interval=0.1)
        proc = ps.start(
            "short",
            cmd=["sleep", "0.1"],
            restart_policy="always",
            restart_delay=0.1,
        )
        time.sleep(0.5)
        assert proc.restart_count >= 2
        ps.stop("short")

    def test_restart_policy_never(self):
        ps = ProcessSupervisor(poll_interval=0.1)
        proc = ps.start(
            "short",
            cmd=["sleep", "0.1"],
            restart_policy="never",
        )
        time.sleep(0.3)
        assert proc.status == "exited"
        assert proc.restart_count == 1  # Only initial start

    def test_max_restarts(self):
        ps = ProcessSupervisor(poll_interval=0.1)
        proc = ps.start(
            "short",
            cmd=["sleep", "0.05"],
            restart_policy="always",
            max_restarts=2,
            restart_delay=0.05,
            restart_window=10.0,
        )
        time.sleep(0.5)
        assert proc.status == "failed" or proc.restart_count <= 3
        ps.shutdown()

    def test_health_check(self):
        healthy = [True, True, False]
        ps = ProcessSupervisor(poll_interval=0.1)
        proc = ps.start(
            "hc",
            cmd=["sleep", "10"],
            health_check=lambda: healthy.pop(0),
        )
        time.sleep(0.3)
        assert proc.status in ("running", "unhealthy")
        ps.stop("hc")

    def test_shutdown(self):
        ps = ProcessSupervisor()
        ps.start("a", cmd=["sleep", "10"])
        ps.start("b", cmd=["sleep", "10"])
        ps.shutdown()
        assert ps.status("a")["status"] == "stopped"
        assert ps.status("b")["status"] == "stopped"

    def test_stats(self):
        ps = ProcessSupervisor()
        ps.start("a", cmd=["sleep", "10"])
        stats = ps.stats()
        assert stats["total"] == 1
        assert stats["running"] == 1
        ps.stop("a")

    def test_repr(self):
        ps = ProcessSupervisor()
        assert "ProcessSupervisor" in repr(ps)
