"""Tests for sandbox_runner.py — Isolated task execution.

Run: python3 -m pytest tests/test_sandbox_runner.py -v --tb=short
"""

from __future__ import annotations

import time

import pytest

from fleet.sandbox_runner import SandboxRunner


class TestSandboxRunner:
    def test_create(self):
        runner = SandboxRunner(timeout_sec=5, max_memory_mb=64)
        assert runner.stats()["timeout_sec"] == 5

    def test_run_echo(self):
        runner = SandboxRunner()
        result = runner.run("echo hello")
        assert result.stdout == "hello\n"
        assert result.returncode == 0
        assert result.timed_out is False

    def test_run_error(self):
        runner = SandboxRunner()
        result = runner.run("exit 1", shell=True)
        assert result.returncode == 1

    def test_run_timeout(self):
        runner = SandboxRunner(timeout_sec=0.05)
        result = runner.run("sleep 10", shell=True)
        assert result.timed_out is True
        assert result.returncode == -1

    def test_run_stderr(self):
        runner = SandboxRunner()
        result = runner.run("echo error >&2; exit 1", shell=True)
        assert "error" in result.stderr

    def test_repr(self):
        runner = SandboxRunner()
        assert "SandboxRunner" in repr(runner)
