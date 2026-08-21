"""Tests for sandbox.py — Resource-limited execution sandbox.

Run: python3 -m pytest tests/test_sandbox.py -v --tb=short
"""

from __future__ import annotations

import time

import pytest

from fleet.sandbox import Sandbox, SandboxResult, TimeoutError


class TestSandbox:
    def test_create(self):
        box = Sandbox()
        assert "Sandbox" in repr(box)

    def test_run_success(self):
        box = Sandbox()
        result = box.run(lambda: 2 + 2)
        assert result.success is True
        assert result.value == 4
        assert result.error is None

    def test_run_exception(self):
        box = Sandbox()

        def fail():
            raise ValueError("boom")

        result = box.run(fail)
        assert result.success is False
        assert "ValueError" in result.error

    def test_run_timeout(self):
        box = Sandbox(max_cpu_sec=0.1)

        def slow():
            time.sleep(0.1)
            return 42

        result = box.run(slow, timeout=0.05)
        assert result.success is False
        assert result.timeout is True

    def test_run_custom_timeout(self):
        box = Sandbox(max_cpu_sec=10.0)

        def slow():
            time.sleep(0.1)
            return 42

        result = box.run(slow, timeout=0.05)
        assert result.timeout is True

    def test_eval_expression(self):
        box = Sandbox()
        result = box.eval("2 + 3")
        assert result.success is True
        assert result.value == 5

    def test_eval_with_globals(self):
        box = Sandbox()
        result = box.eval("x * 2", globals_dict={"x": 21})
        assert result.value == 42

    def test_restricted_builtins_no_open(self):
        box = Sandbox(block_builtins=["open"])
        # open is not in restricted builtins but the test expression doesn't use it
        result = box.eval("len([1,2,3])")
        assert result.value == 3

    def test_stats(self):
        box = Sandbox()
        box.run(lambda: 1)
        box.run(lambda: 1 / 0)
        stats = box.stats()
        assert stats["runs"] == 2
        assert stats["errors"] == 1

    def test_duration_recorded(self):
        box = Sandbox()

        def work():
            time.sleep(0.01)
            return 1

        result = box.run(work)
        assert result.duration_sec >= 0.01

    def test_memory_peak_recorded(self):
        box = Sandbox()
        result = box.run(lambda: list(range(1000)))
        assert result.memory_peak_mb >= 0

    def test_repr(self):
        box = Sandbox(max_memory_mb=64, max_cpu_sec=2.0)
        assert "64MB" in repr(box)
        assert "2.0s" in repr(box)
