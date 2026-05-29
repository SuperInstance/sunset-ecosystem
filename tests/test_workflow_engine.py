"""Tests for workflow_engine.py — DAG workflow executor.

Run: python3 -m pytest tests/test_workflow_engine.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.workflow_engine import WorkflowEngine, WorkflowError, WorkflowResult


class TestWorkflowEngine:
    def test_create(self):
        wf = WorkflowEngine()
        assert wf.step_count() == 0

    def test_add_step(self):
        wf = WorkflowEngine()
        wf.add_step("build", lambda ctx: "done")
        assert wf.step_count() == 1
        assert wf.dependencies("build") == set()

    def test_linear_run(self):
        wf = WorkflowEngine()
        results = []
        wf.add_step("a", lambda ctx: results.append("a"))
        wf.add_step("b", lambda ctx: results.append("b"), depends_on=["a"])
        wf.add_step("c", lambda ctx: results.append("c"), depends_on=["b"])
        r = wf.run()
        assert r.success is True
        assert results == ["a", "b", "c"]

    def test_diamond_run(self):
        wf = WorkflowEngine()
        results = []
        wf.add_step("a", lambda ctx: results.append("a"))
        wf.add_step("b", lambda ctx: results.append("b"), depends_on=["a"])
        wf.add_step("c", lambda ctx: results.append("c"), depends_on=["a"])
        wf.add_step("d", lambda ctx: results.append("d"), depends_on=["b", "c"])
        r = wf.run()
        assert r.success is True
        assert results[0] == "a"
        assert set(results[1:3]) == {"b", "c"}
        assert results[3] == "d"

    def test_failure_stop(self):
        wf = WorkflowEngine()
        wf.add_step("a", lambda ctx: None)
        wf.add_step("b", lambda ctx: (_ for _ in ()).throw(ValueError("boom")), depends_on=["a"])
        wf.add_step("c", lambda ctx: None, depends_on=["b"])
        r = wf.run(stop_on_failure=True)
        assert r.success is False
        assert r.step_results["b"].success is False
        assert r.step_results["c"].skipped is True

    def test_failure_continue(self):
        wf = WorkflowEngine()
        wf.add_step("a", lambda ctx: (_ for _ in ()).throw(ValueError("boom")))
        wf.add_step("b", lambda ctx: "ok", depends_on=["a"])
        r = wf.run(stop_on_failure=False)
        assert r.success is False
        assert r.step_results["b"].success is True

    def test_context_passed(self):
        wf = WorkflowEngine()
        wf.add_step("read", lambda ctx: ctx["x"] * 2)
        r = wf.run(context={"x": 21})
        assert r.step_results["read"].value == 42

    def test_timeout(self):
        wf = WorkflowEngine()
        wf.add_step("slow", lambda ctx: time.sleep(2), timeout=0.05)
        r = wf.run()
        assert r.success is False
        assert r.step_results["slow"].error == "Timeout exceeded"

    def test_missing_dependency(self):
        wf = WorkflowEngine()
        wf.add_step("a", lambda ctx: None, depends_on=["missing"])
        with pytest.raises(WorkflowError):
            wf.run()

    def test_cycle_detection(self):
        wf = WorkflowEngine()
        wf.add_step("a", lambda ctx: None, depends_on=["b"])
        wf.add_step("b", lambda ctx: None, depends_on=["a"])
        with pytest.raises(WorkflowError):
            wf.run()

    def test_step_result_fields(self):
        wf = WorkflowEngine()
        wf.add_step("fast", lambda ctx: "result")
        r = wf.run()
        sr = r.step_results["fast"]
        assert sr.name == "fast"
        assert sr.success is True
        assert sr.value == "result"
        assert sr.duration_sec >= 0

    def test_repr(self):
        wf = WorkflowEngine()
        wf.add_step("a", lambda ctx: None)
        assert "steps=1" in repr(wf)
