"""Tests for XLangRuntime — unified xlang-foundation runtime.

Reference: fleet/xlang_runtime.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fleet.xlang_runtime import (
    ConstraintResult,
    ExecutionTrace,
    FlowStep,
    VDBSnapshot,
    XLangRuntime,
)


class TestFlowStep:
    def test_fields(self) -> None:
        s = FlowStep(
            step_id="step-1",
            step_type="agent",
            input_data={},
            output_data={"result": "ok"},
            duration_ms=100.0,
            success=True,
        )
        assert s.step_id == "step-1"
        assert s.success is True
        assert s.error is None

    def test_failure(self) -> None:
        s = FlowStep(
            step_id="step-1",
            step_type="agent",
            input_data={},
            output_data={},
            duration_ms=50.0,
            success=False,
            error="boom",
        )
        assert s.success is False
        assert s.error == "boom"


class TestConstraintResult:
    def test_passed(self) -> None:
        c = ConstraintResult(rule_id="r1", rule_text="test", passed=True)
        assert c.passed is True
        assert c.violations == []
        assert c.rollback_applied is False

    def test_violations(self) -> None:
        c = ConstraintResult(
            rule_id="r1",
            rule_text="test",
            passed=False,
            violations=["a > b"],
            rollback_applied=True,
        )
        assert c.passed is False
        assert c.violations == ["a > b"]
        assert c.rollback_applied is True


class TestVDBSnapshot:
    def test_fields(self) -> None:
        v = VDBSnapshot(vector_count=10, pending_inserts=2, timestamp=123.0)
        assert v.vector_count == 10
        assert v.pending_inserts == 2


class TestExecutionTrace:
    def test_empty(self) -> None:
        t = ExecutionTrace(trace_id="test")
        assert t.step_count == 0
        assert t.total_duration_ms == 0.0
        assert t.constraint_pass_rate == 1.0
        assert t.failed_steps == []
        assert t.overall_success is True

    def test_with_steps(self) -> None:
        t = ExecutionTrace(trace_id="test")
        t.flow_steps.append(
            FlowStep(
                step_id="s1",
                step_type="agent",
                input_data={},
                output_data={},
                duration_ms=100.0,
                success=True,
            )
        )
        assert t.step_count == 1
        assert t.total_duration_ms == 0.0  # start_time and end_time not set

    def test_failed_steps(self) -> None:
        t = ExecutionTrace(trace_id="test")
        t.flow_steps.append(
            FlowStep(
                step_id="s1",
                step_type="agent",
                input_data={},
                output_data={},
                duration_ms=100.0,
                success=True,
            )
        )
        t.flow_steps.append(
            FlowStep(
                step_id="s2",
                step_type="agent",
                input_data={},
                output_data={},
                duration_ms=50.0,
                success=False,
                error="fail",
            )
        )
        assert len(t.failed_steps) == 1
        assert t.failed_steps[0].step_id == "s2"

    def test_constraint_pass_rate(self) -> None:
        t = ExecutionTrace(trace_id="test")
        t.constraints.append(
            ConstraintResult(rule_id="r1", rule_text="t1", passed=True)
        )
        t.constraints.append(
            ConstraintResult(rule_id="r2", rule_text="t2", passed=False)
        )
        assert t.constraint_pass_rate == 0.5

    def test_to_dict(self) -> None:
        t = ExecutionTrace(trace_id="test")
        t.start_time = 1.0
        t.end_time = 2.0
        t.flow_steps.append(
            FlowStep(
                step_id="s1",
                step_type="agent",
                input_data={},
                output_data={"x": 1},
                duration_ms=100.0,
                success=True,
            )
        )
        d = t.to_dict()
        assert d["trace_id"] == "test"
        assert d["total_duration_ms"] == 1000.0
        assert d["step_count"] == 1
        assert d["overall_success"] is True
        assert len(d["flow_steps"]) == 1


class TestXLangRuntime:
    def test_init(self) -> None:
        runtime = XLangRuntime()
        assert runtime._xlang is None
        assert runtime._caslang is None
        assert runtime._vdb is None
        assert runtime._trace is None
        assert runtime._flow_steps == []
        assert runtime._constraints == []
        assert runtime.enable_vdb is True
        assert runtime.enable_constraints is True

    def test_init_disabled(self) -> None:
        runtime = XLangRuntime(enable_vdb=False, enable_constraints=False)
        assert runtime.enable_vdb is False
        assert runtime.enable_constraints is False

    def test_load_flow_from_string(self) -> None:
        runtime = XLangRuntime()
        yaml = """
steps:
  - id: step1
    type: agent
    config:
      model: gpt-4
      prompt: Hello
"""
        result = runtime.load_flow_from_string(yaml)
        assert result is True
        assert len(runtime._flow_steps) > 0

    def test_load_flow_from_string_bad_yaml(self) -> None:
        runtime = XLangRuntime()
        result = runtime.load_flow_from_string("not: yaml: [")
        assert result is False

    def test_load_flow_missing_file(self) -> None:
        runtime = XLangRuntime()
        result = runtime.load_flow("/nonexistent/flow.yaml")
        assert result is False

    def test_load_constraints_from_string(self) -> None:
        runtime = XLangRuntime()
        jsonl = '{"rule_id": "r1", "type": "check", "condition": "x > 0"}\n'
        result = runtime.load_constraints_from_string(jsonl)
        assert result is True
        assert len(runtime._constraints) > 0

    def test_load_constraints_from_string_bad(self) -> None:
        runtime = XLangRuntime()
        result = runtime.load_constraints_from_string("not json")
        assert result is False

    def test_load_constraints_missing_file(self) -> None:
        runtime = XLangRuntime()
        result = runtime.load_constraints("/nonexistent/rules.jsonl")
        assert result is False

    def test_execute_empty(self) -> None:
        runtime = XLangRuntime()
        trace = runtime.execute()
        assert isinstance(trace, ExecutionTrace)
        assert trace.step_count == 0
        assert trace.overall_success is True

    def test_execute_flow(self) -> None:
        runtime = XLangRuntime()
        yaml = """
steps:
  - id: step1
    type: agent
    config:
      model: gpt-4
      prompt: Hello
  - id: step2
    type: decision
    config:
      condition: "true"
      true_branch: A
  - id: step3
    type: action
    config:
      action: save
"""
        runtime.load_flow_from_string(yaml)
        trace = runtime.execute()
        assert trace.step_count == 3
        assert trace.overall_success is True
        assert all(s.success for s in trace.flow_steps)

    def test_execute_with_context(self) -> None:
        runtime = XLangRuntime()
        yaml = """
steps:
  - id: step1
    type: agent
    config:
      model: gpt-4
      prompt: Hello
"""
        runtime.load_flow_from_string(yaml)
        trace = runtime.execute(context={"key": "value"})
        assert trace.step_count == 1

    def test_execute_step_types(self) -> None:
        runtime = XLangRuntime()
        yaml = """
steps:
  - id: step1
    type: agent
    config:
      model: gpt-4
      prompt: Hello
  - id: step2
    type: decision
    config:
      condition: "true"
  - id: step3
    type: action
    config:
      action: save
  - id: step4
    type: unknown
    config:
      x: 1
"""
        runtime.load_flow_from_string(yaml)
        trace = runtime.execute()
        assert trace.step_count == 4
        assert trace.flow_steps[0].step_type == "agent"
        assert trace.flow_steps[1].step_type == "decision"
        assert trace.flow_steps[2].step_type == "action"
        assert trace.flow_steps[3].step_type == "unknown"

    def test_get_trace(self) -> None:
        runtime = XLangRuntime()
        yaml = """
steps:
  - id: step1
    type: agent
    config:
      model: gpt-4
      prompt: Hello
"""
        runtime.load_flow_from_string(yaml)
        runtime.execute()
        trace = runtime.get_trace()
        assert trace is not None
        assert trace.step_count == 1

    def test_get_trace_before_execute(self) -> None:
        runtime = XLangRuntime()
        assert runtime.get_trace() is None

    def test_get_trace_json(self) -> None:
        runtime = XLangRuntime()
        yaml = """
steps:
  - id: step1
    type: agent
    config:
      model: gpt-4
      prompt: Hello
"""
        runtime.load_flow_from_string(yaml)
        runtime.execute()
        json_str = runtime.get_trace_json()
        data = json.loads(json_str)
        assert data["trace_id"].startswith("trace-")
        assert data["step_count"] == 1

    def test_get_stats(self) -> None:
        runtime = XLangRuntime()
        yaml = """
steps:
  - id: step1
    type: agent
    config:
      model: gpt-4
      prompt: Hello
"""
        runtime.load_flow_from_string(yaml)
        runtime.execute()
        stats = runtime.get_stats()
        assert stats["executions"] == 1
        assert stats["total_steps"] == 1
        assert stats["success_rate"] == 1.0
        assert stats["failed_steps"] == 0
        assert stats["total_duration_ms"] > 0

    def test_get_stats_empty(self) -> None:
        runtime = XLangRuntime()
        stats = runtime.get_stats()
        assert stats["executions"] == 0
        assert stats["success_rate"] == 0.0

    def test_execute_vdb_step_disabled(self) -> None:
        runtime = XLangRuntime(enable_vdb=False)
        yaml = """
steps:
  - id: step1
    type: vdb_insert
    config:
      vector: [1.0, 2.0, 3.0]
      metadata:
        key: value
"""
        runtime.load_flow_from_string(yaml)
        trace = runtime.execute()
        assert trace.step_count == 1
        assert trace.flow_steps[0].output_data.get("status") == "vdb_disabled"

    def test_execute_constraint_step_disabled(self) -> None:
        runtime = XLangRuntime(enable_constraints=False)
        yaml = """
steps:
  - id: step1
    type: constraint_check
    config:
      rules: []
"""
        runtime.load_flow_from_string(yaml)
        trace = runtime.execute()
        assert trace.step_count == 1
        assert trace.flow_steps[0].output_data.get("status") == "no_constraints"

    def test_context_propagation(self) -> None:
        runtime = XLangRuntime()
        yaml = """
steps:
  - id: step1
    type: action
    config:
      action: set_x
  - id: step2
    type: action
    config:
      action: use_x
"""
        runtime.load_flow_from_string(yaml)
        trace = runtime.execute(context={"initial": "value"})
        assert trace.step_count == 2
        # Each step should see previous context
        assert "initial" in trace.flow_steps[0].input_data
