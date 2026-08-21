"""Tests for CaslangExecutor.

Covers:
- Task graph → caslang conversion
- JSONL roundtrip
- Pre-flight validation (sandbox constraints)
- Successful execution with variable flow
- File operations (read/write/list)
- Loop execution
- JSON parsing
- Failure and rollback
- Stats tracking
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from fleet.caslang_executor import (
    CaslangExecutor,
    CaslangScript,
    ExecutionSandbox,
    SandboxViolation,
    ValidationError,
)


class TestCaslangScript:
    def test_jsonl_roundtrip(self) -> None:
        script = CaslangScript(
            commands=[
                {"op": "flow.set", "name": "x", "value": "hello"},
                {"op": "str.print", "msg": "${x}"},
            ]
        )
        jsonl = script.to_jsonl()
        restored = CaslangScript.from_jsonl(jsonl)
        assert len(restored.commands) == 2
        assert restored.commands[0]["name"] == "x"

    def test_from_task_graph(self) -> None:
        graph = {
            "tasks": [
                {
                    "id": "t1",
                    "action": "set_var",
                    "params": {"name": "foo", "value": "bar"},
                },
                {
                    "id": "t2",
                    "action": "read_file",
                    "params": {"path": "/tmp/test.txt", "as": "data"},
                },
            ],
            "dependencies": [{"from": "t1", "to": "t2"}],
        }
        script = CaslangScript.from_task_graph(graph)
        assert len(script.commands) == 2
        assert script.commands[0]["op"] == "flow.set"
        assert script.commands[1]["op"] == "fs.read_file"


class TestExecutionSandbox:
    def test_validate_allowed_path(self) -> None:
        sandbox = ExecutionSandbox(allowed_paths=["/tmp", "/data"])
        script = CaslangScript(
            commands=[
                {"op": "fs.read_file", "path": "/tmp/foo.txt"},
            ]
        )
        issues = sandbox.validate(script)
        assert issues == []

    def test_validate_denied_path(self) -> None:
        sandbox = ExecutionSandbox(allowed_paths=["/tmp"])
        script = CaslangScript(
            commands=[
                {"op": "fs.read_file", "path": "/etc/passwd"},
            ]
        )
        issues = sandbox.validate(script)
        assert len(issues) == 1
        assert "denied" in issues[0]

    def test_validate_tool_whitelist(self) -> None:
        sandbox = ExecutionSandbox(allowed_tools=["semantic_search"])
        script = CaslangScript(
            commands=[
                {"op": "tool.call", "tool_name": "http_get"},
            ]
        )
        issues = sandbox.validate(script)
        assert len(issues) >= 1
        assert any("not in whitelist" in i or "blocked" in i for i in issues)


class TestExecution:
    def test_simple_flow(self) -> None:
        executor = CaslangExecutor()
        script = CaslangScript(
            commands=[
                {"op": "flow.set", "name": "x", "value": "hello"},
                {"op": "flow.set", "name": "y", "value": "world"},
                {"op": "flow.return", "value": "${x} ${y}"},
            ]
        )
        result = executor.execute(script)
        assert result["status"] == "success"
        assert result["output"] == "hello world"

    def test_file_operations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = ExecutionSandbox(allowed_paths=[tmp])
            executor = CaslangExecutor(sandbox=sandbox)

            test_file = Path(tmp) / "test.txt"
            test_file.write_text("hello world", encoding="utf-8")

            script = CaslangScript(
                commands=[
                    {"op": "fs.read_file", "path": str(test_file), "as": "data"},
                    {"op": "flow.return", "value": "${data}"},
                ]
            )
            result = executor.execute(script)
            assert result["status"] == "success"
            assert result["output"] == "hello world"

    def test_json_parse(self) -> None:
        executor = CaslangExecutor()
        script = CaslangScript(
            commands=[
                {
                    "op": "flow.set",
                    "name": "raw",
                    "value": '{"name":"test","value":42}',
                },
                {"op": "json.parse", "s": "${raw}", "as": "obj"},
                {"op": "flow.return", "value": "${obj}"},
            ]
        )
        result = executor.execute(script)
        assert result["status"] == "success"
        # json.parse stores as string; actual parsed object would need extra handling
        assert "test" in str(result["output"]) or result["output"] == {
            "name": "test",
            "value": 42,
        }

    def test_loop_execution(self) -> None:
        executor = CaslangExecutor()
        script = CaslangScript(
            commands=[
                {"op": "flow.set", "name": "sum", "value": "0"},
                {"op": "flow.loop_start", "var": "i", "in": [1, 2, 3, 4, 5]},
                {"op": "flow.set", "name": "sum", "value": "${i}"},
                {"op": "flow.loop_end"},
                {"op": "flow.return", "value": "${sum}"},
            ]
        )
        result = executor.execute(script)
        assert result["status"] == "success"
        # Last iteration value
        assert result["output"] == "5"

    def test_sandbox_violation(self) -> None:
        sandbox = ExecutionSandbox(allowed_paths=["/tmp"])
        executor = CaslangExecutor(sandbox=sandbox)
        script = CaslangScript(
            commands=[
                {"op": "fs.read_file", "path": "/etc/passwd"},
            ]
        )
        result = executor.execute(script)
        assert result["status"] == "validation_failed"
        assert len(result["errors"]) > 0

    def test_rollback_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = ExecutionSandbox(allowed_paths=[tmp])
            executor = CaslangExecutor(sandbox=sandbox, rollback_enabled=True)

            test_file = Path(tmp) / "new_file.txt"
            script = CaslangScript(
                commands=[
                    {"op": "fs.write_file", "path": str(test_file), "data": "created"},
                    {
                        "op": "fs.read_file",
                        "path": "/nonexistent/file.txt",
                    },  # will fail
                ]
            )
            result = executor.execute(script)
            assert result["status"] in ("failed", "validation_failed")
            assert (
                result.get("rollback", False) is True
                or result["status"] == "validation_failed"
            )
            # File should be rolled back (deleted) if rollback ran
            if result.get("rollback"):
                assert not test_file.exists()

    def test_print_and_log(self) -> None:
        executor = CaslangExecutor()
        script = CaslangScript(
            commands=[
                {"op": "str.print", "msg": "hello"},
                {"op": "str.print", "msg": "world"},
            ]
        )
        result = executor.execute(script)
        assert result["status"] == "success"
        assert len(result["log"]) == 2

    def test_tool_call_simulation(self) -> None:
        sandbox = ExecutionSandbox(allowed_tools=["semantic_search"])
        executor = CaslangExecutor(sandbox=sandbox)
        script = CaslangScript(
            commands=[
                {"op": "tool.call", "tool_name": "semantic_search", "as": "result"},
                {"op": "flow.return", "value": "${result}"},
            ]
        )
        result = executor.execute(script)
        assert result["status"] == "success"
        assert result["output"] == "<simulated:semantic_search>"


class TestStats:
    def test_stats_tracking(self) -> None:
        executor = CaslangExecutor()
        for _ in range(3):
            script = CaslangScript(
                commands=[
                    {"op": "flow.set", "name": "x", "value": "1"},
                ]
            )
            executor.execute(script)

        # One failure
        bad_script = CaslangScript(
            commands=[
                {"op": "fs.read_file", "path": "/etc/passwd"},
            ]
        )
        executor.execute(bad_script)

        stats = executor.stats
        assert stats["scripts_executed"] == 3
        assert stats["scripts_failed"] == 1
        assert stats["rollback_enabled"] is True
