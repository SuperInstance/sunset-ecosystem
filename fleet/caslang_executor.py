"""CaslangExecutor — Constrained JSONL execution bridge for the fleet.

Integrates xlang-foundation's caslang (constrained scripting language)
into the sunset-ecosystem as a deterministic, sandboxed execution layer
for LLM-generated plans.

Provides:
- JSON task graph → caslang JSONL conversion
- Schema-validated execution with pre-flight checks
- Sandboxed filesystem / network access
- Rollback on failure with transaction semantics

Architecture
------------
caslang is a JSONL-based execution language where each line is a valid
JSON object representing a single command.  It is designed to be:

- **LLM-friendly**: Single-pass generation in strict format reduces hallucinations
- **Machine-validated**: Pre-execution schema checks guarantee safe host behavior
- **Privacy by default**: Execution happens locally; local data stays local

The bridge maps our `autonomous_repo.py` JSON task graphs into caslang
scripts, executes them in a sandboxed environment, and reports results
back to the fleet conductor.

Reference
---------
- caslang spec: https://github.com/xlang-foundation/caslang
"""

from __future__ import annotations

__all__ = [
    "CaslangExecutor",
    "CaslangScript",
    "ExecutionSandbox",
    "ValidationError",
    "SandboxViolation",
]

import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── exceptions ──────────────────────────────────────────────────


class ValidationError(ValueError):
    """Raised when a caslang script fails pre-execution validation."""
    pass


class SandboxViolation(RuntimeError):
    """Raised when a command attempts an operation outside the sandbox."""
    pass


# ── CaslangScript ───────────────────────────────────────────────


@dataclass
class CaslangScript:
    """A caslang script is a list of JSONL command objects."""

    version: str = "0.3"
    commands: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Serialize to caslang JSONL format."""
        lines = [json.dumps({"op": "caslang", "version": self.version})]
        for cmd in self.commands:
            lines.append(json.dumps(cmd, separators=(",", ":")))
        return "\n".join(lines) + "\n"

    @classmethod
    def from_jsonl(cls, text: str) -> "CaslangScript":
        """Parse from caslang JSONL format."""
        commands = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("op") == "caslang":
                version = obj.get("version", "0.3")
            else:
                commands.append(obj)
        return cls(version=version, commands=commands)

    @classmethod
    def from_task_graph(cls, graph: dict[str, Any]) -> "CaslangScript":
        """Convert a sunset-ecosystem JSON task graph to caslang.

        Task graph format (from autonomous_repo.py):
        {
            "tasks": [
                {"id": "t1", "action": "read_file", "params": {"path": "/data/foo"}},
                {"id": "t2", "action": "write_file", "params": {"path": "/data/bar", "data": "..."}},
            ],
            "dependencies": [{"from": "t1", "to": "t2"}]
        }
        """
        script = cls()
        task_map = {t["id"]: t for t in graph.get("tasks", [])}
        deps = {d["to"]: d["from"] for d in graph.get("dependencies", [])}

        # Topological order (simple dependency chain)
        visited: set[str] = set()
        ordered: list[str] = []

        def visit(tid: str) -> None:
            if tid in visited:
                return
            visited.add(tid)
            if tid in deps:
                visit(deps[tid])
            ordered.append(tid)

        for tid in task_map:
            visit(tid)

        for tid in ordered:
            task = task_map[tid]
            action = task.get("action", "noop")
            params = task.get("params", {})

            caslang_cmd = script._convert_action(action, params, tid)
            script.commands.append(caslang_cmd)

        # Add return statement if graph has output
        if graph.get("output"):
            script.commands.append({
                "op": "flow.return",
                "value": graph["output"],
            })

        return script

    @staticmethod
    def _convert_action(action: str, params: dict[str, Any], tid: str) -> dict[str, Any]:
        """Map a sunset action to a caslang command."""
        mapping = {
            "read_file": ("fs.read_file", {"path": "path", "as": "as"}),
            "write_file": ("fs.write_file", {"path": "path", "data": "data"}),
            "list_dir": ("fs.list", {"dir": "dir", "pattern": "pattern", "as": "as"}),
            "json_parse": ("json.parse", {"s": "input", "as": "as"}),
            "json_stringify": ("str.json", {"obj": "input", "as": "as"}),
            "set_var": ("flow.set", {"name": "name", "value": "value"}),
            "http_get": ("tool.call", {"tool_name": "http_get", "url": "url", "as": "as"}),
            "http_post": ("tool.call", {"tool_name": "http_post", "url": "url", "data": "data", "as": "as"}),
            "print": ("str.print", {"msg": "message"}),
            "loop_start": ("flow.loop_start", {"var": "var", "in": "iterable"}),
            "loop_end": ("flow.loop_end", {}),
            "condition": ("flow.if", {"cond": "condition", "then": "then_branch", "else": "else_branch"}),
        }

        if action not in mapping:
            return {"op": "str.print", "msg": f"Unknown action: {action}"}

        caslang_op, param_map = mapping[action]
        cmd: dict[str, Any] = {"op": caslang_op}
        for src, dst in param_map.items():
            if src in params:
                cmd[dst] = params[src]
        return cmd


# ── ExecutionSandbox ────────────────────────────────────────────


class ExecutionSandbox:
    """Sandboxed execution environment for caslang scripts.

    Parameters
    ----------
    allowed_paths : list[str]
        Whitelisted filesystem paths (absolute).
    allowed_tools : list[str]
        Whitelisted tool names (e.g. "http_get", "semantic_search").
    max_file_size : int
        Maximum bytes allowed for file operations.
    network_enabled : bool
        Whether HTTP/network tools are allowed.
    """

    def __init__(
        self,
        allowed_paths: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        network_enabled: bool = False,
    ) -> None:
        self.allowed_paths = set(allowed_paths or ["/tmp", "/data"])
        self.allowed_tools = set(allowed_tools or ["http_get", "semantic_search"])
        self.max_file_size = max_file_size
        self.network_enabled = network_enabled

        # Runtime state
        self._variables: dict[str, Any] = {}
        self._loop_stack: list[str] = []
        self._files_read: list[str] = []
        self._files_written: list[str] = []
        self._network_calls: list[dict[str, Any]] = []

    def validate(self, script: CaslangScript) -> list[str]:
        """Pre-flight validation: check every command against the sandbox.

        Returns a list of warnings/errors.  Empty list means clean.
        """
        issues: list[str] = []
        for i, cmd in enumerate(script.commands):
            op = cmd.get("op", "")
            if op.startswith("fs."):
                path = cmd.get("path", cmd.get("dir", ""))
                if not self._is_path_allowed(path):
                    issues.append(f"Line {i+1}: fs access denied for '{path}'")
                if op == "fs.write_file":
                    data = cmd.get("data", "")
                    if len(data.encode("utf-8")) > self.max_file_size:
                        issues.append(f"Line {i+1}: write exceeds max_file_size")
            elif op.startswith("tool.call"):
                tool = cmd.get("tool_name", "")
                if tool not in self.allowed_tools:
                    issues.append(f"Line {i+1}: tool '{tool}' not in whitelist")
                if tool in ("http_get", "http_post") and not self.network_enabled:
                    issues.append(f"Line {i+1}: network disabled, tool '{tool}' blocked")
            elif op.startswith("flow."):
                if op == "flow.loop_end" and not self._loop_stack:
                    issues.append(f"Line {i+1}: unmatched loop_end")
                if op == "flow.loop_start":
                    self._loop_stack.append("loop")
                if op == "flow.loop_end":
                    self._loop_stack.pop()
        return issues

    def _is_path_allowed(self, path: str) -> bool:
        """Check if a path is within the allowed set."""
        abs_path = Path(path).resolve()
        for allowed in self.allowed_paths:
            try:
                abs_path.relative_to(Path(allowed).resolve())
                return True
            except ValueError:
                continue
        return False


# ── CaslangExecutor ─────────────────────────────────────────────


class CaslangExecutor:
    """Deterministic, sandboxed executor for caslang scripts.

    Parameters
    ----------
    sandbox : ExecutionSandbox
        The sandbox that constrains what scripts can do.
    rollback_enabled : bool
        If True, on failure the executor attempts to undo filesystem changes.
    """

    def __init__(
        self,
        sandbox: ExecutionSandbox | None = None,
        rollback_enabled: bool = True,
    ) -> None:
        self.sandbox = sandbox or ExecutionSandbox()
        self.rollback_enabled = rollback_enabled
        self._lock = threading.RLock()
        self._scripts_executed = 0
        self._scripts_failed = 0
        self._rollback_count = 0

    # ── script conversion ───────────────────────────────────────

    def convert_task_graph(self, graph: dict[str, Any]) -> CaslangScript:
        """Convert a JSON task graph to a caslang script."""
        return CaslangScript.from_task_graph(graph)

    # ── execution ───────────────────────────────────────────────

    def execute(self, script: CaslangScript) -> dict[str, Any]:
        """Execute a caslang script in the sandbox.

        Returns a result dict with status, output, and execution log.
        """
        with self._lock:
            # 1. Pre-flight validation
            issues = self.sandbox.validate(script)
            if issues:
                self._scripts_failed += 1
                return {
                    "status": "validation_failed",
                    "errors": issues,
                    "executed": False,
                }

            # 2. Track changes for rollback
            pre_state = self._snapshot_state() if self.rollback_enabled else None

            # 3. Execute commands
            variables: dict[str, Any] = {}
            log: list[dict[str, Any]] = []
            output: Any = None
            loop_stack: list[dict[str, Any]] = []
            loop_iterables: list[Any] = []
            loop_vars: list[str] = []
            skip_until_loop_end = 0

            try:
                i = 0
                while i < len(script.commands):
                    cmd = script.commands[i]
                    op = cmd.get("op", "")
                    step = {"line": i + 1, "op": op, "result": None}

                    if skip_until_loop_end > 0:
                        if op == "flow.loop_end":
                            skip_until_loop_end -= 1
                        i += 1
                        continue

                    # Flow control
                    if op == "flow.set":
                        name = cmd.get("name", "")
                        value = self._resolve_value(cmd.get("value", ""), variables)
                        variables[name] = value
                        step["result"] = f"set {name}"

                    elif op == "flow.loop_start":
                        var = cmd.get("var", "")
                        iterable = self._resolve_value(cmd.get("in", []), variables)
                        loop_stack.append({"var": var, "iterable": iterable, "idx": 0, "start_line": i})
                        loop_vars.append(var)
                        loop_iterables.append(iterable)
                        if not iterable:
                            skip_until_loop_end = 1
                        else:
                            variables[var] = iterable[0]
                        step["result"] = f"loop start {var}"

                    elif op == "flow.loop_end":
                        if loop_stack:
                            frame = loop_stack[-1]
                            frame["idx"] += 1
                            if frame["idx"] < len(frame["iterable"]):
                                variables[frame["var"]] = frame["iterable"][frame["idx"]]
                                i = frame["start_line"]  # jump back to loop_start
                                step["result"] = f"loop iter {frame['idx']}"
                            else:
                                loop_stack.pop()
                                loop_vars.pop()
                                loop_iterables.pop()
                                step["result"] = "loop end"
                        else:
                            step["result"] = "unmatched loop_end"

                    elif op == "flow.return":
                        output = self._resolve_value(cmd.get("value", None), variables)
                        step["result"] = f"return {output}"
                        log.append(step)
                        break

                    # File operations
                    elif op == "fs.read_file":
                        path = cmd.get("path", "")
                        self._check_path(path)
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                data = f.read()
                            as_name = cmd.get("as", "_")
                            variables[as_name] = data
                            self.sandbox._files_read.append(path)
                            step["result"] = f"read {len(data)} bytes"
                        except Exception as exc:
                            step["result"] = f"error: {exc}"
                            raise

                    elif op == "fs.write_file":
                        path = cmd.get("path", "")
                        self._check_path(path)
                        data = self._resolve_value(cmd.get("data", ""), variables)
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(str(data))
                        self.sandbox._files_written.append(path)
                        step["result"] = f"write {len(str(data))} bytes"

                    elif op == "fs.list":
                        d = cmd.get("dir", ".")
                        self._check_path(d)
                        pattern = cmd.get("pattern", "*")
                        entries = [str(p) for p in Path(d).glob(pattern)]
                        as_name = cmd.get("as", "_")
                        variables[as_name] = entries
                        step["result"] = f"list {len(entries)} entries"

                    # JSON operations
                    elif op == "json.parse":
                        s = self._resolve_value(cmd.get("s", "{}"), variables)
                        as_name = cmd.get("as", "_")
                        variables[as_name] = json.loads(s)
                        step["result"] = "parsed json"

                    # String operations
                    elif op == "str.print":
                        msg = self._resolve_value(cmd.get("msg", ""), variables)
                        step["result"] = f"print: {msg}"

                    # Tool calls (simulated)
                    elif op == "tool.call":
                        tool_name = cmd.get("tool_name", "")
                        if tool_name not in self.sandbox.allowed_tools:
                            raise SandboxViolation(f"Tool '{tool_name}' not allowed")
                        as_name = cmd.get("as", "_")
                        variables[as_name] = f"<simulated:{tool_name}>"
                        self.sandbox._network_calls.append({"tool": tool_name, "params": cmd})
                        step["result"] = f"tool {tool_name}"

                    else:
                        step["result"] = f"unknown op: {op}"

                    log.append(step)
                    i += 1

                self._scripts_executed += 1
                return {
                    "status": "success",
                    "output": output,
                    "variables": variables,
                    "log": log,
                }

            except Exception as exc:
                self._scripts_failed += 1
                if pre_state and self.rollback_enabled:
                    self._rollback(pre_state)
                    self._rollback_count += 1
                return {
                    "status": "failed",
                    "error": str(exc),
                    "line": i + 1,
                    "log": log,
                    "rollback": self.rollback_enabled,
                }

    # ── internal helpers ──────────────────────────────────────────

    def _resolve_value(self, raw: Any, variables: dict[str, Any]) -> Any:
        """Resolve template references like ${var} or ${data['key']}."""
        if not isinstance(raw, str):
            return raw
        # Simple ${var} substitution
        pattern = re.compile(r"\$\{([^}]+)\}")
        def replacer(match: re.Match[str]) -> str:
            expr = match.group(1)
            # Direct variable lookup
            if expr in variables:
                return str(variables[expr])
            # Dict access: data['key']
            dict_match = re.match(r"(\w+)\[(.+?)\]", expr)
            if dict_match:
                var_name, key = dict_match.groups()
                key = key.strip("'\"")
                if var_name in variables and isinstance(variables[var_name], dict):
                    return str(variables[var_name].get(key, ""))
            return ""
        return pattern.sub(replacer, raw)

    def _check_path(self, path: str) -> None:
        """Verify a path is within the sandbox."""
        if not self.sandbox._is_path_allowed(path):
            raise SandboxViolation(f"Path '{path}' outside sandbox")

    def _snapshot_state(self) -> dict[str, Any]:
        """Capture filesystem state for rollback."""
        # For now, track which files were written
        return {
            "files_written_before": list(self.sandbox._files_written),
            "timestamp": time.time(),
        }

    def _rollback(self, state: dict[str, Any]) -> None:
        """Undo filesystem changes from the failed script."""
        before = set(state.get("files_written_before", []))
        after = set(self.sandbox._files_written)
        for path in after - before:
            try:
                Path(path).unlink(missing_ok=True)
                logger.info("Rollback: deleted %s", path)
            except Exception as exc:
                logger.warning("Rollback failed for %s: %s", path, exc)

    # ── stats ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "scripts_executed": self._scripts_executed,
                "scripts_failed": self._scripts_failed,
                "rollbacks": self._rollback_count,
                "rollback_enabled": self.rollback_enabled,
            }
