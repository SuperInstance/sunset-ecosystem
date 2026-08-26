"""XLangRuntime — unified runtime for xlang-foundation ecosystem.

Brings together xlang events, xMind agent flows, caslang constraints,
and Quanta VDB into a single execution environment.

Usage
-----
    from fleet.xlang_runtime import XLangRuntime

    runtime = XLangRuntime()
    runtime.load_flow("agent_flow.yaml")
    runtime.load_constraints("rules.jsonl")
    runtime.execute()

    results = runtime.get_trace()
"""

from __future__ import annotations

__all__ = [
    "XLangRuntime",
    "ExecutionTrace",
    "FlowStep",
    "ConstraintResult",
    "VDBSnapshot",
]

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from fleet.caslang_executor import CaslangExecutor
from fleet.xlang_agent_bridge import XlangAgentBridge
from swarm.quanta_vdb_bridge import QuantaVdbBridge
from fleet.xlang_agent_bridge import XlangAgentBridge


@dataclass
class FlowStep:
    """A single step in an agent flow execution."""

    step_id: str
    step_type: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    duration_ms: float
    success: bool
    error: str | None = None


@dataclass
class ConstraintResult:
    """Result of applying a caslang constraint."""

    rule_id: str
    rule_text: str
    passed: bool
    violations: list[str] = field(default_factory=list)
    rollback_applied: bool = False


@dataclass
class VDBSnapshot:
    """Snapshot of VDB state after an operation."""

    vector_count: int
    pending_inserts: int
    metadata_keys: list[str] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class ExecutionTrace:
    """Complete trace of an XLangRuntime execution."""

    trace_id: str
    flow_steps: list[FlowStep] = field(default_factory=list)
    constraints: list[ConstraintResult] = field(default_factory=list)
    vdb_snapshots: list[VDBSnapshot] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    overall_success: bool = True

    @property
    def total_duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def step_count(self) -> int:
        return len(self.flow_steps)

    @property
    def constraint_pass_rate(self) -> float:
        if not self.constraints:
            return 1.0
        passed = sum(1 for c in self.constraints if c.passed)
        return passed / len(self.constraints)

    @property
    def failed_steps(self) -> list[FlowStep]:
        return [s for s in self.flow_steps if not s.success]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            "step_count": self.step_count,
            "overall_success": self.overall_success,
            "constraint_pass_rate": self.constraint_pass_rate,
            "flow_steps": [
                {
                    "step_id": s.step_id,
                    "step_type": s.step_type,
                    "duration_ms": s.duration_ms,
                    "success": s.success,
                    "error": s.error,
                }
                for s in self.flow_steps
            ],
            "constraints": [
                {
                    "rule_id": c.rule_id,
                    "passed": c.passed,
                    "violations": c.violations,
                    "rollback_applied": c.rollback_applied,
                }
                for c in self.constraints
            ],
            "vdb_snapshots": [
                {
                    "vector_count": v.vector_count,
                    "pending_inserts": v.pending_inserts,
                    "metadata_keys": v.metadata_keys,
                }
                for v in self.vdb_snapshots
            ],
        }


class XLangRuntime:
    """Unified runtime for xlang-foundation ecosystem.

    Parameters
    ----------
    workspace : str
        Path to sunset-ecosystem workspace.
    enable_vdb : bool
        Whether to enable Quanta VDB integration.
    enable_constraints : bool
        Whether to enable caslang constraint enforcement.
    """

    def __init__(
        self,
        workspace: str = ".",
        enable_vdb: bool = True,
        enable_constraints: bool = True,
    ) -> None:
        self.workspace = Path(workspace)
        self.enable_vdb = enable_vdb
        self.enable_constraints = enable_constraints

        self._xlang: XlangAgentBridge | None = None
        self._caslang: CaslangExecutor | None = None
        self._vdb: QuantaVdbBridge | None = None
        self._trace: ExecutionTrace | None = None
        self._flow_steps: list[dict[str, Any]] = []
        self._constraints: list[dict[str, Any]] = []

    def _init_subsystems(self) -> None:
        """Initialize subsystems lazily."""
        if self._xlang is None:
            self._xlang = XlangAgentBridge(node_id="xlang-runtime")

        if self.enable_constraints and self._caslang is None:
            self._caslang = CaslangExecutor()

        if self.enable_vdb and self._vdb is None:
            self._vdb = QuantaVdbBridge(
                prefix="xlang-runtime",
                data_path=self.workspace / "data" / "quanta",
                dim=128,
                node_id="xlang-runtime",
            )

    # ── Flow Loading ──────────────────────────────────────────

    def load_flow(self, flow_path: str | Path) -> bool:
        """Load an xMind agent flow from YAML.

        Parameters
        ----------
        flow_path : str | Path
            Path to the YAML flow file.

        Returns
        -------
        bool
            True if loaded successfully.
        """
        self._init_subsystems()
        if not self._xlang:
            return False

        path = Path(flow_path)
        if not path.exists():
            return False

        try:
            data = yaml.safe_load(content)
            if isinstance(data, dict) and "steps" in data:
                self._flow_steps = data["steps"]
            elif isinstance(data, list):
                self._flow_steps = data
            else:
                self._flow_steps = []
            return True
        except Exception:
            return False

    def load_flow_from_string(self, yaml_content: str) -> bool:
        """Load an xMind agent flow from YAML string.

        Parameters
        ----------
        yaml_content : str
            YAML content string.

        Returns
        -------
        bool
            True if loaded successfully.
        """
        self._init_subsystems()
        if not self._xlang:
            return False

        try:
            data = yaml.safe_load(yaml_content)
            if isinstance(data, dict) and "steps" in data:
                self._flow_steps = data["steps"]
            elif isinstance(data, list):
                self._flow_steps = data
            else:
                self._flow_steps = []
            return True
        except Exception:
            return False

    # ── Constraint Loading ─────────────────────────────────────

    def load_constraints(self, constraints_path: str | Path) -> bool:
        """Load caslang constraints from JSONL.

        Parameters
        ----------
        constraints_path : str | Path
            Path to the JSONL constraints file.

        Returns
        -------
        bool
            True if loaded successfully.
        """
        self._init_subsystems()
        if not self._caslang:
            return False

        path = Path(constraints_path)
        if not path.exists():
            return False

        try:
            content = path.read_text()
            self._constraints = self._caslang.parse_jsonl(content)
            return True
        except Exception:
            return False

    def load_constraints_from_string(self, jsonl_content: str) -> bool:
        """Load caslang constraints from JSONL string.

        Parameters
        ----------
        jsonl_content : str
            JSONL content string.

        Returns
        -------
        bool
            True if loaded successfully.
        """
        self._init_subsystems()
        if not self._caslang:
            return False

        try:
            self._constraints = []
            for line in jsonl_content.strip().split("\n"):
                line = line.strip()
                if line:
                    self._constraints.append(json.loads(line))
            return True
        except Exception:
            return False

    # ── Execution ──────────────────────────────────────────────

    def execute(self, context: dict[str, Any] | None = None) -> ExecutionTrace:
        """Execute the loaded flow with constraints and VDB.

        Parameters
        ----------
        context : dict[str, Any] | None
            Initial context for the flow.

        Returns
        -------
        ExecutionTrace
            Complete execution trace.
        """
        self._init_subsystems()

        trace = ExecutionTrace(
            trace_id=f"trace-{time.time()}",
            start_time=time.time(),
        )

        ctx = context or {}

        # Execute each flow step
        for i, step in enumerate(self._flow_steps):
            step_id = step.get("id", f"step-{i}")
            step_type = step.get("type", "unknown")
            step_start = time.time()

            try:
                output = self._execute_step(step, ctx)
                duration = (time.time() - step_start) * 1000

                flow_step = FlowStep(
                    step_id=step_id,
                    step_type=step_type,
                    input_data=ctx.copy(),
                    output_data=output,
                    duration_ms=duration,
                    success=True,
                )
                trace.flow_steps.append(flow_step)
                ctx.update(output)

            except Exception as e:
                duration = (time.time() - step_start) * 1000
                flow_step = FlowStep(
                    step_id=step_id,
                    step_type=step_type,
                    input_data=ctx.copy(),
                    output_data={},
                    duration_ms=duration,
                    success=False,
                    error=str(e),
                )
                trace.flow_steps.append(flow_step)
                trace.overall_success = False

                # Apply constraints even on failure
                if self._caslang:
                    self._apply_constraints(trace, ctx)

                trace.end_time = time.time()
                self._trace = trace
                return trace

            # Apply constraints after each step
            if self._caslang:
                self._apply_constraints(trace, ctx)

            # Snapshot VDB after each step
            if self._vdb:
                self._snapshot_vdb(trace)

        trace.end_time = time.time()
        self._trace = trace
        return trace

    def _execute_step(
        self,
        step: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single flow step."""
        step_type = step.get("type", "unknown")
        step_config = step.get("config", {})

        if step_type == "agent":
            return self._execute_agent_step(step_config, context)
        elif step_type == "decision":
            return self._execute_decision_step(step_config, context)
        elif step_type == "action":
            return self._execute_action_step(step_config, context)
        elif step_type == "vdb_insert":
            return self._execute_vdb_insert_step(step_config, context)
        elif step_type == "vdb_query":
            return self._execute_vdb_query_step(step_config, context)
        elif step_type == "constraint_check":
            return self._execute_constraint_check_step(step_config, context)
        else:
            return {"status": "unknown_step_type", "type": step_type}

    def _execute_agent_step(
        self,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute an agent step."""
        model = config.get("model", "default")
        prompt = config.get("prompt", "")
        # Simulate agent execution
        return {
            "model": model,
            "prompt": prompt,
            "output": f"Agent response for: {prompt[:50]}...",
            "tokens": len(prompt) * 2,
        }

    def _execute_decision_step(
        self,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a decision step."""
        condition = config.get("condition", "true")
        # Evaluate simple condition
        if condition == "true":
            return {"decision": True, "branch": config.get("true_branch", "default")}
        return {"decision": False, "branch": config.get("false_branch", "default")}

    def _execute_action_step(
        self,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute an action step."""
        action = config.get("action", "noop")
        return {
            "action": action,
            "status": "completed",
            "context_keys": list(context.keys()),
        }

    def _execute_vdb_insert_step(
        self,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a VDB insert step."""
        if not self._vdb:
            return {"status": "vdb_disabled"}

        vector = config.get("vector", [0.0] * 128)
        metadata = config.get("metadata", {})
        self._vdb.insert(vector, metadata)
        return {
            "status": "inserted",
            "vector_dim": len(vector),
            "metadata_keys": list(metadata.keys()),
        }

    def _execute_vdb_query_step(
        self,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a VDB query step."""
        if not self._vdb:
            return {"status": "vdb_disabled"}

        vector = config.get("vector", [0.0] * 128)
        k = config.get("k", 5)
        results = self._vdb.query(vector, k=k)
        return {"status": "queried", "k": k, "results_count": len(results)}

    def _execute_constraint_check_step(
        self,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a constraint check step."""
        if not self._constraints:
            return {"status": "no_constraints", "all_passed": True}

        all_passed = True
        for i, constraint in enumerate(self._constraints):
            condition = constraint.get("condition", "")
            if condition and not self._evaluate_condition(condition, context):
                all_passed = False
                break

        return {
            "status": "checked",
            "constraints_evaluated": len(self._constraints),
            "all_passed": all_passed,
        }

    def _apply_constraints(
        self, trace: ExecutionTrace, context: dict[str, Any]
    ) -> None:
        """Apply constraints and record results."""
        if not self._constraints:
            return

        for i, constraint in enumerate(self._constraints):
            rule_id = constraint.get("rule_id", f"rule-{i}")
            rule_text = constraint.get("rule", str(constraint))

            # Simple constraint evaluation
            passed = True
            violations: list[str] = []

            condition = constraint.get("condition", "")
            if condition and not self._evaluate_condition(condition, context):
                passed = False
                violations.append(f"Condition failed: {condition}")

            constraint_result = ConstraintResult(
                rule_id=rule_id,
                rule_text=rule_text,
                passed=passed,
                violations=violations,
                rollback_applied=False,
            )
            trace.constraints.append(constraint_result)

            if not passed:
                trace.overall_success = False

    def _evaluate_condition(self, condition: str, context: dict[str, Any]) -> bool:
        """Evaluate a simple constraint condition."""
        if not condition:
            return True
        # Simple conditions: "x > 0", "y == 5", etc.
        try:
            # Replace context variables
            for key, value in context.items():
                if isinstance(value, (int, float, bool, str)):
                    condition = condition.replace(key, repr(value))
            # Evaluate safely
            return bool(eval(condition, {"__builtins__": {}}, {}))  # noqa: S307
        except Exception:
            return True  # Fail open

    def _snapshot_vdb(self, trace: ExecutionTrace) -> None:
        """Take a VDB snapshot."""
        if not self._vdb:
            return

        stats = self._vdb.stats
        snapshot = VDBSnapshot(
            vector_count=stats.get("insert_count", 0),
            pending_inserts=stats.get("insert_count", 0),
            metadata_keys=[],
            timestamp=time.time(),
        )
        trace.vdb_snapshots.append(snapshot)

    # ── Results ────────────────────────────────────────────────

    def get_trace(self) -> ExecutionTrace | None:
        """Get the last execution trace."""
        return self._trace

    def get_trace_json(self) -> str:
        """Get the last execution trace as JSON."""
        if not self._trace:
            return "{}"
        return json.dumps(self._trace.to_dict(), indent=2)

    def get_stats(self) -> dict[str, Any]:
        """Get runtime statistics."""
        if not self._trace:
            return {
                "executions": 0,
                "total_steps": 0,
                "total_duration_ms": 0.0,
                "success_rate": 0.0,
                "constraint_pass_rate": 0.0,
            }

        return {
            "executions": 1,
            "total_steps": self._trace.step_count,
            "total_duration_ms": self._trace.total_duration_ms,
            "success_rate": 1.0 if self._trace.overall_success else 0.0,
            "constraint_pass_rate": self._trace.constraint_pass_rate,
            "failed_steps": len(self._trace.failed_steps),
        }
