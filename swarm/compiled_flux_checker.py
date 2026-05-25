"""Compiled FLUX Checker — Path B integration layer.

Wraps ``FluxCompiler`` + ``FluxVMRunner`` into the same
``FluxGatingChecker`` interface used by ``PythonFluxFallback``.
Compiles constraints to bytecode once, caches the result, and
executes via the VM on every candidate check with runtime values
patched into the constant pool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from swarm.flux_compiler import (
    CmpOp,
    Const,
    FluxCompiler,
    IfNode,
    Var,
)
from swarm.flux_gating import (
    FluxGatingChecker,
    FluxGatingConfig,
    FluxViolation,
    GatingResult,
    ViolationSeverity,
)
from swarm.flux_vm_runner import FluxTrap, FluxVMRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CompiledConstraint:
    """Cache entry for a compiled constraint."""
    bytecode: bytes
    const_pool: List[float]
    var_slots: Dict[str, int]


class CompiledFluxChecker(FluxGatingChecker):
    """FLUX constraint checker using compiled bytecode + VM execution.

    Usage::

        checker = CompiledFluxChecker(config)
        result = checker.check_candidate(parent_idx, mutation_plan)

    Constraints are compiled once (lazy, on first use) and cached.
    The VM executes the bytecode with runtime values patched into
    the constant pool via ``var_slots``.

    If the VM raises ``FluxTrap`` (Validate sees 0.0), the candidate
    is blocked.  Any other exception falls back to logging the error
    and returning ``passed=False`` (safe default).
    """

    def __init__(self, config: Optional[FluxGatingConfig] = None) -> None:
        super().__init__(config)
        self._cache: Dict[str, _CompiledConstraint] = {}
        self._compiler = FluxCompiler(prefer_range_check=True)
        self._check_count = 0
        self._violation_count = 0

    # ── compilation helpers ─────────────────────────────────

    def _compile_weight_bounds(self) -> _CompiledConstraint:
        """Compile ``|weight| <= w_max`` to bytecode."""
        w_min, w_max = self.config.weight_bounds
        expr = IfNode(
            CmpOp("LE", Var("weight_norm"), Const(w_max)),
            then_expr=Const(1.0),
            else_expr=Const(0.0),
        )
        emitter = self._compiler.compile_constraint(expr, with_validate=False, with_halt=False)
        return _CompiledConstraint(
            bytecode=emitter.to_bytes(),
            const_pool=list(emitter.const_pool),
            var_slots=dict(emitter.var_slots),
        )

    def _compile_chaos_limit(self) -> _CompiledConstraint:
        """Compile ``chaos <= c_limit`` to bytecode."""
        expr = IfNode(
            CmpOp("LE", Var("chaos"), Const(self.config.chaos_limit)),
            then_expr=Const(1.0),
            else_expr=Const(0.0),
        )
        emitter = self._compiler.compile_constraint(expr, with_validate=False, with_halt=False)
        return _CompiledConstraint(
            bytecode=emitter.to_bytes(),
            const_pool=list(emitter.const_pool),
            var_slots=dict(emitter.var_slots),
        )

    def _compile_thermal_budget(self) -> _CompiledConstraint:
        """Compile ``thermal <= t_limit`` to bytecode."""
        expr = IfNode(
            CmpOp("LE", Var("thermal"), Const(self.config.thermal_budget_limit)),
            then_expr=Const(1.0),
            else_expr=Const(0.0),
        )
        emitter = self._compiler.compile_constraint(expr, with_validate=False, with_halt=False)
        return _CompiledConstraint(
            bytecode=emitter.to_bytes(),
            const_pool=list(emitter.const_pool),
            var_slots=dict(emitter.var_slots),
        )

    def _get_cached(self, key: str, factory: Any) -> _CompiledConstraint:
        if key not in self._cache:
            self._cache[key] = factory()
            logger.debug("Compiled and cached constraint: %s", key)
        return self._cache[key]

    def _run_vm(self, cc: _CompiledConstraint, runtime_vars: Dict[str, float]) -> float:
        """Run the VM with runtime variables patched into the constant pool."""
        # Clone constant pool and patch variable slots
        pool = list(cc.const_pool)
        for name, idx in cc.var_slots.items():
            if name in runtime_vars:
                pool[idx] = runtime_vars[name]
        runner = FluxVMRunner(pool)
        return runner.run(cc.bytecode)

    # ── single candidate ───────────────────────────────────

    def check_candidate(
        self,
        parent_idx: int,
        mutation_plan: dict[str, Any],
    ) -> GatingResult:
        self._check_count += 1
        violations: List[FluxViolation] = []

        # Weight bounds check (Python — weights are vectors, not VM scalars)
        weights = mutation_plan.get("weights")
        if weights is not None:
            w_norm = float(np.linalg.norm(weights))
            w_min, w_max = self.config.weight_bounds
            if w_norm < w_min or w_norm > w_max:
                v = FluxViolation(
                    room_id=parent_idx,
                    constraint_id="weight_bounds",
                    severity=ViolationSeverity.CRITICAL,
                    context={"norm": w_norm, "bounds": [w_min, w_max]},
                )
                violations.append(v)
                self.record_violation(
                    parent_idx, "weight_bounds", ViolationSeverity.CRITICAL, {"norm": w_norm}
                )

        # Chaos check — compile once, execute via VM with runtime value patched
        chaos = mutation_plan.get("chaos")
        if chaos is not None:
            cc = self._get_cached("chaos", self._compile_chaos_limit)
            try:
                result = self._run_vm(cc, {"chaos": float(chaos)})
                if result == 0.0:
                    v = FluxViolation(
                        room_id=parent_idx,
                        constraint_id="chaos_limit",
                        severity=ViolationSeverity.WARNING,
                        context={"chaos": chaos, "limit": self.config.chaos_limit},
                    )
                    violations.append(v)
                    self.record_violation(
                        parent_idx, "chaos_limit", ViolationSeverity.WARNING, {"chaos": chaos}
                    )
            except FluxTrap:
                v = FluxViolation(
                    room_id=parent_idx,
                    constraint_id="chaos_limit",
                    severity=ViolationSeverity.CRITICAL,
                    context={"chaos": chaos, "limit": self.config.chaos_limit, "trap": True},
                )
                violations.append(v)
                self.record_violation(
                    parent_idx, "chaos_limit", ViolationSeverity.CRITICAL, {"chaos": chaos, "trap": True}
                )
            except Exception as e:
                logger.warning("VM error in chaos check for parent %d: %s", parent_idx, e)
                v = FluxViolation(
                    room_id=parent_idx,
                    constraint_id="chaos_limit",
                    severity=ViolationSeverity.CRITICAL,
                    context={"error": str(e)},
                )
                violations.append(v)

        # Thermal budget check — compile once, execute via VM with runtime value patched
        thermal = mutation_plan.get("thermal")
        if thermal is not None:
            cc = self._get_cached("thermal", self._compile_thermal_budget)
            try:
                result = self._run_vm(cc, {"thermal": float(thermal)})
                if result == 0.0:
                    v = FluxViolation(
                        room_id=parent_idx,
                        constraint_id="thermal_budget",
                        severity=ViolationSeverity.WARNING,
                        context={"thermal": thermal, "limit": self.config.thermal_budget_limit},
                    )
                    violations.append(v)
                    self.record_violation(
                        parent_idx, "thermal_budget", ViolationSeverity.WARNING, {"thermal": thermal}
                    )
            except FluxTrap:
                v = FluxViolation(
                    room_id=parent_idx,
                    constraint_id="thermal_budget",
                    severity=ViolationSeverity.CRITICAL,
                    context={"thermal": thermal, "limit": self.config.thermal_budget_limit, "trap": True},
                )
                violations.append(v)
                self.record_violation(
                    parent_idx, "thermal_budget", ViolationSeverity.CRITICAL, {"thermal": thermal, "trap": True}
                )
            except Exception as e:
                logger.warning("VM error in thermal check for parent %d: %s", parent_idx, e)
                v = FluxViolation(
                    room_id=parent_idx,
                    constraint_id="thermal_budget",
                    severity=ViolationSeverity.CRITICAL,
                    context={"error": str(e)},
                )
                violations.append(v)

        passed = not violations
        if not passed:
            self._violation_count += 1

        return GatingResult(
            candidate_id=str(parent_idx),
            passed=passed,
            score=1.0 if passed else 0.0,
            violations=violations,
            metadata={"checker_type": "compiled_flux"},
        )

    # ── batch check ────────────────────────────────────────

    def check_batch(
        self,
        parent_indices: List[int],
        mutation_plans: List[dict[str, Any]],
    ) -> List[GatingResult]:
        """Check a batch of candidates (serial for now; VM is fast enough)."""
        return [self.check_candidate(pi, mp) for pi, mp in zip(parent_indices, mutation_plans)]

    def stats(self) -> dict[str, Any]:
        return {
            "check_count": self._check_count,
            "violation_count": self._violation_count,
            "cache_size": len(self._cache),
        }
