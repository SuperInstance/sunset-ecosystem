"""Compiled FLUX Checker — Path B integration layer.

Wraps ``FluxCompiler`` + ``FluxVMRunner`` into the same API as
``PythonFluxFallback``.

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
from swarm.flux_gating import FluxGatingConfig, FluxCheckResult
from swarm.flux_vm_runner import FluxTrap, FluxVMRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CompiledConstraint:
    """Cache entry for a compiled constraint."""

    bytecode: bytes
    const_pool: List[float]
    var_slots: Dict[str, int]


class CompiledFluxChecker:
    """FLUX constraint checker using compiled bytecode + VM execution.

    Matches the ``PythonFluxFallback`` API so it can be swapped in
    as a drop-in replacement.

    Usage::

        checker = CompiledFluxChecker(config)
        result = checker.check_candidate(weights, chaos=0.5, thermal_pressure=0.2)

    Constraints are compiled once (lazy, on first use) and cached.
    The VM executes the bytecode with runtime values patched into
    the constant pool via ``var_slots``.
    """

    def __init__(self, config: FluxGatingConfig) -> None:
        self.config = config
        self._cache: Dict[str, _CompiledConstraint] = {}
        self._compiler = FluxCompiler(prefer_range_check=True)
        self._check_count = 0
        self._violation_count = 0

    # ── compilation helpers ─────────────────────────────────

    def _compile_chaos_limit(self) -> _CompiledConstraint:
        """Compile ``chaos <= max_chaos`` to bytecode."""
        expr = IfNode(
            CmpOp("LE", Var("chaos"), Const(self.config.max_chaos)),
            then_expr=Const(1.0),
            else_expr=Const(0.0),
        )
        emitter = self._compiler.compile_constraint(
            expr, with_validate=False, with_halt=False
        )
        return _CompiledConstraint(
            bytecode=emitter.to_bytes(),
            const_pool=list(emitter.const_pool),
            var_slots=dict(emitter.var_slots),
        )

    def _compile_thermal_budget(self) -> _CompiledConstraint:
        """Compile ``thermal <= thermal_budget_gate`` to bytecode."""
        expr = IfNode(
            CmpOp("LE", Var("thermal"), Const(self.config.thermal_budget_gate)),
            then_expr=Const(1.0),
            else_expr=Const(0.0),
        )
        emitter = self._compiler.compile_constraint(
            expr, with_validate=False, with_halt=False
        )
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
        pool = list(cc.const_pool)
        for name, idx in cc.var_slots.items():
            if name in runtime_vars:
                pool[idx] = runtime_vars[name]
        runner = FluxVMRunner(pool)
        return runner.run(cc.bytecode)

    # ── single candidate ───────────────────────────────────

    def check_candidate(
        self,
        weights: np.ndarray,
        chaos: float = 0.3,
        thermal_pressure: float = 0.0,
    ) -> FluxCheckResult:
        self._check_count += 1
        cfg = self.config
        violations: dict[str, float] = {}

        w = np.asarray(weights, dtype=np.float32)
        w_min, w_max = cfg.weight_bounds

        # 1. Bounds
        if w.size > 0:
            over = float(np.max(w)) - w_max
            under = w_min - float(np.min(w))
            if over > 0:
                violations["bounds"] = over / max(abs(w_max), 1e-6)
            if under > 0:
                violations["bounds"] = max(
                    violations.get("bounds", 0.0),
                    under / max(abs(w_min), 1e-6),
                )

        # 2. L2 norm
        l2 = float(np.linalg.norm(w))
        if l2 > cfg.max_l2_norm:
            violations["l2_norm"] = (l2 - cfg.max_l2_norm) / max(cfg.max_l2_norm, 1e-6)

        # 3. Variance
        if w.size > 1:
            var = float(np.var(w))
            if var > cfg.max_variance:
                violations["variance"] = (var - cfg.max_variance) / max(
                    cfg.max_variance, 1e-6
                )

        # 4. Chaos — compile once, execute via VM with runtime value patched
        cc = self._get_cached("chaos", self._compile_chaos_limit)
        try:
            result = self._run_vm(cc, {"chaos": float(chaos)})
            if result == 0.0:
                violations["chaos"] = (chaos - cfg.max_chaos) / max(cfg.max_chaos, 1e-6)
        except Exception as e:
            logger.warning("VM error in chaos check: %s", e)
            violations["chaos"] = (chaos - cfg.max_chaos) / max(cfg.max_chaos, 1e-6)

        # 5. Thermal — compile once, execute via VM with runtime value patched
        cc = self._get_cached("thermal", self._compile_thermal_budget)
        try:
            result = self._run_vm(cc, {"thermal": float(thermal_pressure)})
            if result == 0.0:
                violations["thermal"] = (
                    thermal_pressure - cfg.thermal_budget_gate
                ) / max(cfg.thermal_budget_gate, 1e-6)
        except Exception as e:
            logger.warning("VM error in thermal check: %s", e)
            violations["thermal"] = (thermal_pressure - cfg.thermal_budget_gate) / max(
                cfg.thermal_budget_gate, 1e-6
            )

        # Aggregate score
        score = 0.0
        for key, val in violations.items():
            weight = cfg.severity_weights.get(key, 1.0)
            score += val * weight
        score = min(score, 1.0)

        passed = score < cfg.pass_threshold
        if not passed:
            self._violation_count += 1

        return FluxCheckResult(passed=passed, score=score, violations=violations)

    # ── batch check ────────────────────────────────────────

    def check_batch(
        self,
        weights_batch: np.ndarray,
        chaos_vec: np.ndarray | None = None,
        thermal_vec: np.ndarray | None = None,
    ) -> list[FluxCheckResult]:
        """Check a batch of candidates."""
        batch = np.asarray(weights_batch, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        n = batch.shape[0]

        if chaos_vec is None:
            chaos_vec = np.full(n, 0.3, dtype=np.float32)
        else:
            chaos_vec = np.asarray(chaos_vec, dtype=np.float32)

        if thermal_vec is None:
            thermal_vec = np.zeros(n, dtype=np.float32)
        else:
            thermal_vec = np.asarray(thermal_vec, dtype=np.float32)

        results: list[FluxCheckResult] = []
        for i in range(n):
            results.append(
                self.check_candidate(
                    batch[i], float(chaos_vec[i]), float(thermal_vec[i])
                )
            )
        return results

    def stats(self) -> dict[str, Any]:
        return {
            "check_count": self._check_count,
            "violation_count": self._violation_count,
            "cache_size": len(self._cache),
        }
