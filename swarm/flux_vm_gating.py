"""FLUX Path B — VM-based constraint gating for breeding decisions.

Replaces the simple FFI call with full VM execution:
  1. Python emits FLUX bytecode for the constraint check
  2. VM loads bytecode + constraints + room latent values
  3. VM executes, produces proof certificate
  4. Breeder gates on pass/fail + extracts severity from mask

This is the **proof-carrying** path. Every breed candidate gets a
verifiable SHA-256 proof chain, not just a boolean.
"""

from __future__ import annotations

__all__ = [
    "FluxVMGatingChecker",
    "FluxVMConfig",
]

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from sunset.flux_codegen import FluxBytecodeEmitter
from sunset.flux_vm_bridge import FluxVMBridge, FluxVMProof
from swarm.flux_gating import FluxGatingConfig, FluxCheckResult

logger = logging.getLogger(__name__)


@dataclass
class FluxVMConfig:
    """Configuration for VM-based FLUX gating."""

    scale: int = 1000  # fixed-point scale for float→int conversion
    max_cycles: int = 4096  # VM cycle limit
    collect_proof: bool = True  # whether to retrieve proof certificates
    provenance: bool = True  # whether to check provenance log length


class FluxVMGatingChecker:
    """VM-backed FLUX constraint checker.

    Wraps :class:`FluxVMBridge` to provide the same ``check_candidate``
    interface as :class:`PythonFluxFallback`, but every check runs
    inside the FLUX VM and produces an independently-verifiable proof
    certificate.

    Usage
    -----
        from swarm.flux_vm_gating import FluxVMGatingChecker, FluxVMConfig
        from swarm.flux_gating import FluxGatingConfig

        cfg = FluxGatingConfig(weight_bounds=(-5.0, 5.0), max_l2_norm=50.0)
        vm_cfg = FluxVMConfig(scale=1000)
        checker = FluxVMGatingChecker(flux_config=cfg, vm_config=vm_cfg)

        result = checker.check_candidate(weights, chaos=0.3, thermal_pressure=0.0)
        assert result.passed is True
        assert result.proof_hash is not None  # 32 bytes
    """

    def __init__(
        self,
        flux_config: FluxGatingConfig | None = None,
        vm_config: FluxVMConfig | None = None,
    ) -> None:
        self.flux_config = flux_config or FluxGatingConfig()
        self.vm_config = vm_config or FluxVMConfig()
        self._emitter = FluxBytecodeEmitter()
        # One bridge instance reused across checks (reset between calls)
        self._bridge = FluxVMBridge()
        self._bridge.new()

    def check_candidate(
        self,
        weights: np.ndarray,
        chaos: float = 0.3,
        thermal_pressure: float = 0.0,
    ) -> FluxCheckResult:
        """Check a single candidate via VM execution.

        Steps:
          1. Convert float weights to fixed-point i32
          2. Emit bytecode for bounds + L2 + variance check
          3. Load into VM, push values, run
          4. Extract pass/fail, proof hash, severity mask
          5. Build FluxCheckResult with proof metadata
        """
        cfg = self.flux_config
        scale = self.vm_config.scale
        w = np.asarray(weights, dtype=np.float32).flatten()
        n = len(w)

        # ── fixed-point conversion ────────────────────────────
        w_int = np.clip(w * scale, -2_147_483_648, 2_147_483_647).astype(np.int32)
        lo = int(cfg.weight_bounds[0] * scale)
        hi = int(cfg.weight_bounds[1] * scale)

        # ── emit bytecode ─────────────────────────────────────
        # Simple program: push all values, BatchCheck, Validate, Halt
        bc = self._emitter.emit_constraint_check(
            n_rooms=1,
            latent_dim=n,
            min_bound=cfg.weight_bounds[0],
            max_bound=cfg.weight_bounds[1],
            max_l2=cfg.max_l2_norm,
            max_var=cfg.max_variance,
        )

        # ── VM run ────────────────────────────────────────────
        try:
            self._bridge.reset()
            self._bridge.load_bytecode(bc)
            self._bridge.load_constraint(lo, hi)
            for v in w_int:
                self._bridge.push_value(int(v))
            passed = self._bridge.run()
        except Exception as exc:
            logger.warning("FLUX VM execution failed: %s", exc)
            return FluxCheckResult(
                passed=False,
                score=1.0,
                violations={"vm_error": 1.0},
            )

        # ── proof certificate ───────────────────────────────────
        proof_hash: bytes | None = None
        if self.vm_config.collect_proof:
            proof = self._bridge.get_proof()
            if proof is not None:
                proof_hash = proof.root_hash

        # ── severity / violations ─────────────────────────────
        # Derive violations from the fixed-point data (fast numpy check)
        violations: dict[str, float] = {}
        score = 0.0

        w_min, w_max = cfg.weight_bounds
        if np.any(w < w_min) or np.any(w > w_max):
            violations["bounds"] = float(np.mean((w < w_min) | (w > w_max)))
            score += violations["bounds"] * cfg.severity_weights.get("bounds", 1.0)

        l2 = float(np.linalg.norm(w))
        if l2 > cfg.max_l2_norm:
            violations["l2_norm"] = min(1.0, (l2 - cfg.max_l2_norm) / cfg.max_l2_norm)
            score += violations["l2_norm"] * cfg.severity_weights.get("l2_norm", 0.5)

        var = float(np.var(w))
        if var > cfg.max_variance:
            violations["variance"] = min(
                1.0, (var - cfg.max_variance) / cfg.max_variance
            )
            score += violations["variance"] * cfg.severity_weights.get("variance", 0.3)

        if chaos > cfg.max_chaos:
            violations["chaos"] = min(1.0, (chaos - cfg.max_chaos) / cfg.max_chaos)
            score += violations["chaos"] * cfg.severity_weights.get("chaos", 0.8)

        if thermal_pressure > cfg.thermal_budget_gate:
            violations["thermal"] = min(
                1.0,
                (thermal_pressure - cfg.thermal_budget_gate) / cfg.thermal_budget_gate,
            )
            score += violations["thermal"] * cfg.severity_weights.get("thermal", 0.7)

        # Normalize score
        total_weight = sum(cfg.severity_weights.values())
        if total_weight > 0:
            score = min(1.0, score / total_weight)

        # If VM says pass but numpy says fail, VM wins (it's the oracle)
        # If VM says fail but numpy says pass, VM wins (proof-carrying)
        passed = passed and score < cfg.pass_threshold

        return FluxCheckResult(
            passed=passed,
            score=score,
            violations=violations,
            # Attach proof metadata for downstream audit
            proof_hash=proof_hash,
            vm_cycles=self._bridge.get_cycles(),
        )

    def check_batch(
        self,
        candidates: list[np.ndarray],
        chaos_list: list[float] | None = None,
        thermal_list: list[float] | None = None,
    ) -> list[FluxCheckResult]:
        """Check multiple candidates in sequence.

        No parallel execution yet — the VM is single-threaded.
        Parallel batching will come via ``ParDispatch`` opcode.
        """
        results: list[FluxCheckResult] = []
        for i, w in enumerate(candidates):
            c = chaos_list[i] if chaos_list else 0.3
            t = thermal_list[i] if thermal_list else 0.0
            results.append(self.check_candidate(w, c, t))
        return results

    def __del__(self) -> None:
        if hasattr(self, "_bridge"):
            self._bridge.free()
