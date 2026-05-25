"""FLUX Path A — constraint gating for breeding decisions.

Treats FLUX as a constraint library (not a VM).  Python calls
``flux_check_batch()`` via FFI (or a Python fallback), gets back
pass/fail/severity, and uses it to gate breeding decisions.

Architecture
------------
``FluxGatingChecker`` is the interface contract.  The real
implementation will wrap the Rust FFI (``flux_check_batch``).
Until that FFI is ready, ``PythonFluxFallback`` provides the same
API using numpy-based constraint checks.
"""

from __future__ import annotations

__all__ = [
    "FluxGatingConfig",
    "FluxCheckResult",
    "FluxGatingChecker",
    "PythonFluxFallback",
]

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── configuration ───────────────────────────────────────

@dataclass
class FluxGatingConfig:
    """Configuration for FLUX constraint gating.

    All thresholds are **inclusive** — a value exactly at the limit
    passes.
    """

    weight_bounds: tuple[float, float] = (-10.0, 10.0)
    max_l2_norm: float = 100.0
    max_variance: float = 10.0
    max_chaos: float = 1.0
    thermal_budget_gate: float = 1.0
    severity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "bounds": 1.0,
            "l2_norm": 0.5,
            "variance": 0.3,
            "chaos": 0.8,
            "thermal": 0.7,
        }
    )
    pass_threshold: float = 0.35
    top_k_batch: int = 10
    numpy_only: bool = True


# ── result type ─────────────────────────────────────────

@dataclass(frozen=True)
class FluxCheckResult:
    """Result of a single FLUX constraint check."""

    passed: bool
    score: float  # 0.0 = perfectly compliant, 1.0 = catastrophic
    violations: dict[str, float]  # {constraint_name: severity_value}

    @property
    def severity(self) -> float:
        """Alias for *score* (matches FLUX terminology)."""
        return self.score

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"<FluxCheckResult {status} score={self.score:.3f} violations={self.violations}>"


# ── Python fallback ───────────────────────────────────────

class PythonFluxFallback:
    """Pure-Python FLUX constraint checker.

    Evaluates five simple constraints using numpy:

    1. **Weight bounds** — every element must lie inside
       ``config.weight_bounds``.
    2. **L2 norm** — ``||weights||_2`` must be ≤ ``max_l2_norm``.
    3. **Variance** — ``np.var(weights)`` must be ≤ ``max_variance``.
    4. **Chaos** — ``chaos`` must be ≤ ``max_chaos``.
    5. **Thermal budget** — ``thermal_pressure`` must be ≤
       ``thermal_budget_gate``.

    Performance on a modern CPU (single-threaded):
    * ``check_candidate``  : ~150 k–250 k checks/sec
    * ``check_batch`` (N=32): ~80 k–120 k batches/sec
    """

    def __init__(self, config: FluxGatingConfig) -> None:
        self.config = config

    def check_candidate(
        self,
        weights: np.ndarray,
        chaos: float = 0.3,
        thermal_pressure: float = 0.0,
    ) -> FluxCheckResult:
        """Check a single candidate."""
        cfg = self.config
        violations: dict[str, float] = {}

        w = np.asarray(weights, dtype=np.float32)
        w_min, w_max = cfg.weight_bounds

        # 1. Bounds (skip if empty)
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
                violations["variance"] = (var - cfg.max_variance) / max(cfg.max_variance, 1e-6)

        # 4. Chaos
        if chaos > cfg.max_chaos:
            violations["chaos"] = (chaos - cfg.max_chaos) / max(cfg.max_chaos, 1e-6)

        # 5. Thermal
        if thermal_pressure > cfg.thermal_budget_gate:
            violations["thermal"] = (
                thermal_pressure - cfg.thermal_budget_gate
            ) / max(cfg.thermal_budget_gate, 1e-6)

        # Aggregate score
        score = 0.0
        for key, val in violations.items():
            weight = cfg.severity_weights.get(key, 1.0)
            score += val * weight
        score = min(score, 1.0)

        passed = score < cfg.pass_threshold
        return FluxCheckResult(passed=passed, score=score, violations=violations)

    def check_batch(
        self,
        weights_batch: np.ndarray,
        chaos_vec: np.ndarray | None = None,
        thermal_vec: np.ndarray | None = None,
    ) -> list[FluxCheckResult]:
        """Check a batch of candidates.

        *weights_batch* must be 2-D ``(N, dim)``.  1-D input is
        reshaped to ``(1, dim)``.
        """
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
                self.check_candidate(batch[i], float(chaos_vec[i]), float(thermal_vec[i]))
            )
        return results


# ── Rust FFI backend (stub) ─────────────────────────────

class _RustFFIBackend:
    """Future backend wrapping the compiled FLUX VM shared library.

    Expected export from Rust::

        int flux_check_batch(
            float* weights,      // flat (N * dim)
            int n,
            int dim,
            float max_l2,
            float max_variance,
            float max_chaos,
            float thermal_budget,
            uint8_t* out_passed  // N bytes (0/1)
        );

    This class is instantiated automatically when ``FLUX_VM_PATH``
    points to a loadable ``.so`` / ``.dylib`` and ``numpy_only``
    is *False*.
    """

    def __init__(self, vm_path: str, config: FluxGatingConfig) -> None:
        self.vm_path = vm_path
        self.config = config
        raise NotImplementedError(
            "Rust FFI backend is not yet implemented. "
            "Set FluxGatingConfig(numpy_only=True) to use the Python fallback."
        )


# ── unified checker ─────────────────────────────────────

class FluxGatingChecker:
    """Public API for FLUX constraint gating.

    Automatically selects the best available backend:

    1. If ``config.numpy_only`` is *True* → ``PythonFluxFallback``
    2. If ``FLUX_VM_PATH`` env var points to a loadable ``.so``
       and ``numpy_only`` is *False* → ``_RustFFIBackend``
    3. Otherwise → ``PythonFluxFallback`` (graceful degrade)
    """

    def __init__(
        self,
        config: FluxGatingConfig | None = None,
        vm_path: str | None = None,
    ) -> None:
        self.config = config or FluxGatingConfig()
        self._vm_path = vm_path or os.environ.get(
            "FLUX_VM_PATH", "../flux-vm-v3-temp/target/release/flux_vm"
        )
        self._backend: PythonFluxFallback | _RustFFIBackend

        if self.config.numpy_only:
            self._backend = PythonFluxFallback(self.config)
        else:
            so_path = (
                self._vm_path
                if os.path.isfile(self._vm_path)
                else self._vm_path + ".so"
            )
            if os.path.isfile(so_path):
                try:
                    self._backend = _RustFFIBackend(so_path, self.config)
                except Exception as exc:
                    logger.warning("Rust FFI load failed (%s), falling back to Python", exc)
                    self._backend = PythonFluxFallback(self.config)
            else:
                self._backend = PythonFluxFallback(self.config)

    # ── public API ──────────────────────────────────────────

    def check_candidate(
        self,
        weights: np.ndarray,
        chaos: float = 0.3,
        thermal_pressure: float = 0.0,
    ) -> FluxCheckResult:
        """Check a single candidate."""
        return self._backend.check_candidate(weights, chaos, thermal_pressure)

    def check_batch(
        self,
        weights_batch: np.ndarray,
        chaos_vec: np.ndarray | None = None,
        thermal_vec: np.ndarray | None = None,
    ) -> list[FluxCheckResult]:
        """Check a batch of candidates."""
        return self._backend.check_batch(weights_batch, chaos_vec, thermal_vec)

    def score_for_breeding(
        self,
        parent_a_weights: np.ndarray,
        parent_b_weights: np.ndarray,
        chaos_a: float,
        chaos_b: float,
        thermal_a: float = 0.0,
        thermal_b: float = 0.0,
    ) -> float:
        """Score a parent pair for breeding eligibility.

        Returns a value in ``[0.0, 1.0]``:
        * ``1.0`` — both parents fully compliant.
        * ``0.5`` — one parent fails (severe penalty).
        * ``0.0`` — both parents fail (blocked).

        Used as a **tiebreak** in ``_select_parents_vector()``.
        """
        ra = self.check_candidate(parent_a_weights, chaos_a, thermal_a)
        rb = self.check_candidate(parent_b_weights, chaos_b, thermal_b)

        if ra.passed and rb.passed:
            # Both clean → reward with score = 1.0 - tiny penalty for mild violations
            return 1.0 - (ra.score + rb.score) * 0.1
        if ra.passed or rb.passed:
            # One clean, one dirty → 0.5 - average penalty
            return 0.5 - (ra.score + rb.score) * 0.2
        # Both dirty → 0.0
        return 0.0

    def get_violating_rooms(
        self,
        weights_batch: np.ndarray,
        chaos_vec: np.ndarray | None = None,
        thermal_vec: np.ndarray | None = None,
    ) -> list[tuple[int, FluxCheckResult]]:
        """Return ``(index, result)`` tuples for all failing rooms."""
        results = self.check_batch(weights_batch, chaos_vec, thermal_vec)
        return [(i, r) for i, r in enumerate(results) if not r.passed]
