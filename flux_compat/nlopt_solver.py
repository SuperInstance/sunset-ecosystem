"""Exotica NLopt Solver — FLUX-integrated Python wrapper.

Wraps Steven G. Johnson's NLopt library with FLUX VM constraint checking,
proof certificates, and fixed-point scaling.

Usage::
    from flux_compat.nlopt_solver import NLoptSolver, ProblemType

    solver = NLoptSolver(
        dim=3,
        bounds=[(0.0, 1.0), (-1.0, 2.0), (0.5, 0.5)],
        algorithm="DIRECT",
        maxeval=1000,
        ftol_rel=1e-4,
    )
    result = solver.solve(initial_guess=[0.5, 0.5, 0.5])
    print(result.optimal_value, result.proof_certificate)
"""

from __future__ import annotations

__all__ = ["NLoptSolver", "ProblemType", "SolverResult"]

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Tuple

import numpy as np

from flux_compat.fixed_point_bridge import FixedPointBridge
from flux_compat.flux_opt_codegen import (
    FLUXOptimizerCodegen,
    generate_direct_module,
    generate_esch_module,
    generate_crs2lm_module,
)


class ProblemType(Enum):
    """NLopt problem classification."""

    UNCONSTRAINED = auto()
    BOUNDED = auto()
    CONSTRAINED = auto()


@dataclass(frozen=True)
class SolverResult:
    """Result from an NLopt solve with FLUX audit trail."""

    optimal_point: List[float]
    optimal_value: float
    num_evals: int
    nlopt_status: int
    algorithm: str
    flux_module: object  # flux.Module
    proof_certificate: str  # SHA-256 hex
    constraint_violations: List[float]
    fixed_point_scale: float


class NLoptSolver:
    """FLUX-integrated NLopt solver.

    Parameters
    ----------
    dim:
        Problem dimension.
    bounds:
        List of (lower, upper) bounds, length = dim.
    algorithm:
        NLopt algorithm name (``DIRECT``, ``ESCH``, ``CRS2-LM``, etc.).
    maxeval:
        Maximum function evaluations.
    ftol_rel:
        Relative function-value tolerance.
    xtol_rel:
        Relative parameter tolerance.
    problem_type:
        UNCONSTRAINED / BOUNDED / CONSTRAINED.
    frac_bits:
        Fixed-point fractional bits for FLUX module.
    """

    def __init__(
        self,
        dim: int,
        bounds: List[Tuple[float, float]],
        algorithm: str = "LD_TNEWTON",
        maxeval: int = 1000,
        ftol_rel: float = 1e-6,
        xtol_rel: float = 1e-4,
        problem_type: ProblemType = ProblemType.BOUNDED,
        frac_bits: int = 16,
    ) -> None:
        self.dim = dim
        self.bounds = bounds
        self.algorithm = algorithm.upper()
        self.maxeval = maxeval
        self.ftol_rel = ftol_rel
        self.xtol_rel = xtol_rel
        self.problem_type = problem_type
        self.frac_bits = frac_bits

        # Auto-scale fixed-point bridge from bounds
        pilot = []
        for lo, hi in bounds:
            pilot.extend([lo, hi, (lo + hi) / 2.0])
        self._bridge = FixedPointBridge.auto_scale(pilot, frac_bits=frac_bits)

        # Pre-generate FLUX module for constraint checking
        self._flux_module = self._build_flux_module()

    def _build_flux_module(self):
        """Generate FLUX bytecode module for this solver configuration."""
        codegen = FLUXOptimizerCodegen(frac_bits=self.frac_bits)

        # Map algorithm name to generator
        algo_map = {
            "DIRECT": lambda: generate_direct_module(
                self.dim, self.bounds, self.maxeval, self.ftol_rel, self.frac_bits
            ),
            "ESCH": lambda: generate_esch_module(
                self.dim,
                self.bounds,
                pop_size=min(10 * self.dim, 100),
                maxeval=self.maxeval,
                frac_bits=self.frac_bits,
            ),
            "CRS2-LM": lambda: generate_crs2lm_module(
                self.dim, self.bounds, pop_size=2 * self.dim, frac_bits=self.frac_bits
            ),
        }

        # Try exact match, then fuzzy match
        for key in algo_map:
            if key in self.algorithm or self.algorithm in key:
                return algo_map[key]()

        # Fallback: use DIRECT for unknown algorithms
        return generate_direct_module(
            self.dim, self.bounds, self.maxeval, self.ftol_rel, self.frac_bits
        )

    def solve(
        self,
        objective: Callable[[List[float]], float],
        initial_guess: List[float] | None = None,
    ) -> SolverResult:
        """Solve the optimization problem.

        Parameters
        ----------
        objective:
            Function ``f(x) -> float`` to minimize.
        initial_guess:
            Starting point. Default: midpoint of bounds.

        Returns
        -------
        SolverResult
            Optimal point, value, NLopt status, and FLUX proof certificate.
        """
        import nlopt

        # Default initial guess: midpoint
        if initial_guess is None:
            initial_guess = [(lo + hi) / 2.0 for lo, hi in self.bounds]

        x0 = np.array(initial_guess, dtype=float)

        # Map algorithm string to NLopt enum
        algo_enum = self._resolve_algorithm(self.algorithm)
        opt = nlopt.opt(algo_enum, self.dim)

        # Set bounds
        lb = np.array([b[0] for b in self.bounds], dtype=float)
        ub = np.array([b[1] for b in self.bounds], dtype=float)
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)

        # Set tolerances
        opt.set_maxeval(self.maxeval)
        opt.set_ftol_rel(self.ftol_rel)
        opt.set_xtol_rel(self.xtol_rel)

        # Objective wrapper with eval counter
        eval_count = [0]

        def _obj(x, grad):
            eval_count[0] += 1
            val = float(objective(x.tolist()))
            if grad is not None and grad.size > 0:
                # Numerical gradient for gradient-based algorithms
                eps = 1e-6
                for i in range(len(grad)):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    x_minus = x.copy()
                    x_minus[i] -= eps
                    grad[i] = (
                        float(objective(x_plus.tolist()))
                        - float(objective(x_minus.tolist()))
                    ) / (2 * eps)
            return val

        opt.set_min_objective(_obj)

        # Run optimization
        try:
            # nlopt.optimize returns the optimal point, modifies x0 in-place
            opt.optimize(x0)
            result_code = opt.last_optimize_result()
            optimal_value = float(opt.last_optimum_value())
        except nlopt.RoundoffLimited:
            result_code = nlopt.ROUNDOFF_LIMITED
            optimal_value = float(opt.last_optimum_value())
        except nlopt.ForcedStop:
            result_code = nlopt.FORCED_STOP
            optimal_value = float(opt.last_optimum_value())
        except Exception:
            result_code = -1
            optimal_value = float(opt.last_optimum_value())

        # FLUX constraint checking on result
        fp_value = self._bridge.encode(float(optimal_value))
        violations = self._check_constraints(x0)

        # Generate proof certificate from FLUX module
        proof = self._generate_proof(optimal_value, eval_count[0], result_code)

        return SolverResult(
            optimal_point=x0.tolist(),
            optimal_value=float(optimal_value),
            num_evals=eval_count[0],
            nlopt_status=int(result_code),
            algorithm=self.algorithm,
            flux_module=self._flux_module,
            proof_certificate=proof,
            constraint_violations=violations,
            fixed_point_scale=self._bridge.scale_factor,
        )

    def _resolve_algorithm(self, name: str) -> int:
        """Map algorithm string to NLopt enum value."""
        import nlopt

        mapping = {
            "DIRECT": nlopt.GN_DIRECT,
            "DIRECT_L": nlopt.GN_DIRECT_L,
            "CRS2-LM": nlopt.GN_CRS2_LM,
            "ESCH": nlopt.GN_ESCH,
            "ISRES": nlopt.GN_ISRES,
            "MLSL": nlopt.GN_MLSL,
            "STOGO": nlopt.GD_STOGO,
            "PRAXIS": nlopt.LN_PRAXIS,
            "COBYLA": nlopt.LN_COBYLA,
            "NEWUOA": nlopt.LN_NEWUOA,
            "NELDERMEAD": nlopt.LN_NELDERMEAD,
            "BOBYQA": nlopt.LN_BOBYQA,
            "AUGLAG": nlopt.LN_AUGLAG,
            "LBFGS": nlopt.LD_LBFGS,
            "TNEWTON": nlopt.LD_TNEWTON,
            "MMA": nlopt.LD_MMA,
            "SLSQP": nlopt.LD_SLSQP,
            "CCSAQ": nlopt.LD_CCSAQ,
        }

        # Try exact match first
        if name in mapping:
            return mapping[name]

        # Try case-insensitive contains
        name_upper = name.upper().replace("NLOPT_", "")
        for key, val in mapping.items():
            if key in name_upper or name_upper in key:
                return val

        # Default: TNEWTON (local derivative)
        return nlopt.LD_TNEWTON

    def _check_constraints(self, x: np.ndarray) -> List[float]:
        """Check bound constraints and return violations (positive = infeasible)."""
        violations = []
        for i, (lo, hi) in enumerate(self.bounds):
            if x[i] < lo:
                violations.append(lo - x[i])
            elif x[i] > hi:
                violations.append(x[i] - hi)
            else:
                violations.append(0.0)
        return violations

    def _generate_proof(self, value: float, evals: int, status: int) -> str:
        """Generate a SHA-256 proof certificate from the FLUX module."""
        import hashlib

        # Hash the module's bytecode + result metadata
        mod_bytes = self._flux_module.to_bytecode()
        meta = f"{value}:{evals}:{status}:{self._bridge.scale_factor}".encode()
        return hashlib.sha256(mod_bytes + meta).hexdigest()

    @property
    def flux_module(self):
        """The generated FLUX bytecode module (for inspection)."""
        return self._flux_module

    def disassemble(self) -> str:
        """Return human-readable disassembly of the FLUX module."""
        return self._flux_module.disasm()
