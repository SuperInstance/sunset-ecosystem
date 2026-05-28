"""Tests for Exotica NLopt solver with FLUX integration.

Run: python3 -m pytest tests/test_nlopt_solver.py -v --tb=short
"""
from __future__ import annotations

import pytest

from flux_compat.nlopt_solver import NLoptSolver, ProblemType, SolverResult
from flux_compat.v3_module import Module


# ── Objective functions ───────────────────────────────────

def _sphere(x):
    return sum(v ** 2 for v in x)

def _rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


# ── Construction ────────────────────────────────────────────

class TestConstruction:
    def test_basic_construction(self):
        solver = NLoptSolver(
            dim=2,
            bounds=[(0.0, 1.0), (-1.0, 2.0)],
            algorithm="DIRECT",
            maxeval=100,
        )
        assert solver.dim == 2
        assert solver.algorithm == "DIRECT"
        assert solver.flux_module is not None

    def test_algorithm_case_insensitive(self):
        s1 = NLoptSolver(dim=1, bounds=[(0.0, 1.0)], algorithm="direct")
        s2 = NLoptSolver(dim=1, bounds=[(0.0, 1.0)], algorithm="DIRECT")
        assert s1.algorithm == s2.algorithm == "DIRECT"

    def test_flux_module_type(self):
        solver = NLoptSolver(dim=2, bounds=[(0.0, 1.0), (0.0, 1.0)], algorithm="ESCH")
        assert isinstance(solver.flux_module, Module)
        assert solver.flux_module.version == 3

    def test_disassemble_non_empty(self):
        solver = NLoptSolver(dim=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
        asm = solver.disassemble()
        assert isinstance(asm, str)
        assert len(asm) > 0

    def test_fixed_point_bridge_exists(self):
        solver = NLoptSolver(dim=2, bounds=[(0.0, 10.0), (-5.0, 5.0)])
        assert solver._bridge is not None
        assert solver._bridge.scale_factor > 0

    def test_bounds_mismatch_raises(self):
        with pytest.raises(ValueError):
            NLoptSolver(dim=3, bounds=[(0.0, 1.0), (0.0, 1.0)])  # only 2 bounds for dim=3


# ── Algorithm resolution ────────────────────────────────────

class TestAlgorithmResolution:
    def test_direct(self):
        s = NLoptSolver(dim=2, bounds=[(0.0, 1.0), (0.0, 1.0)], algorithm="DIRECT")
        assert "DIRECT" in s.flux_module.metadata.get("algorithm", "")

    def test_esch(self):
        s = NLoptSolver(dim=2, bounds=[(0.0, 1.0), (0.0, 1.0)], algorithm="ESCH")
        assert "ESCH" in s.flux_module.metadata.get("algorithm", "")

    def test_crs2lm(self):
        s = NLoptSolver(dim=2, bounds=[(0.0, 1.0), (0.0, 1.0)], algorithm="CRS2-LM")
        assert "CRS2" in s.flux_module.metadata.get("algorithm", "")

    def test_unknown_algorithm_fallback(self):
        s = NLoptSolver(dim=2, bounds=[(0.0, 1.0), (0.0, 1.0)], algorithm="UNKNOWN")
        # Falls back to DIRECT
        assert s._flux_module is not None

    def test_local_algorithm_no_flux_codegen(self):
        # Local derivative algorithms don't have dedicated FLUX generators
        s = NLoptSolver(dim=2, bounds=[(0.0, 1.0), (0.0, 1.0)], algorithm="LBFGS")
        assert s._flux_module is not None


# ── Solving ─────────────────────────────────────────────────

class TestSolve:
    def test_sphere_direct(self):
        solver = NLoptSolver(
            dim=2,
            bounds=[(-1.0, 1.0), (-1.0, 1.0)],
            algorithm="DIRECT",
            maxeval=200,
            ftol_rel=1e-3,
        )
        result = solver.solve(_sphere)
        assert isinstance(result, SolverResult)
        assert result.num_evals > 0
        assert result.num_evals <= 200
        assert result.optimal_value >= 0.0
        # Should be near origin
        assert all(abs(v) < 0.5 for v in result.optimal_point)

    def test_sphere_esch(self):
        solver = NLoptSolver(
            dim=2,
            bounds=[(-2.0, 2.0), (-2.0, 2.0)],
            algorithm="ESCH",
            maxeval=500,
            ftol_rel=1e-2,
        )
        result = solver.solve(_sphere)
        assert result.num_evals > 0
        assert result.optimal_value >= 0.0

    def test_sphere_crs2lm(self):
        solver = NLoptSolver(
            dim=2,
            bounds=[(-1.0, 1.0), (-1.0, 1.0)],
            algorithm="CRS2-LM",
            maxeval=300,
        )
        result = solver.solve(_sphere)
        assert result.num_evals > 0
        assert result.optimal_value >= 0.0

    def test_rosenbrock_crs2lm(self):
        # CRS2-LM is derivative-free and works without gradients
        solver = NLoptSolver(
            dim=2,
            bounds=[(-2.0, 2.0), (-1.0, 3.0)],
            algorithm="CRS2-LM",
            maxeval=1000,
            ftol_rel=1e-4,
        )
        result = solver.solve(_rosenbrock, initial_guess=[-1.0, 2.0])
        assert result.optimal_value < 1.0  # Should find valley

    def test_custom_initial_guess(self):
        solver = NLoptSolver(
            dim=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            algorithm="DIRECT",
            maxeval=100,
        )
        result = solver.solve(_sphere, initial_guess=[0.9, 0.9])
        assert result.num_evals > 0

    def test_default_initial_guess(self):
        solver = NLoptSolver(
            dim=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            algorithm="DIRECT",
            maxeval=50,
        )
        result = solver.solve(_sphere)
        # Default guess was midpoint [0.5, 0.5]
        assert result.num_evals > 0


# ── Result validation ─────────────────────────────────────

class TestResult:
    def test_proof_certificate(self):
        solver = NLoptSolver(
            dim=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            algorithm="DIRECT",
            maxeval=100,
        )
        result = solver.solve(_sphere)
        assert len(result.proof_certificate) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in result.proof_certificate)

    def test_constraint_violations(self):
        solver = NLoptSolver(
            dim=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            algorithm="DIRECT",
            maxeval=100,
        )
        result = solver.solve(_sphere)
        assert len(result.constraint_violations) == 2
        # DIRECT is global and should respect bounds
        assert all(v == 0.0 for v in result.constraint_violations)

    def test_fixed_point_scale(self):
        solver = NLoptSolver(
            dim=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            algorithm="DIRECT",
            maxeval=100,
            frac_bits=16,
        )
        result = solver.solve(_sphere)
        assert result.fixed_point_scale > 0
        # auto_scale uses max_fp / (anchor * safety_margin), not 2^frac_bits
        # With bounds [0,1] pilot gives anchor ~1.0, total_bits=32
        # max_fp = 2^31 - 1, safety_margin = 2.0
        # scale ≈ 2^30 ≈ 1.07e9
        assert result.fixed_point_scale > 1e8

    def test_flux_module_in_result(self):
        solver = NLoptSolver(
            dim=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            algorithm="ESCH",
            maxeval=100,
        )
        result = solver.solve(_sphere)
        assert isinstance(result.flux_module, Module)

    def test_nlopt_status(self):
        solver = NLoptSolver(
            dim=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            algorithm="DIRECT",
            maxeval=1000,
        )
        result = solver.solve(_sphere)
        assert result.nlopt_status >= 0  # NLopt success codes are non-negative

    def test_algorithm_name_in_result(self):
        solver = NLoptSolver(
            dim=1,
            bounds=[(0.0, 1.0)],
            algorithm="DIRECT",
        )
        result = solver.solve(_sphere)
        assert result.algorithm == "DIRECT"


# ── Problem types ───────────────────────────────────────────

class TestProblemType:
    def test_unconstrained(self):
        s = NLoptSolver(
            dim=1,
            bounds=[(-10.0, 10.0)],
            algorithm="PRAXIS",
            problem_type=ProblemType.UNCONSTRAINED,
        )
        result = s.solve(_sphere)
        assert result.num_evals > 0

    def test_bounded(self):
        s = NLoptSolver(
            dim=1,
            bounds=[(0.0, 1.0)],
            algorithm="DIRECT",
            problem_type=ProblemType.BOUNDED,
        )
        result = s.solve(_sphere)
        assert all(0.0 <= v <= 1.0 for v in result.optimal_point)

    def test_constrained(self):
        s = NLoptSolver(
            dim=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            algorithm="COBYLA",
            problem_type=ProblemType.CONSTRAINED,
        )
        result = s.solve(_sphere)
        assert result.num_evals > 0


# ── Edge cases ──────────────────────────────────────────────

class TestEdgeCases:
    def test_1d_problem(self):
        solver = NLoptSolver(
            dim=1,
            bounds=[(-1.0, 1.0)],
            algorithm="DIRECT",
            maxeval=50,
        )
        result = solver.solve(_sphere)
        assert len(result.optimal_point) == 1
        assert result.optimal_value >= 0.0

    def test_high_dim(self):
        solver = NLoptSolver(
            dim=10,
            bounds=[(-1.0, 1.0)] * 10,
            algorithm="ESCH",
            maxeval=200,
        )
        result = solver.solve(_sphere)
        assert len(result.optimal_point) == 10
        assert result.num_evals > 0

    def test_narrow_bounds(self):
        solver = NLoptSolver(
            dim=2,
            bounds=[(0.499, 0.501), (0.499, 0.501)],
            algorithm="DIRECT",
            maxeval=20,
        )
        result = solver.solve(_sphere)
        # Should quickly find that the minimum is at the center
        assert result.num_evals > 0

    def test_frac_bits_variations(self):
        for bits in [8, 16, 24]:
            solver = NLoptSolver(
                dim=2,
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                algorithm="DIRECT",
                frac_bits=bits,
                maxeval=50,
            )
            result = solver.solve(_sphere)
            # frac_bits doesn't directly affect scale_factor in auto_scale
            # it only affects resolution: 1/scale
            assert result.fixed_point_scale > 1e6
            assert solver._bridge.resolution < 1.0
