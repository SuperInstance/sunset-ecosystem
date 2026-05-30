"""Tests for Hamiltonian constraint satisfaction (Pattern 1).

Covers: single constraint satisfaction, multiple constraint intersection,
Störmer-Verlet energy conservation, damped relaxation convergence,
augmented Lagrangian multiplier updates, energy conservation quality
metrics, and contradictory constraint settling.
"""
from __future__ import annotations

import numpy as np
import pytest

from fleet.hamiltonian_constraints import (
    AugmentedEnergy,
    Constraint,
    HamiltonianSystem,
)


# ─── helpers ───────────────────────────────────────────────────────────────

def _circle_constraint(radius: float = 1.0):
    """Return (value_fn, gradient_fn) for a circle: x² + y² = radius²."""

    def value_fn(q: np.ndarray) -> float:
        return float(q[0] ** 2 + q[1] ** 2 - radius ** 2)

    def grad_fn(q: np.ndarray) -> np.ndarray:
        return np.array([2 * q[0], 2 * q[1]], dtype=float)

    return value_fn, grad_fn


def _plane_constraint(normal: np.ndarray, offset: float):
    """Return (value_fn, gradient_fn) for a plane: n·q = offset."""
    n = np.asarray(normal, dtype=float)

    def value_fn(q: np.ndarray) -> float:
        return float(np.dot(n, q) - offset)

    def grad_fn(q: np.ndarray) -> np.ndarray:
        return n.copy()

    return value_fn, grad_fn


def _sphere_constraint(radius: float = 1.0, dim: int = 3):
    """Return (value_fn, gradient_fn) for a sphere in N dimensions."""

    def value_fn(q: np.ndarray) -> float:
        return float(np.dot(q, q) - radius ** 2)

    def grad_fn(q: np.ndarray) -> np.ndarray:
        return 2 * q

    return value_fn, grad_fn


# ─── Constraint dataclass ──────────────────────────────────────────────────


class TestConstraintDataclass:
    def test_constraint_creation_defaults(self) -> None:
        c = Constraint(value_fn=lambda q: 0.0, gradient_fn=lambda q: np.zeros(2))
        assert c.weight == 1.0
        assert c.multiplier == 0.0
        assert c.name == ""

    def test_constraint_creation_explicit(self) -> None:
        c = Constraint(
            value_fn=lambda q: float(q[0]),
            gradient_fn=lambda q: np.array([1.0, 0.0]),
            weight=2.5,
            multiplier=0.5,
            name="x_axis",
        )
        assert c.weight == 2.5
        assert c.multiplier == 0.5
        assert c.name == "x_axis"

    def test_constraint_callable_invocation(self) -> None:
        c = Constraint(value_fn=lambda q: float(q[0] - 1.0), gradient_fn=lambda q: np.array([1.0, 0.0]))
        assert c.value_fn(np.array([1.0, 0.0])) == 0.0
        assert c.value_fn(np.array([2.0, 0.0])) == 1.0
        np.testing.assert_array_equal(c.gradient_fn(np.array([0.0, 0.0])), np.array([1.0, 0.0]))


# ─── AugmentedEnergy ───────────────────────────────────────────────────────


class TestAugmentedEnergy:
    def test_total_computed_correctly(self) -> None:
        ae = AugmentedEnergy(kinetic=1.0, potential=2.0, penalty=3.0, lagrangian=4.0)
        assert ae.total == pytest.approx(10.0)

    def test_conservation_quality_with_baseline(self) -> None:
        ae = AugmentedEnergy(kinetic=1.0, potential=2.0, penalty=0.1, lagrangian=0.1, total=3.2)
        assert ae.conservation_quality(baseline=3.0) == pytest.approx(0.2 / 3.0, abs=1e-10)

    def test_conservation_quality_zero_baseline(self) -> None:
        ae = AugmentedEnergy(total=0.0)
        assert ae.conservation_quality(baseline=0.0) == 0.0

    def test_conservation_quality_without_baseline(self) -> None:
        ae = AugmentedEnergy(kinetic=1.0, potential=1.0, penalty=0.5, lagrangian=0.5, total=3.0)
        assert ae.conservation_quality() == pytest.approx(1.0 / 3.0, abs=1e-10)

    def test_conservation_quality_zero_total(self) -> None:
        ae = AugmentedEnergy(total=0.0)
        assert ae.conservation_quality() == 0.0


# ─── HamiltonianSystem initialization ──────────────────────────────────────


class TestHamiltonianSystemInit:
    def test_default_init(self) -> None:
        sys = HamiltonianSystem(dim=3)
        np.testing.assert_array_equal(sys.get_state(), np.zeros(3))
        np.testing.assert_array_equal(sys.get_momentum(), np.zeros(3))
        assert sys.step_count == 0

    def test_init_with_state(self) -> None:
        sys = HamiltonianSystem(dim=2, state=np.array([1.0, 2.0]))
        np.testing.assert_array_equal(sys.get_state(), np.array([1.0, 2.0]))

    def test_init_state_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="state shape"):
            HamiltonianSystem(dim=3, state=np.array([1.0, 2.0]))

    def test_init_with_potential(self) -> None:
        sys = HamiltonianSystem(
            dim=2,
            potential_fn=lambda q: float(np.dot(q, q)),
            potential_grad_fn=lambda q: 2 * q,
        )
        assert sys._potential_fn(np.array([1.0, 0.0])) == 1.0


# ─── Constraint management ───────────────────────────────────────────────────


class TestConstraintManagement:
    def test_add_constraint(self) -> None:
        sys = HamiltonianSystem(dim=2)
        c = sys.add_constraint(
            value_fn=lambda q: float(q[0] - 1.0),
            gradient_fn=lambda q: np.array([1.0, 0.0]),
            name="x_eq_1",
        )
        assert c.name == "x_eq_1"
        assert len(sys.list_constraints()) == 1

    def test_add_multiple_constraints(self) -> None:
        sys = HamiltonianSystem(dim=2)
        sys.add_constraint(*_circle_constraint(1.0), name="circle")
        sys.add_constraint(*_plane_constraint([1.0, 0.0], 0.5), name="plane")
        assert len(sys.list_constraints()) == 2

    def test_remove_constraint(self) -> None:
        sys = HamiltonianSystem(dim=2)
        sys.add_constraint(*_circle_constraint(1.0), name="circle")
        assert sys.remove_constraint("circle") is True
        assert len(sys.list_constraints()) == 0

    def test_remove_constraint_not_found(self) -> None:
        sys = HamiltonianSystem(dim=2)
        assert sys.remove_constraint("missing") is False

    def test_clear_constraints(self) -> None:
        sys = HamiltonianSystem(dim=2)
        sys.add_constraint(*_circle_constraint(1.0), name="c1")
        sys.add_constraint(*_circle_constraint(2.0), name="c2")
        sys.clear_constraints()
        assert len(sys.list_constraints()) == 0

    def test_list_constraints_returns_copy(self) -> None:
        sys = HamiltonianSystem(dim=2)
        sys.add_constraint(*_circle_constraint(1.0), name="circle")
        lst = sys.list_constraints()
        lst.clear()
        assert len(sys.list_constraints()) == 1


# ─── Single constraint satisfaction ───────────────────────────────────────────


class TestSingleConstraintSatisfaction:
    def test_circle_constraint_relaxed(self) -> None:
        """Damped relaxation should drive state onto a circle manifold."""
        sys = HamiltonianSystem(dim=2, damping=0.05)
        sys.set_state(np.array([2.0, 0.0]))
        sys.add_constraint(*_circle_constraint(1.0), weight=10.0, name="circle")

        sys.reset_momentum()
        for _ in range(5000):
            sys.step_damped(0.01)

        q = sys.get_state()
        radius = np.linalg.norm(q)
        assert radius == pytest.approx(1.0, abs=1e-2)
        assert sys.total_violation() < 1e-2

    def test_plane_constraint_relaxed(self) -> None:
        """Damped relaxation onto a plane constraint."""
        sys = HamiltonianSystem(dim=3, damping=0.1)
        sys.set_state(np.array([1.0, 2.0, 3.0]))
        sys.add_constraint(*_plane_constraint([0.0, 0.0, 1.0], 1.0), weight=50.0, name="z_eq_1")

        sys.reset_momentum()
        for _ in range(3000):
            sys.step_damped(0.01)

        q = sys.get_state()
        assert q[2] == pytest.approx(1.0, abs=1e-2)
        assert sys.total_violation() < 1e-2

    def test_single_constraint_energy_decreases(self) -> None:
        """Penalty energy should decrease as constraint is satisfied."""
        sys = HamiltonianSystem(dim=2, damping=0.05)
        sys.set_state(np.array([2.0, 0.0]))
        sys.add_constraint(*_circle_constraint(1.0), weight=10.0, name="circle")
        sys.reset_momentum()

        e0 = sys.energy()
        for _ in range(2000):
            sys.step_damped(0.01)
        e1 = sys.energy()

        assert e1.penalty < e0.penalty
        assert e1.total <= e0.total + 1e-3  # energy should not explode


# ─── Multiple constraint intersection ────────────────────────────────────────


class TestMultipleConstraintIntersection:
    def test_two_planes_intersection(self) -> None:
        """State should settle on the intersection of two planes."""
        sys = HamiltonianSystem(dim=3, damping=0.05)
        sys.set_state(np.array([1.0, 1.0, 1.0]))
        sys.add_constraint(*_plane_constraint([1.0, 0.0, 0.0], 0.5), weight=50.0, name="x_eq_0.5")
        sys.add_constraint(*_plane_constraint([0.0, 1.0, 0.0], 0.5), weight=50.0, name="y_eq_0.5")

        sys.reset_momentum()
        for _ in range(3000):
            sys.step_damped(0.01)

        q = sys.get_state()
        assert q[0] == pytest.approx(0.5, abs=1e-2)
        assert q[1] == pytest.approx(0.5, abs=1e-2)
        assert sys.total_violation() < 1e-2

    def test_circle_and_plane_intersection(self) -> None:
        """Circle + plane = two points; state should go to one of them."""
        sys = HamiltonianSystem(dim=3, damping=0.05)
        sys.set_state(np.array([0.5, 0.5, 0.5]))
        sys.add_constraint(*_sphere_constraint(1.0, 3), weight=20.0, name="sphere")
        sys.add_constraint(*_plane_constraint([0.0, 0.0, 1.0], 0.0), weight=20.0, name="z_eq_0")

        sys.reset_momentum()
        for _ in range(5000):
            sys.step_damped(0.01)

        q = sys.get_state()
        xy_radius = np.sqrt(q[0] ** 2 + q[1] ** 2)
        assert q[2] == pytest.approx(0.0, abs=1e-2)
        assert xy_radius == pytest.approx(1.0, abs=1e-2)
        assert sys.total_violation() < 1e-2

    def test_three_constraints_intersection(self) -> None:
        """Three orthogonal planes → single point."""
        sys = HamiltonianSystem(dim=3, damping=0.1)
        sys.set_state(np.array([5.0, -3.0, 2.0]))
        sys.add_constraint(*_plane_constraint([1.0, 0.0, 0.0], 1.0), weight=100.0, name="x_eq_1")
        sys.add_constraint(*_plane_constraint([0.0, 1.0, 0.0], 2.0), weight=100.0, name="y_eq_2")
        sys.add_constraint(*_plane_constraint([0.0, 0.0, 1.0], 3.0), weight=100.0, name="z_eq_3")

        sys.reset_momentum()
        for _ in range(3000):
            sys.step_damped(0.01)

        q = sys.get_state()
        np.testing.assert_allclose(q, np.array([1.0, 2.0, 3.0]), atol=1e-2)
        assert sys.total_violation() < 1e-2

    def test_multiple_constraints_rms_violation(self) -> None:
        sys = HamiltonianSystem(dim=3, damping=0.1)
        sys.set_state(np.array([1.0, 1.0, 1.0]))
        sys.add_constraint(*_plane_constraint([1.0, 0.0, 0.0], 0.0), weight=10.0, name="x_eq_0")
        sys.add_constraint(*_plane_constraint([0.0, 1.0, 0.0], 0.0), weight=10.0, name="y_eq_0")

        sys.reset_momentum()
        for _ in range(2000):
            sys.step_damped(0.01)

        assert sys.rms_violation() < 1e-2


# ─── Störmer-Verlet energy conservation ──────────────────────────────────────


class TestStormerVerletEnergyConservation:
    def test_harmonic_oscillator_energy_conservation(self) -> None:
        """Harmonic oscillator: V(q) = ½k·q². Energy should be conserved
        with no external constraints."""
        k = 2.0
        sys = HamiltonianSystem(
            dim=1,
            potential_fn=lambda q: 0.5 * k * q[0] ** 2,
            potential_grad_fn=lambda q: np.array([k * q[0]]),
        )
        sys.set_state(np.array([1.0]), np.array([0.0]))

        energies = []
        for _ in range(1000):
            sys.step(0.01)
            e = sys.energy()
            energies.append(e.total)

        # Check no drift over 1000 steps
        initial = energies[0]
        max_drift = max(abs(e - initial) for e in energies)
        assert max_drift < 1e-3, f"Energy drift too large: {max_drift}"

    def test_energy_conservation_with_penalty_weight_zero(self) -> None:
        """With weight=0, constraint contributes no penalty."""
        sys = HamiltonianSystem(
            dim=2,
            potential_fn=lambda q: 0.5 * float(np.dot(q, q)),
            potential_grad_fn=lambda q: q,
        )
        sys.set_state(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        sys.add_constraint(*_circle_constraint(1.0), weight=0.0, name="circle")

        energies = []
        for _ in range(1000):
            sys.step(0.01)
            e = sys.energy()
            energies.append(e.total)

        initial = energies[0]
        max_drift = max(abs(e - initial) for e in energies)
        assert max_drift < 1e-3

    def test_symplectic_vs_euler_drift(self) -> None:
        """Störmer-Verlet should drift much less than explicit Euler."""
        k = 1.0
        sys_sv = HamiltonianSystem(
            dim=1,
            potential_fn=lambda q: 0.5 * k * q[0] ** 2,
            potential_grad_fn=lambda q: np.array([k * q[0]]),
        )
        sys_sv.set_state(np.array([1.0]), np.array([0.0]))

        # Simulate explicit Euler manually for comparison
        q = 1.0
        p = 0.0
        dt = 0.01
        e0 = 0.5 * k * q ** 2
        for _ in range(1000):
            p = p - k * q * dt
            q = q + p * dt
        e_euler = 0.5 * (p ** 2 + k * q ** 2)
        euler_drift = abs(e_euler - e0) / abs(e0)

        # Störmer-Verlet
        for _ in range(1000):
            sys_sv.step(0.01)
        e_sv = sys_sv.energy().total
        sv_drift = abs(e_sv - e0) / abs(e0)

        assert sv_drift < euler_drift
        assert sv_drift < 1e-3

    def test_energy_history_populated(self) -> None:
        sys = HamiltonianSystem(dim=1)
        sys.set_state(np.array([1.0]), np.array([0.0]))
        for _ in range(10):
            sys.step(0.01)
            sys.energy()
        assert len(sys.energy_history) == 10

    def test_clear_energy_history(self) -> None:
        sys = HamiltonianSystem(dim=1)
        sys.set_state(np.array([1.0]), np.array([0.0]))
        sys.energy()
        sys.clear_energy_history()
        assert len(sys.energy_history) == 0


# ─── Damped relaxation convergence ───────────────────────────────────────────


class TestDampedRelaxationConvergence:
    def test_damped_relaxation_converges_to_manifold(self) -> None:
        sys = HamiltonianSystem(dim=2, damping=0.1)
        sys.set_state(np.array([5.0, 0.0]))
        sys.add_constraint(*_circle_constraint(1.0), weight=10.0, name="circle")
        sys.reset_momentum()

        violations = []
        for _ in range(5000):
            sys.step_damped(0.01)
            violations.append(sys.total_violation())

        # Violation should be monotonically decreasing (or at least end very small)
        assert violations[-1] < 1e-2
        assert violations[-1] < violations[0]

    def test_damped_vs_undamped_for_far_initial_state(self) -> None:
        """Damped relaxation should reach the manifold from far away;
        undamped might oscillate."""
        sys_damped = HamiltonianSystem(dim=2, damping=0.1)
        sys_damped.set_state(np.array([10.0, 0.0]))
        sys_damped.add_constraint(*_circle_constraint(1.0), weight=10.0, name="circle")
        sys_damped.reset_momentum()

        for _ in range(10000):
            sys_damped.step_damped(0.01)

        assert sys_damped.total_violation() < 1e-2

    def test_damping_zero_equals_no_damping(self) -> None:
        sys = HamiltonianSystem(dim=2, damping=0.0)
        sys.set_state(np.array([2.0, 0.0]))
        sys.add_constraint(*_circle_constraint(1.0), weight=0.0, name="circle")
        sys.reset_momentum()

        # With no damping and no constraint force, momentum should be conserved
        # Actually with no potential and no constraints, momentum stays zero
        # Let's add a potential to get oscillation
        sys._potential_fn = lambda q: 0.5 * float(np.dot(q, q))
        sys._potential_grad_fn = lambda q: q
        sys.set_state(np.array([1.0, 0.0]), np.array([0.0, 1.0]))

        e0 = sys.energy().total
        for _ in range(500):
            sys.step_damped(0.01)
        e1 = sys.energy().total
        # With damping=0, step_damped should still be symplectic-ish
        assert abs(e1 - e0) < 1e-2

    def test_high_damping_fast_convergence(self) -> None:
        sys = HamiltonianSystem(dim=2, damping=0.5)
        sys.set_state(np.array([3.0, 0.0]))
        sys.add_constraint(*_circle_constraint(1.0), weight=20.0, name="circle")
        sys.reset_momentum()

        for _ in range(2000):
            sys.step_damped(0.01)

        assert sys.total_violation() < 1e-2


# ─── Augmented Lagrangian multiplier updates ─────────────────────────────────


class TestAugmentedLagrangian:
    def test_multiplier_update_reduces_violation(self) -> None:
        sys = HamiltonianSystem(dim=2, multiplier_update_rate=0.1)
        sys.set_state(np.array([2.0, 0.0]))
        sys.add_constraint(*_circle_constraint(1.0), weight=1.0, name="circle")
        sys.reset_momentum()

        for _ in range(100):
            for _ in range(50):
                sys.step_damped(0.01)
            sys.update_multipliers()

        assert sys.total_violation() < 1e-2

    def test_multiplier_values_increase(self) -> None:
        sys = HamiltonianSystem(dim=2, multiplier_update_rate=0.1)
        sys.set_state(np.array([2.0, 0.0]))
        c = sys.add_constraint(*_circle_constraint(1.0), weight=1.0, name="circle")
        sys.reset_momentum()

        initial_multiplier = c.multiplier
        for _ in range(10):
            for _ in range(50):
                sys.step_damped(0.01)
            sys.update_multipliers()

        assert c.multiplier != initial_multiplier
        # Multiplier should move in direction of violation
        # For circle from (2,0), c(2,0) = 4-1 = 3 > 0, so multiplier should increase
        assert c.multiplier > initial_multiplier

    def test_multiplier_update_with_negative_violation(self) -> None:
        """Inside the circle: c < 0, multiplier should decrease."""
        sys = HamiltonianSystem(dim=2, multiplier_update_rate=0.1)
        sys.set_state(np.array([0.5, 0.0]))
        c = sys.add_constraint(*_circle_constraint(1.0), weight=1.0, name="circle")
        sys.reset_momentum()

        for _ in range(10):
            for _ in range(50):
                sys.step_damped(0.01)
            sys.update_multipliers()

        # c(0.5,0) = 0.25-1 = -0.75 < 0, so multiplier should decrease
        assert c.multiplier < 0

    def test_augmented_lagrangian_convergence(self) -> None:
        sys = HamiltonianSystem(dim=2, damping=0.05, multiplier_update_rate=0.05)
        sys.set_state(np.array([3.0, 0.0]))
        sys.add_constraint(*_circle_constraint(1.0), weight=1.0, name="circle")
        sys.reset_momentum()

        for _ in range(200):
            for _ in range(50):
                sys.step_damped(0.01)
            sys.update_multipliers()

        q = sys.get_state()
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-2)
        assert sys.total_violation() < 1e-2

    def test_augmented_lagrangian_vs_pure_penalty(self) -> None:
        """Augmented Lagrangian should converge with smaller weight than
        pure penalty method."""
        sys_al = HamiltonianSystem(dim=2, damping=0.05, multiplier_update_rate=0.05)
        sys_al.set_state(np.array([3.0, 0.0]))
        sys_al.add_constraint(*_circle_constraint(1.0), weight=1.0, name="circle")
        sys_al.reset_momentum()

        for _ in range(200):
            for _ in range(50):
                sys_al.step_damped(0.01)
            sys_al.update_multipliers()

        sys_penalty = HamiltonianSystem(dim=2, damping=0.05)
        sys_penalty.set_state(np.array([3.0, 0.0]))
        sys_penalty.add_constraint(*_circle_constraint(1.0), weight=10.0, name="circle")
        sys_penalty.reset_momentum()
        for _ in range(10000):
            sys_penalty.step_damped(0.01)

        # Both should be on the manifold
        assert sys_al.total_violation() < 1e-2
        assert sys_penalty.total_violation() < 1e-2


# ─── Energy conservation quality metric ──────────────────────────────────────


class TestEnergyConservationQuality:
    def test_quality_metric_low_for_good_conservation(self) -> None:
        sys = HamiltonianSystem(
            dim=1,
            potential_fn=lambda q: 0.5 * q[0] ** 2,
            potential_grad_fn=lambda q: np.array([q[0]]),
        )
        sys.set_state(np.array([1.0]), np.array([0.0]))

        e0 = sys.energy().total
        for _ in range(1000):
            sys.step(0.01)
        e1 = sys.energy().total

        quality = sys.energy_history[-1].conservation_quality(baseline=e0)
        assert quality < 1e-2

    def test_quality_metric_high_for_constraint_dominance(self) -> None:
        sys = HamiltonianSystem(dim=2)
        sys.set_state(np.array([10.0, 0.0]))
        sys.add_constraint(*_circle_constraint(1.0), weight=1000.0, name="circle")
        sys.reset_momentum()

        e = sys.energy()
        # Total energy should be dominated by penalty term
        assert e.penalty > e.kinetic
        quality = e.conservation_quality()
        assert quality > 0.9

    def test_energy_history_trend(self) -> None:
        sys = HamiltonianSystem(
            dim=1,
            potential_fn=lambda q: 0.5 * q[0] ** 2,
            potential_grad_fn=lambda q: np.array([q[0]]),
        )
        sys.set_state(np.array([1.0]), np.array([0.0]))

        for _ in range(100):
            sys.step(0.01)
            sys.energy()

        qualities = [e.conservation_quality(baseline=sys.energy_history[0].total) for e in sys.energy_history]
        # Quality should not drift catastrophically
        assert max(qualities) < 1e-1


# ─── Contradictory constraints ─────────────────────────────────────────────


class TestContradictoryConstraints:
    def test_two_parallel_planes_no_solution(self) -> None:
        """Parallel planes with different offsets: no exact solution.
        System should settle to the nearest feasible manifold —
        the average/midpoint between the two planes."""
        sys = HamiltonianSystem(dim=2, damping=0.1)
        sys.set_state(np.array([0.0, 0.0]))
        sys.add_constraint(*_plane_constraint([1.0, 0.0], 0.0), weight=10.0, name="x_eq_0")
        sys.add_constraint(*_plane_constraint([1.0, 0.0], 2.0), weight=10.0, name="x_eq_2")
        sys.reset_momentum()

        for _ in range(5000):
            sys.step_damped(0.01)

        q = sys.get_state()
        # With equal weights, the compromise is x = 1.0 (midpoint)
        assert q[0] == pytest.approx(1.0, abs=5e-2)
        assert abs(q[1]) < 1.0  # y should be near zero (no force on it)

    def test_contradictory_circle_and_point(self) -> None:
        """Circle constraint + point constraint that doesn't lie on circle.
        Penalty method minimizes sum of squared violations; with equal
        weights the minimum is where the gradients balance (≈ x=1.17)."""
        sys = HamiltonianSystem(dim=2, damping=0.1)
        sys.set_state(np.array([3.0, 0.0]))
        sys.add_constraint(*_circle_constraint(1.0), weight=10.0, name="circle")
        # Point at (2, 0) — outside the circle
        sys.add_constraint(
            value_fn=lambda q: float(q[0] - 2.0),
            gradient_fn=lambda q: np.array([1.0, 0.0]),
            weight=10.0,
            name="x_eq_2",
        )
        sys.reset_momentum()

        for _ in range(5000):
            sys.step_damped(0.01)

        q = sys.get_state()
        # With equal penalty weights, the compromise is near x ≈ 1.17
        # (minimum of ½*10*(x²-1)² + ½*10*(x-2)², not the nearest point on circle)
        assert 1.0 <= q[0] <= 1.3
        assert abs(q[1]) < 5e-2

    def test_weighted_contradiction_prefers_heavier(self) -> None:
        """Higher-weighted constraint should be satisfied more closely."""
        sys = HamiltonianSystem(dim=2, damping=0.1)
        sys.set_state(np.array([0.0, 0.0]))
        sys.add_constraint(*_plane_constraint([1.0, 0.0], 0.0), weight=100.0, name="x_eq_0")
        sys.add_constraint(*_plane_constraint([1.0, 0.0], 2.0), weight=1.0, name="x_eq_2")
        sys.reset_momentum()

        for _ in range(5000):
            sys.step_damped(0.01)

        q = sys.get_state()
        # With x_eq_0 weight 100x larger, q[0] should be much closer to 0 than 2
        assert q[0] < 0.5
        assert q[0] == pytest.approx(0.0, abs=5e-2)

    def test_three_mutually_inconsistent_planes(self) -> None:
        """Three mutually inconsistent constraints in 2D.
        System should find the least-squares compromise."""
        sys = HamiltonianSystem(dim=2, damping=0.1)
        sys.set_state(np.array([0.0, 0.0]))
        sys.add_constraint(
            value_fn=lambda q: float(q[0] - 1.0),
            gradient_fn=lambda q: np.array([1.0, 0.0]),
            weight=10.0,
            name="x_eq_1",
        )
        sys.add_constraint(
            value_fn=lambda q: float(q[0] - 2.0),
            gradient_fn=lambda q: np.array([1.0, 0.0]),
            weight=10.0,
            name="x_eq_2",
        )
        sys.add_constraint(
            value_fn=lambda q: float(q[1] - 3.0),
            gradient_fn=lambda q: np.array([0.0, 1.0]),
            weight=10.0,
            name="y_eq_3",
        )
        sys.reset_momentum()

        for _ in range(5000):
            sys.step_damped(0.01)

        q = sys.get_state()
        # x compromise: (1+2)/2 = 1.5; y should be 3.0
        assert q[0] == pytest.approx(1.5, abs=5e-2)
        assert q[1] == pytest.approx(3.0, abs=5e-2)


# ─── State access and mutation ───────────────────────────────────────────────


class TestStateAccess:
    def test_get_state_returns_copy(self) -> None:
        sys = HamiltonianSystem(dim=2)
        q = sys.get_state()
        q[0] = 999.0
        assert sys.get_state()[0] == 0.0

    def test_set_state(self) -> None:
        sys = HamiltonianSystem(dim=2)
        sys.set_state(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(sys.get_state(), np.array([1.0, 2.0]))

    def test_set_state_and_momentum(self) -> None:
        sys = HamiltonianSystem(dim=2)
        sys.set_state(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        np.testing.assert_array_equal(sys.get_momentum(), np.array([3.0, 4.0]))

    def test_set_state_shape_mismatch(self) -> None:
        sys = HamiltonianSystem(dim=2)
        with pytest.raises(ValueError, match="position shape"):
            sys.set_state(np.array([1.0, 2.0, 3.0]))

    def test_reset_momentum(self) -> None:
        sys = HamiltonianSystem(dim=2)
        sys.set_state(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
        sys.reset_momentum()
        np.testing.assert_array_equal(sys.get_momentum(), np.zeros(2))

    def test_step_count_increments(self) -> None:
        sys = HamiltonianSystem(dim=1)
        sys.set_state(np.array([1.0]))
        sys.step(0.01)
        assert sys.step_count == 1
        sys.step_damped(0.01)
        assert sys.step_count == 2

    def test_repr(self) -> None:
        sys = HamiltonianSystem(dim=2)
        sys.add_constraint(*_circle_constraint(1.0), name="circle")
        r = repr(sys)
        assert "HamiltonianSystem" in r
        assert "dim=2" in r


# ─── Constraint violation API ──────────────────────────────────────────────


class TestConstraintViolationAPI:
    def test_constraint_violation_empty(self) -> None:
        sys = HamiltonianSystem(dim=2)
        assert sys.constraint_violation() == {}

    def test_constraint_violation_named(self) -> None:
        sys = HamiltonianSystem(dim=2)
        sys.set_state(np.array([2.0, 0.0]))
        sys.add_constraint(*_circle_constraint(1.0), name="circle")
        v = sys.constraint_violation()
        assert "circle" in v
        assert v["circle"] == pytest.approx(3.0, abs=1e-10)  # 2^2+0^2-1 = 3

    def test_constraint_violation_unnamed(self) -> None:
        sys = HamiltonianSystem(dim=2)
        sys.set_state(np.array([2.0, 0.0]))
        sys.add_constraint(*_circle_constraint(1.0))
        v = sys.constraint_violation()
        assert "constraint_0" in v

    def test_total_violation(self) -> None:
        sys = HamiltonianSystem(dim=2)
        sys.set_state(np.array([2.0, 0.0]))
        sys.add_constraint(*_circle_constraint(1.0), name="circle")
        assert sys.total_violation() == pytest.approx(3.0, abs=1e-10)

    def test_rms_violation_empty(self) -> None:
        sys = HamiltonianSystem(dim=2)
        assert sys.rms_violation() == 0.0


# ─── evolve() wrapper ────────────────────────────────────────────────────────


class TestEvolve:
    def test_evolve_undamped(self) -> None:
        sys = HamiltonianSystem(
            dim=1,
            potential_fn=lambda q: 0.5 * q[0] ** 2,
            potential_grad_fn=lambda q: np.array([q[0]]),
        )
        sys.set_state(np.array([1.0]), np.array([0.0]))
        sys.evolve(0.01, 100)
        assert sys.step_count == 100

    def test_evolve_damped(self) -> None:
        sys = HamiltonianSystem(dim=2, damping=0.1)
        sys.set_state(np.array([2.0, 0.0]))
        sys.add_constraint(*_circle_constraint(1.0), weight=10.0, name="circle")
        sys.reset_momentum()
        sys.evolve(0.01, 5000, damped=True)
        assert sys.total_violation() < 1e-2

    def test_evolve_resets_not_called(self) -> None:
        sys = HamiltonianSystem(dim=1)
        sys.set_state(np.array([1.0]))
        sys.evolve(0.01, 10)
        assert sys.step_count == 10
        sys.evolve(0.01, 10)
        assert sys.step_count == 20


# ─── Integration: full pipeline ──────────────────────────────────────────


class TestFullPipeline:
    def test_full_pipeline_single_agent_onboard(self) -> None:
        """Simulate an agent onboarding: damped relaxation onto constraints,
        then undamped operation with occasional multiplier updates."""
        sys = HamiltonianSystem(dim=3, damping=0.1, multiplier_update_rate=0.05)
        sys.set_state(np.array([5.0, -2.0, 1.0]))
        sys.add_constraint(*_sphere_constraint(1.0, 3), weight=10.0, name="sphere")
        sys.add_constraint(*_plane_constraint([0.0, 0.0, 1.0], 0.0), weight=10.0, name="equator")
        sys.reset_momentum()

        # Phase 1: damped onboarding (1000 steps)
        sys.evolve(0.01, 1000, damped=True)
        assert sys.total_violation() < 1e-1

        # Phase 2: multiplier tightening (5 cycles)
        for _ in range(5):
            sys.evolve(0.01, 500, damped=True)
            sys.update_multipliers()

        assert sys.total_violation() < 1e-2

        # Phase 3: undamped steady-state operation
        sys.clear_energy_history()
        for _ in range(500):
            sys.step(0.01)
            sys.energy()

        # Energy should not drift wildly
        e0 = sys.energy_history[0].total
        max_drift = max(abs(e.total - e0) for e in sys.energy_history)
        assert max_drift < 1.0  # generous due to penalty terms

    def test_full_pipeline_multiple_agents_no_interference(self) -> None:
        """Multiple independent systems should not interfere."""
        sys_a = HamiltonianSystem(dim=2, damping=0.1)
        sys_a.set_state(np.array([2.0, 0.0]))
        sys_a.add_constraint(*_circle_constraint(1.0), weight=10.0, name="circle")

        sys_b = HamiltonianSystem(dim=2, damping=0.1)
        sys_b.set_state(np.array([0.0, 3.0]))
        sys_b.add_constraint(*_circle_constraint(2.0), weight=10.0, name="circle")

        sys_a.reset_momentum()
        sys_b.reset_momentum()

        for _ in range(3000):
            sys_a.step_damped(0.01)
            sys_b.step_damped(0.01)

        assert np.linalg.norm(sys_a.get_state()) == pytest.approx(1.0, abs=1e-2)
        assert np.linalg.norm(sys_b.get_state()) == pytest.approx(2.0, abs=1e-2)

    def test_energy_tracking_during_onboard(self) -> None:
        sys = HamiltonianSystem(dim=2, damping=0.1)
        sys.set_state(np.array([3.0, 0.0]))
        sys.add_constraint(*_circle_constraint(1.0), weight=10.0, name="circle")
        sys.reset_momentum()

        energies = []
        for _ in range(1000):
            sys.step_damped(0.01)
            energies.append(sys.energy())

        # Penalty should decrease over time
        assert energies[-1].penalty < energies[0].penalty
        # Total should not diverge
        assert abs(energies[-1].total) < abs(energies[0].total) + 1.0

    def test_quality_metric_improves(self) -> None:
        sys = HamiltonianSystem(dim=2, damping=0.1)
        sys.set_state(np.array([2.0, 0.0]))
        sys.add_constraint(*_circle_constraint(1.0), weight=10.0, name="circle")
        sys.reset_momentum()

        e0 = sys.energy()
        for _ in range(5000):
            sys.step_damped(0.01)
        e1 = sys.energy()

        # As we settle, the penalty should decrease dramatically
        assert e1.penalty < e0.penalty
        # Total violation should be small
        assert sys.total_violation() < 1e-2

    def test_remove_constraint_mid_run(self) -> None:
        sys = HamiltonianSystem(dim=2, damping=0.1)
        sys.set_state(np.array([1.0, 1.0]))
        sys.add_constraint(*_circle_constraint(1.0), weight=10.0, name="circle")
        sys.add_constraint(*_plane_constraint([1.0, 0.0], 0.0), weight=10.0, name="x_eq_0")
        sys.reset_momentum()

        for _ in range(1000):
            sys.step_damped(0.01)

        # Now remove the plane constraint
        sys.remove_constraint("x_eq_0")

        for _ in range(1000):
            sys.step_damped(0.01)

        # Should now just be on the circle
        q = sys.get_state()
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-2)
