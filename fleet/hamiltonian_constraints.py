"""Hamiltonian constraint satisfaction for agent state validation.

Implements Pattern 1 from the SuperInstance audit: encode agent behavioral
constraints as potential energy surfaces c(q) = 0. The system evolves toward
satisfaction via symplectic dynamics — Störmer-Verlet preserves energy
structure, Augmented Lagrangian tightens constraints, and damped relaxation
drives arbitrary states onto the constraint manifold.

Reference: constraint-hamiltonian pattern from SuperInstance ecosystem audit
(May 30, 2026).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ─── type aliases ──────────────────────────────────────────────────────────
ValueFn = Callable[[np.ndarray], float]
GradientFn = Callable[[np.ndarray], np.ndarray]


# ─── data classes ────────────────────────────────────────────────────────────

@dataclass
class Constraint:
    """A single constraint on agent state.

    Attributes:
        value_fn: Computes constraint violation c(q). c(q) = 0 means satisfied.
        gradient_fn: Computes ∇c(q), the gradient of the constraint surface.
        weight: Penalty weight w for the augmented term ½w·c².
        multiplier: Lagrange multiplier λ for the term λ·c.
        name: Optional human-readable identifier.
    """

    value_fn: ValueFn
    gradient_fn: GradientFn
    weight: float = 1.0
    multiplier: float = 0.0
    name: str = ""


@dataclass
class AugmentedEnergy:
    """Tracks the augmented Hamiltonian H = K + V + Σ(½w·c² + λ·c).

    Attributes:
        kinetic: Kinetic energy K = ½pᵀp (assuming unit mass).
        potential: Potential energy V(q) — user-defined cost landscape.
        penalty: Sum of penalty terms Σ ½w·c(q)².
        lagrangian: Sum of multiplier terms Σ λ·c(q).
        total: Augmented Hamiltonian H (auto-computed).
    """

    kinetic: float = 0.0
    potential: float = 0.0
    penalty: float = 0.0
    lagrangian: float = 0.0
    total: float = 0.0

    def __post_init__(self):
        self.total = self.kinetic + self.potential + self.penalty + self.lagrangian

    def conservation_quality(self, baseline: Optional[float] = None) -> float:
        """Return a conservation quality metric.

        If baseline is provided, returns |H - baseline| / |baseline|.
        Otherwise returns |penalty + lagrangian| / |total| as a ratio
        indicating how much constraint terms dominate the Hamiltonian.
        """
        if baseline is not None and baseline != 0.0:
            return abs(self.total - baseline) / abs(baseline)
        if self.total == 0.0:
            return 0.0
        return abs(self.penalty + self.lagrangian) / abs(self.total)


@dataclass
class SystemState:
    """Internal state of the Hamiltonian system."""

    position: np.ndarray
    momentum: np.ndarray
    constraints: list[Constraint] = field(default_factory=list)
    step_count: int = 0


# ─── HamiltonianSystem ───────────────────────────────────────────────────────

class HamiltonianSystem:
    """Hamiltonian constraint dynamics for agent state validation.

    Supports multiple constraints on a single agent state. Uses symplectic
    Störmer-Verlet integration for long-term stability, augmented
    Lagrangian for constraint enforcement, and damped relaxation for
    initialization onto the feasible manifold.
    """

    def __init__(
        self,
        dim: int,
        potential_fn: Optional[Callable[[np.ndarray], float]] = None,
        potential_grad_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        damping: float = 0.1,
        multiplier_update_rate: float = 0.01,
        state: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the Hamiltonian system.

        Args:
            dim: Dimensionality of the agent state vector q.
            potential_fn: V(q) — optional external potential energy.
            potential_grad_fn: ∇V(q) — gradient of external potential.
            damping: Damping coefficient for relaxation steps.
            multiplier_update_rate: Rate λ update in augmented Lagrangian.
            state: Initial position vector (defaults to zeros).
        """
        self.dim = dim
        self._potential_fn = potential_fn or (lambda q: 0.0)
        self._potential_grad_fn = potential_grad_fn or (lambda q: np.zeros(dim))
        self.damping = damping
        self.multiplier_update_rate = multiplier_update_rate

        q0 = np.zeros(dim, dtype=float) if state is None else np.asarray(state, dtype=float)
        if q0.shape != (dim,):
            raise ValueError(f"state shape {q0.shape} != ({dim},)")

        self._state = SystemState(position=q0.copy(), momentum=np.zeros(dim, dtype=float))
        self._energy_history: list[AugmentedEnergy] = []

    # ── constraint management ───────────────────────────────────────────────

    def add_constraint(
        self,
        value_fn: ValueFn,
        gradient_fn: GradientFn,
        weight: float = 1.0,
        multiplier: float = 0.0,
        name: str = "",
    ) -> Constraint:
        """Add a new constraint to the system."""
        c = Constraint(
            value_fn=value_fn,
            gradient_fn=gradient_fn,
            weight=weight,
            multiplier=multiplier,
            name=name,
        )
        self._state.constraints.append(c)
        return c

    def remove_constraint(self, name: str) -> bool:
        """Remove a constraint by name. Returns True if found."""
        for i, c in enumerate(self._state.constraints):
            if c.name == name:
                self._state.constraints.pop(i)
                return True
        return False

    def list_constraints(self) -> list[Constraint]:
        """Return a copy of the current constraint list."""
        return list(self._state.constraints)

    def clear_constraints(self) -> None:
        """Remove all constraints."""
        self._state.constraints.clear()

    # ── core dynamics ───────────────────────────────────────────────────────

    def _compute_constraint_forces(self, q: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Compute constraint forces and energy contributions.

        Returns:
            force: Total constraint force on the agent state (negative gradient).
            penalty: Σ ½w·c².
            lagrangian: Σ λ·c.
        """
        force = np.zeros(self.dim, dtype=float)
        penalty = 0.0
        lagrangian = 0.0
        max_force_mag = 1000.0  # Clamp to prevent numerical explosion

        for c in self._state.constraints:
            val = c.value_fn(q)
            grad = c.gradient_fn(q)
            # Augmented Lagrangian force: -∇(½w·c² + λ·c) = -(w·c + λ)·∇c
            raw_force = (c.weight * val + c.multiplier) * grad
            force_mag = np.linalg.norm(raw_force)
            if force_mag > max_force_mag:
                raw_force = raw_force * (max_force_mag / force_mag)
            force -= raw_force
            penalty += 0.5 * c.weight * val * val
            lagrangian += c.multiplier * val

        return force, penalty, lagrangian

    def _potential_force(self, q: np.ndarray) -> np.ndarray:
        """Return the negative gradient of the external potential."""
        return -self._potential_grad_fn(q)

    def step(self, dt: float) -> None:
        """Symplectic Störmer-Verlet integration step (undamped).

        Preserves the Hamiltonian structure over long timescales.
        Uses the standard leapfrog scheme:
            p_{n+½} = p_n - ½dt·∇H(q_n)
            q_{n+1} = q_n + dt·p_{n+½}
            p_{n+1} = p_{n+½} - ½dt·∇H(q_{n+1})
        """
        q = self._state.position
        p = self._state.momentum

        # Half-step momentum update
        f_ext = self._potential_force(q)
        f_c, _, _ = self._compute_constraint_forces(q)
        p_half = p + 0.5 * dt * (f_ext + f_c)

        # Full-step position update
        q_new = q + dt * p_half

        # Half-step momentum update with new position
        f_ext_new = self._potential_force(q_new)
        f_c_new, _, _ = self._compute_constraint_forces(q_new)
        p_new = p_half + 0.5 * dt * (f_ext_new + f_c_new)

        self._state.position = q_new
        self._state.momentum = p_new
        self._state.step_count += 1

    def step_damped(self, dt: float) -> None:
        """Damped relaxation step — drives state onto constraint manifold.

        Uses a modified Verlet scheme with strong per-step velocity decay
        (damping is applied as a fraction per step, not per unit time)
        plus a small position-level stabilization toward the manifold.
        """
        q = self._state.position.copy()
        p = self._state.momentum.copy()

        # Strong per-step damping (damping coefficient is fraction per step)
        beta = max(0.0, 1.0 - self.damping)
        p_damped = p * beta

        # Symplectic step with damped momentum
        f_ext = self._potential_force(q)
        f_c, _, _ = self._compute_constraint_forces(q)
        p_half = p_damped + 0.5 * dt * (f_ext + f_c)

        q_new = q + dt * p_half

        f_ext_new = self._potential_force(q_new)
        f_c_new, _, _ = self._compute_constraint_forces(q_new)
        p_new = p_half + 0.5 * dt * (f_ext_new + f_c_new)

        # Final damping
        p_new = p_new * beta

        # Position-level stabilization: small direct gradient step
        # This accelerates convergence without breaking dynamics
        if self.damping > 0:
            _, penalty, _ = self._compute_constraint_forces(q_new)
            if penalty > 0.0:
                # Coupled Gauss-Newton stabilization (proper multi-constraint step)
                J_rows = []
                w_vals = []
                c_vals = []
                for c in self._state.constraints:
                    val = c.value_fn(q_new)
                    grad = c.gradient_fn(q_new)
                    g_norm = np.linalg.norm(grad)
                    if g_norm > 1e-8:
                        J_rows.append(grad / g_norm)
                        w_vals.append(c.weight)
                        c_vals.append(val * g_norm)  # scaled violation
                if J_rows:
                    J = np.stack(J_rows)
                    W = np.diag(w_vals)
                    c_vec = np.array(c_vals)
                    # Levenberg-Marquardt: (J^T W J + eps I)^{-1} J^T W c
                    JtWJ = J.T @ W @ J + 1e-6 * np.eye(self.dim)
                    try:
                        delta = -np.linalg.solve(JtWJ, J.T @ W @ c_vec)
                        q_new = q_new + 0.005 * delta  # small step
                    except np.linalg.LinAlgError:
                        pass  # Skip stabilization if singular

        self._state.position = q_new
        self._state.momentum = p_new
        self._state.step_count += 1

    def evolve(self, dt: float, steps: int, damped: bool = False) -> None:
        """Evolve the system for a given number of steps.

        Args:
            dt: Time step size.
            steps: Number of steps to take.
            damped: If True, use step_damped() instead of step().
        """
        step_fn = self.step_damped if damped else self.step
        for _ in range(steps):
            step_fn(dt)

    def update_multipliers(self) -> None:
        """Update Lagrange multipliers via augmented Lagrangian iteration.

        λ_{k+1} = λ_k + ρ·w·c(q) where ρ is the update rate.
        This progressively tightens constraints without brute penalties.
        """
        q = self._state.position
        for c in self._state.constraints:
            val = c.value_fn(q)
            c.multiplier += self.multiplier_update_rate * c.weight * val

    # ── energy and violation ────────────────────────────────────────────────

    def energy(self) -> AugmentedEnergy:
        """Compute the current augmented Hamiltonian."""
        q = self._state.position
        p = self._state.momentum

        kinetic = 0.5 * float(np.dot(p, p))
        potential = self._potential_fn(q)
        _, penalty, lagrangian = self._compute_constraint_forces(q)
        # Note: _compute_constraint_forces returns force, but we only need energies
        # Recompute penalty/lagrangian cleanly
        penalty = 0.0
        lagrangian = 0.0
        for c in self._state.constraints:
            val = c.value_fn(q)
            penalty += 0.5 * c.weight * val * val
            lagrangian += c.multiplier * val

        total = kinetic + potential + penalty + lagrangian
        ae = AugmentedEnergy(
            kinetic=kinetic,
            potential=potential,
            penalty=penalty,
            lagrangian=lagrangian,
            total=total,
        )
        self._energy_history.append(ae)
        return ae

    def constraint_violation(self) -> dict[str, float]:
        """Compute per-constraint violation magnitudes.

        Returns a dict mapping constraint name → |c(q)|.
        """
        q = self._state.position
        violations: dict[str, float] = {}
        for i, c in enumerate(self._state.constraints):
            key = c.name if c.name else f"constraint_{i}"
            violations[key] = abs(c.value_fn(q))
        return violations

    def total_violation(self) -> float:
        """Sum of absolute constraint violations."""
        return sum(self.constraint_violation().values())

    def rms_violation(self) -> float:
        """Root-mean-square of constraint violations."""
        vals = list(self.constraint_violation().values())
        if not vals:
            return 0.0
        return math.sqrt(sum(v * v for v in vals) / len(vals))

    # ── state access ──────────────────────────────────────────────────────────

    def get_state(self) -> np.ndarray:
        """Return a copy of the current position vector."""
        return self._state.position.copy()

    def get_momentum(self) -> np.ndarray:
        """Return a copy of the current momentum vector."""
        return self._state.momentum.copy()

    def set_state(self, position: np.ndarray, momentum: Optional[np.ndarray] = None) -> None:
        """Set position and optionally momentum."""
        pos = np.asarray(position, dtype=float)
        if pos.shape != (self.dim,):
            raise ValueError(f"position shape {pos.shape} != ({self.dim},)")
        self._state.position = pos.copy()
        if momentum is not None:
            mom = np.asarray(momentum, dtype=float)
            if mom.shape != (self.dim,):
                raise ValueError(f"momentum shape {mom.shape} != ({self.dim},)")
            self._state.momentum = mom.copy()

    def reset_momentum(self) -> None:
        """Zero out momentum (useful before damped relaxation)."""
        self._state.momentum = np.zeros(self.dim, dtype=float)

    @property
    def step_count(self) -> int:
        return self._state.step_count

    @property
    def energy_history(self) -> list[AugmentedEnergy]:
        """Return recorded energy history."""
        return list(self._energy_history)

    def clear_energy_history(self) -> None:
        """Clear recorded energy history."""
        self._energy_history.clear()

    def __repr__(self) -> str:
        return (
            f"HamiltonianSystem(dim={self.dim}, "
            f"constraints={len(self._state.constraints)}, "
            f"steps={self.step_count}, "
            f"total_violation={self.total_violation():.6f})"
        )
