"""InformationGeometryBreeding — Natural gradient descent on statistical manifolds.

Replaces random mutation in the breeding loop with natural gradient steps
that respect the intrinsic geometry of agent parameter spaces.

Mathematical foundation:
- Statistical manifold: each agent is a point on a manifold parameterized by θ
- Fisher information matrix: I(θ) = E[∇log p(x|θ) ∇log p(x|θ)ᵀ]
- Natural gradient: θ̃ = θ - η · I(θ)⁻¹ ∇J(θ)
  (vs. Euclidean gradient: θ̃ = θ - η · ∇J(θ))
- This respects the metric structure — "distance" on the manifold is KL divergence,
  not Euclidean distance.

Applications in fleet breeding:
1. Mutation becomes a natural gradient step toward higher fitness
2. Crossover respects geodesic paths on the manifold
3. Diversity measured via Fisher-Rao metric (intrinsic distance)

Reference: Amari (2016) "Information Geometry and Its Applications"
"""

from __future__ import annotations

__all__ = [
    "InformationGeometryBreeder",
    "FisherMetric",
    "natural_gradient_step",
    "fisher_rao_distance",
]

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


# ── Fisher Metric ─────────────────────────────────────────────

@dataclass(frozen=True)
class FisherMetric:
    """Fisher information matrix for a parameterized distribution."""

    theta: np.ndarray          # parameter vector (d,)
    fisher_matrix: np.ndarray  # (d, d) positive semi-definite

    def __post_init__(self) -> None:
        # Ensure symmetry
        object.__setattr__(
            self, "fisher_matrix",
            (self.fisher_matrix + self.fisher_matrix.T) / 2.0,
        )

    @property
    def dim(self) -> int:
        return int(self.theta.shape[0])

    def natural_gradient(self, euclidean_gradient: np.ndarray) -> np.ndarray:
        """Convert Euclidean gradient to natural gradient via I(θ)⁻¹."""
        g = np.asarray(euclidean_gradient, dtype=np.float64)
        # Regularized pseudoinverse for numerical stability
        reg = self.fisher_matrix + 1e-4 * np.eye(self.dim)
        nat_grad = np.linalg.solve(reg, g)
        return nat_grad

    def local_distance(self, delta: np.ndarray) -> float:
        """Fisher-Rao local norm: √(δᵀ I(θ) δ)."""
        d = np.asarray(delta, dtype=np.float64)
        return float(np.sqrt(d @ self.fisher_matrix @ d))


# ── Fisher-Rao Distance ───────────────────────────────────────

def fisher_rao_distance(
    theta_a: np.ndarray,
    theta_b: np.ndarray,
    fisher_fn: Callable[[np.ndarray], np.ndarray],
    num_steps: int = 20,
) -> float:
    """Approximate Fisher-Rao distance along geodesic.

    Integrates √(θ̇ᵀ I(θ) θ̇) along a straight-line path in parameter space.
    More accurate methods use ODE integration on the manifold.
    """
    theta_a = np.asarray(theta_a, dtype=np.float64)
    theta_b = np.asarray(theta_b, dtype=np.float64)
    d = theta_b - theta_a

    total = 0.0
    for t in np.linspace(0, 1, num_steps):
        theta_t = theta_a + t * d
        I_t = fisher_fn(theta_t)
        # θ̇ = d (constant velocity in Euclidean space)
        speed = float(np.sqrt(d @ I_t @ d))
        total += speed

    # Trapezoidal rule
    dt = 1.0 / (num_steps - 1)
    return total * dt


def fisher_information_gaussian(
    mean: np.ndarray, cov: np.ndarray
) -> np.ndarray:
    """Fisher information matrix for multivariate Gaussian N(μ, Σ).

    For Gaussian: I(μ) = Σ⁻¹ (information about mean)
    Full I(μ, Σ) is block-diagonal with I(μ) and I(Σ)
    Here we return I(μ) for mean-parameterized agents.
    """
    cov = np.asarray(cov, dtype=np.float64)
    return np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))


# ── Natural Gradient Step ─────────────────────────────────────

def natural_gradient_step(
    theta: np.ndarray,
    grad_J: np.ndarray,
    fisher_fn: Callable[[np.ndarray], np.ndarray],
    step_size: float = 0.01,
    damping: float = 1e-4,
) -> np.ndarray:
    """Single natural gradient step: θ' = θ - η · I(θ)⁻¹ ∇J.

    Args:
        theta: current parameters
        grad_J: Euclidean gradient of objective J
        fisher_fn: I(θ) generator
        step_size: learning rate η
        damping: regularization for matrix inversion
    """
    theta = np.asarray(theta, dtype=np.float64)
    grad_J = np.asarray(grad_J, dtype=np.float64)
    I_theta = fisher_fn(theta)

    # Regularized solve
    reg = I_theta + damping * np.eye(I_theta.shape[0])
    nat_grad = np.linalg.solve(reg, grad_J)
    return theta + step_size * nat_grad


# ── Information Geometry Breeder ──────────────────────────────

class InformationGeometryBreeder:
    """Breeder that uses natural gradient for mutation and Fisher-Rao for diversity.

    Replaces random Gaussian mutation with directed natural gradient steps,
    and Euclidean distance with intrinsic Fisher-Rao distance.
    """

    def __init__(
        self,
        dim: int,
        fisher_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        step_size: float = 0.01,
        damping: float = 1e-4,
    ) -> None:
        self.dim = dim
        self.step_size = step_size
        self.damping = damping
        # Default: isotropic Fisher (Euclidean fallback)
        self.fisher_fn = fisher_fn or (lambda t: np.eye(dim))

    # ── mutation ────────────────────────────────────────────

    def mutate(
        self,
        parent: np.ndarray,
        fitness_gradient: np.ndarray,
    ) -> np.ndarray:
        """Natural gradient mutation: move uphill on the manifold."""
        return natural_gradient_step(
            theta=parent,
            grad_J=fitness_gradient,
            fisher_fn=self.fisher_fn,
            step_size=self.step_size,
            damping=self.damping,
        )

    def mutate_batch(
        self,
        parents: list[np.ndarray],
        gradients: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Vectorized mutation for a batch of parents."""
        return [
            self.mutate(p, g)
            for p, g in zip(parents, gradients)
        ]

    # ── crossover (geodesic interpolation) ─────────────────

    def geodesic_crossover(
        self,
        parent_a: np.ndarray,
        parent_b: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Offspring at α-fraction along the geodesic between parents.

        For Gaussian agents: geodesic is the mixture geodesic.
        For general manifolds: approximate with exponential map.
        """
        a = np.asarray(parent_a, dtype=np.float64)
        b = np.asarray(parent_b, dtype=np.float64)
        # Linear interpolation (first-order geodesic approx)
        # For true geodesic, would need exponential map
        child = (1 - alpha) * a + alpha * b
        # Project back via natural gradient to ensure manifold constraint
        # (simplified: just return interpolated point)
        return child

    # ── diversity ───────────────────────────────────────────

    def diversity_matrix(
        self,
        population: list[np.ndarray],
    ) -> np.ndarray:
        """Pairwise Fisher-Rao distance matrix."""
        n = len(population)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = fisher_rao_distance(
                    population[i], population[j], self.fisher_fn, num_steps=10
                )
                D[i, j] = d
                D[j, i] = d
        return D

    def novelty_score(
        self,
        agent: np.ndarray,
        population: list[np.ndarray],
        k: int = 3,
    ) -> float:
        """Average Fisher-Rao distance to k nearest neighbors."""
        dists = [
            fisher_rao_distance(agent, p, self.fisher_fn, num_steps=10)
            for p in population
        ]
        dists_sorted = sorted(dists)
        # Exclude self (distance 0)
        knn = [d for d in dists_sorted if d > 1e-10][:k]
        return float(np.mean(knn)) if knn else 0.0

    # ── metrics for monitoring ──────────────────────────────

    def compute_metrics(
        self,
        population: list[np.ndarray],
    ) -> dict[str, Any]:
        """Information geometry metrics for the population."""
        if not population:
            return {"n": 0}

        D = self.diversity_matrix(population)
        # Mean pairwise distance
        mask = np.triu(np.ones_like(D, dtype=bool), k=1)
        pairwise = D[mask]

        # Volume: determinant of centered covariance (in tangent space)
        # Approximate with Euclidean covariance
        X = np.stack(population)
        X_c = X - X.mean(axis=0)
        cov = X_c.T @ X_c / len(population)
        volume = float(np.linalg.det(cov + 1e-6 * np.eye(cov.shape[0])))

        return {
            "n": len(population),
            "mean_fisher_rao": float(np.mean(pairwise)) if len(pairwise) else 0.0,
            "max_fisher_rao": float(np.max(pairwise)) if len(pairwise) else 0.0,
            "diversity_volume": volume,
        }
