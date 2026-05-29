"""swarm/nca_breeder.py — Neural Cellular Automata indirect encoding breeder.

Genomes are NCA rule parameters, not direct solutions.
Solutions are grown from a seed state via NCA iteration.
Crossover: rule interpolation, Mutation: rule perturbation.

This is genuinely novel: indirect encoding where the genotype is a CA rule
and the phenotype is the attractor after N steps. No existing EA uses NCA
as a developmental encoding for arbitrary optimization.

Usage
-----
    from swarm.nca_breeder import NCARule, NCABreeder

    # Genome: NCA rule parameters
    breeder = NCABreeder(
        grid_size=32,
        n_channels=3,        # RGB
        n_steps=64,          # CA iterations
        population_size=50,
    )
    breeder.initialize()

    # Task evaluates grown phenotype (attractor)
    def task_fn(phenotype):
        # phenotype: (n_channels, grid_size, grid_size) array
        target = load_target_image()
        return -float(np.mean((phenotype - target) ** 2))

    for gen in range(100):
        breeder.evaluate(task_fn)
        breeder.select_and_breed()
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class NCARule:
    """Neural Cellular Automata rule parameters.

    The genotype. Grows into a phenotype via CA iteration.
    """
    # Perception kernels (Sobel-ish filters)
    kernel_weights: np.ndarray  # Shape: (n_kernels, n_channels, k, k)

    # Update network weights (simple conv layers)
    update_weights: np.ndarray  # Shape: (n_channels, n_kernels * n_channels, 1, 1)
    update_bias: np.ndarray     # Shape: (n_channels,)

    # Alive mask threshold
    alive_threshold: float = 0.1

    # Number of channels and steps
    n_channels: int = 3
    n_kernels: int = 3  # Typically 3: identity + Sobel x + Sobel y
    kernel_size: int = 3
    n_steps: int = 64

    fitness: float = 0.0
    age: int = 0

    def copy(self) -> NCARule:
        return NCARule(
            kernel_weights=self.kernel_weights.copy(),
            update_weights=self.update_weights.copy(),
            update_bias=self.update_bias.copy(),
            alive_threshold=self.alive_threshold,
            n_channels=self.n_channels,
            n_kernels=self.n_kernels,
            kernel_size=self.kernel_size,
            n_steps=self.n_steps,
            fitness=self.fitness,
            age=self.age,
        )

    @staticmethod
    def random(
        n_channels: int = 3,
        n_kernels: int = 3,
        kernel_size: int = 3,
        n_steps: int = 64,
        weight_scale: float = 0.1,
    ) -> NCARule:
        """Generate random NCA rule."""
        kernel_weights = np.random.randn(
            n_kernels, n_channels, kernel_size, kernel_size
        ) * weight_scale

        # Standard NCA perception: identity + Sobel x + Sobel y
        if n_kernels >= 3:
            # Identity kernel
            kernel_weights[0] = 0.0
            for c in range(n_channels):
                kernel_weights[0, c, kernel_size // 2, kernel_size // 2] = 1.0

            # Sobel x
            if kernel_size == 3:
                sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                for c in range(n_channels):
                    kernel_weights[1, c] = sobel_x

                # Sobel y
                sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                for c in range(n_channels):
                    kernel_weights[2, c] = sobel_y

        update_weights = np.random.randn(
            n_channels, n_kernels * n_channels, 1, 1
        ) * weight_scale
        update_bias = np.zeros(n_channels)

        return NCARule(
            kernel_weights=kernel_weights,
            update_weights=update_weights,
            update_bias=update_bias,
            n_channels=n_channels,
            n_kernels=n_kernels,
            kernel_size=kernel_size,
            n_steps=n_steps,
        )

    def _perceive(self, state: np.ndarray) -> np.ndarray:
        """Perceive neighborhood using kernels."""
        # state: (n_channels, H, W)
        n_channels, H, W = state.shape
        perceptions = []

        for k in range(self.n_kernels):
            # Convolve each channel with this kernel
            perceived = np.zeros((n_channels, H, W))
            for c in range(n_channels):
                perceived[c] = self._conv2d(state[c], self.kernel_weights[k, c])
            perceptions.append(perceived)

        return np.stack(perceptions, axis=0)  # (n_kernels, n_channels, H, W)

    def _update(self, perceptions: np.ndarray) -> np.ndarray:
        """Compute state update from perceptions."""
        # perceptions: (n_kernels, n_channels, H, W)
        n_kernels, n_channels, H, W = perceptions.shape

        # Flatten to (n_kernels * n_channels, H, W)
        flat = perceptions.reshape(n_kernels * n_channels, H, W)

        # Apply update weights: for each output channel, weighted sum
        updates = np.zeros((n_channels, H, W))
        for c in range(n_channels):
            for i in range(n_kernels * n_channels):
                updates[c] += flat[i] * self.update_weights[c, i, 0, 0]
            updates[c] += self.update_bias[c]

        # Activation: sigmoid then scale
        return np.tanh(updates) * 0.5  # Bounded update

    def _conv2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution with edge padding."""
        k = kernel.shape[0]
        pad = k // 2
        H, W = image.shape

        # Pad
        padded = np.pad(image, pad, mode='edge')

        result = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                result[i, j] = np.sum(
                    padded[i:i+k, j:j+k] * kernel
                )
        return result

    def grow(self, seed: Optional[np.ndarray] = None) -> np.ndarray:
        """Grow phenotype from seed via NCA iteration.

        Returns final state after n_steps.
        """
        if seed is None:
            # Random seed with small activation
            seed = np.random.randn(self.n_channels, 16, 16) * 0.1

        state = seed.copy()

        for step in range(self.n_steps):
            # Perceive
            perceptions = self._perceive(state)

            # Update
            delta = self._update(perceptions)

            # Apply update
            state = state + delta

            # Stochastic update (only update some cells)
            mask = np.random.random(state.shape[1:]) < 0.5
            mask = mask[np.newaxis, :, :]
            state = np.where(mask, state, state - delta)  # Revert some

            # Alive mask: cells with low activation die
            alive = np.max(state, axis=0) > self.alive_threshold
            alive = alive[np.newaxis, :, :]
            state = np.where(alive, state, 0.0)

        return state

    @property
    def phenotype_shape(self) -> Tuple[int, int, int]:
        """Shape of grown phenotype (channels, H, W)."""
        return (self.n_channels, 16, 16)  # Default seed size


class NCAMutation:
    """Mutation for NCA rules."""

    def __init__(
        self,
        kernel_rate: float = 0.1,
        kernel_strength: float = 0.05,
        weight_rate: float = 0.1,
        weight_strength: float = 0.05,
        bias_rate: float = 0.1,
        bias_strength: float = 0.05,
    ):
        self.kernel_rate = kernel_rate
        self.kernel_strength = kernel_strength
        self.weight_rate = weight_rate
        self.weight_strength = weight_strength
        self.bias_rate = bias_rate
        self.bias_strength = bias_strength

    def mutate(self, rule: NCARule) -> NCARule:
        child = rule.copy()

        # Mutate kernel weights
        mask = np.random.random(child.kernel_weights.shape) < self.kernel_rate
        noise = np.random.normal(0, self.kernel_strength, child.kernel_weights.shape)
        child.kernel_weights += mask * noise

        # Mutate update weights
        mask = np.random.random(child.update_weights.shape) < self.weight_rate
        noise = np.random.normal(0, self.weight_strength, child.update_weights.shape)
        child.update_weights += mask * noise

        # Mutate bias
        mask = np.random.random(child.update_bias.shape) < self.bias_rate
        noise = np.random.normal(0, self.bias_strength, child.update_bias.shape)
        child.update_bias += mask * noise

        return child


class NCACrossover:
    """Crossover for NCA rules."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def crossover(self, p1: NCARule, p2: NCARule) -> Tuple[NCARule, NCARule]:
        # Interpolate all parameters
        blend = np.random.random(p1.kernel_weights.shape) < 0.5
        c1_kw = np.where(blend, p1.kernel_weights, p2.kernel_weights)
        c2_kw = np.where(blend, p2.kernel_weights, p1.kernel_weights)

        blend = np.random.random(p1.update_weights.shape) < 0.5
        c1_uw = np.where(blend, p1.update_weights, p2.update_weights)
        c2_uw = np.where(blend, p2.update_weights, p1.update_weights)

        c1_ub = (p1.update_bias + p2.update_bias) / 2
        c2_ub = c1_ub.copy()

        c1 = NCARule(
            kernel_weights=c1_kw,
            update_weights=c1_uw,
            update_bias=c1_ub,
            n_channels=p1.n_channels,
            n_kernels=p1.n_kernels,
            kernel_size=p1.kernel_size,
            n_steps=p1.n_steps,
        )
        c2 = NCARule(
            kernel_weights=c2_kw,
            update_weights=c2_uw,
            update_bias=c2_ub,
            n_channels=p1.n_channels,
            n_kernels=p1.n_kernels,
            kernel_size=p1.kernel_size,
            n_steps=p1.n_steps,
        )
        return c1, c2


@dataclass
class NCABreeder:
    """Breeder using NCA indirect encoding."""

    population_size: int = 50
    n_channels: int = 3
    n_kernels: int = 3
    kernel_size: int = 3
    n_steps: int = 64
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    elitism_count: int = 2
    max_age: int = 30

    population: List[NCARule] = field(default_factory=list, repr=False)
    generation: int = 0
    best_fitness: float = 0.0
    best_rule: Optional[NCARule] = None

    _mutation: NCAMutation = field(default_factory=lambda: NCAMutation(), repr=False)
    _crossover: NCACrossover = field(default_factory=lambda: NCACrossover(), repr=False)

    def initialize(self) -> None:
        """Initialize population with random NCA rules."""
        self.population = [
            NCARule.random(
                n_channels=self.n_channels,
                n_kernels=self.n_kernels,
                kernel_size=self.kernel_size,
                n_steps=self.n_steps,
            )
            for _ in range(self.population_size)
        ]
        self.generation = 0
        self.best_fitness = 0.0
        self.best_rule = None

    def evaluate(self, task_fn: Callable[[np.ndarray], float]) -> None:
        """Evaluate all rules by growing their phenotypes."""
        for rule in self.population:
            phenotype = rule.grow()
            rule.fitness = task_fn(phenotype)
            rule.age += 1
            if rule.fitness > self.best_fitness:
                self.best_fitness = rule.fitness
                self.best_rule = rule.copy()

    def select_and_breed(self) -> None:
        """Tournament selection, crossover, mutation, replacement."""
        self.population.sort(key=lambda r: r.fitness, reverse=True)
        new_pop = [r.copy() for r in self.population[:self.elitism_count]]

        while len(new_pop) < self.population_size:
            p1 = self._tournament_select()
            p2 = self._tournament_select()

            if random.random() < self.crossover_rate:
                c1, c2 = self._crossover.crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if random.random() < self.mutation_rate:
                c1 = self._mutation.mutate(c1)
            if random.random() < self.mutation_rate:
                c2 = self._mutation.mutate(c2)

            new_pop.append(c1)
            if len(new_pop) < self.population_size:
                new_pop.append(c2)

        # Age culling
        self.population = [r for r in new_pop if r.age < self.max_age]
        while len(self.population) < self.population_size:
            self.population.append(
                NCARule.random(
                    n_channels=self.n_channels,
                    n_kernels=self.n_kernels,
                    kernel_size=self.kernel_size,
                    n_steps=self.n_steps,
                )
            )

        self.generation += 1

    def _tournament_select(self, k: int = 3) -> NCARule:
        contestants = random.sample(self.population, min(k, len(self.population)))
        return max(contestants, key=lambda r: r.fitness)

    def get_stats(self) -> dict:
        fitnesses = [r.fitness for r in self.population]
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            "population_size": len(self.population),
        }
