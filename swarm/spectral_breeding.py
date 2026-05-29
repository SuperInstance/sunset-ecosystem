"""swarm/spectral_breeding.py — Spectral-domain evolutionary algorithm.

Genomes are represented as complex spectra in the Fourier domain.
Crossover is spectral convolution (pointwise multiplication).
Mutation is harmonic shift, phase perturbation, and amplitude noise.
Selection evaluates phenotypes via inverse FFT.

This is genuinely novel: no existing evolutionary algorithm operates natively
in the frequency domain with spectral genomes.

Usage
-----
    from swarm.spectral_breeding import SpectralBreeder, SpectralGenome

    # Genome: 64-point complex spectrum
    breeder = SpectralBreeder(population_size=50, spectrum_size=64)
    breeder.initialize()

    # Task evaluates phenotype (IFFT output)
    def task_fn(phenotype):
        return float(np.sum(phenotype ** 2))

    for gen in range(100):
        breeder.evaluate(task_fn)
        breeder.select_and_breed()
        print(f"Gen {gen}: best={breeder.best_fitness:.4f}")
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class SpectralGenome:
    """Genome represented as a complex frequency spectrum."""
    spectrum: np.ndarray  # Complex-valued, shape (spectrum_size,)
    fitness: float = 0.0
    age: int = 0

    def __post_init__(self):
        if not np.iscomplexobj(self.spectrum):
            self.spectrum = self.spectrum.astype(np.complex128)

    @property
    def spectrum_size(self) -> int:
        return len(self.spectrum)

    @property
    def phenotype(self) -> np.ndarray:
        """Real-valued phenotype via inverse FFT."""
        # IFFT gives real output for Hermitian-symmetric spectra
        return np.fft.ifft(self.spectrum).real

    @property
    def magnitude(self) -> np.ndarray:
        """Magnitude spectrum."""
        return np.abs(self.spectrum)

    @property
    def phase(self) -> np.ndarray:
        """Phase spectrum."""
        return np.angle(self.spectrum)

    def copy(self) -> SpectralGenome:
        return SpectralGenome(
            spectrum=self.spectrum.copy(),
            fitness=self.fitness,
            age=self.age,
        )

    @staticmethod
    def random(spectrum_size: int, band_limit: Optional[float] = None) -> SpectralGenome:
        """Generate random spectrum with optional band-limiting."""
        # Random complex coefficients
        real = np.random.randn(spectrum_size)
        imag = np.random.randn(spectrum_size)
        spectrum = real + 1j * imag

        # Apply band-limit: zero out high frequencies
        if band_limit is not None:
            freqs = np.fft.fftfreq(spectrum_size)
            mask = np.abs(freqs) <= band_limit
            spectrum *= mask

        # Enforce Hermitian symmetry for real phenotype
        spectrum = _enforce_hermitian(spectrum)

        return SpectralGenome(spectrum=spectrum)

    @staticmethod
    def from_phenotype(phenotype: np.ndarray) -> SpectralGenome:
        """Create spectral genome from real phenotype."""
        spectrum = np.fft.fft(phenotype)
        return SpectralGenome(spectrum=spectrum)


def _enforce_hermitian(spectrum: np.ndarray) -> np.ndarray:
    """Enforce Hermitian symmetry so IFFT gives real output."""
    n = len(spectrum)
    result = spectrum.copy()
    # DC component must be real
    result[0] = result[0].real
    # Nyquist (if n is even)
    if n % 2 == 0:
        result[n // 2] = result[n // 2].real
    # Conjugate symmetry
    for i in range(1, (n + 1) // 2):
        result[n - i] = result[i].conjugate()
    return result


class SpectralCrossover:
    """Crossover via spectral convolution (multiplication in freq domain).

    In the frequency domain, convolution in phenotype space becomes
    pointwise multiplication. This is a fundamentally different
    crossover operation from traditional genetic algorithms.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha  # Blend factor

    def crossover(self, parent1: SpectralGenome,
                  parent2: SpectralGenome) -> Tuple[SpectralGenome, SpectralGenome]:
        """Spectral convolution crossover."""
        # Pointwise multiplication = convolution in phenotype space
        conv1 = parent1.spectrum * parent2.spectrum
        conv2 = parent2.spectrum * parent1.spectrum  # Same, but keep symmetry

        # Blend with parents
        blend1 = self.alpha * conv1 + (1 - self.alpha) * parent1.spectrum
        blend2 = self.alpha * conv2 + (1 - self.alpha) * parent2.spectrum

        child1 = SpectralGenome(spectrum=_enforce_hermitian(blend1))
        child2 = SpectralGenome(spectrum=_enforce_hermitian(blend2))
        return child1, child2


class SpectralMutation:
    """Mutation operations in the frequency domain."""

    def __init__(self,
                 harmonic_shift_rate: float = 0.1,
                 phase_noise_rate: float = 0.2,
                 amplitude_noise_rate: float = 0.2,
                 harmonic_shift_std: float = 1.0,
                 phase_noise_std: float = 0.3,
                 amplitude_noise_std: float = 0.1):
        self.harmonic_shift_rate = harmonic_shift_rate
        self.phase_noise_rate = phase_noise_rate
        self.amplitude_noise_rate = amplitude_noise_rate
        self.harmonic_shift_std = harmonic_shift_std
        self.phase_noise_std = phase_noise_std
        self.amplitude_noise_std = amplitude_noise_std

    def mutate(self, genome: SpectralGenome) -> SpectralGenome:
        """Apply spectral mutations."""
        child = genome.copy()
        spectrum = child.spectrum.copy()
        n = len(spectrum)

        # Harmonic shift: cyclically rotate spectrum
        if random.random() < self.harmonic_shift_rate:
            shift = int(np.random.normal(0, self.harmonic_shift_std))
            spectrum = np.roll(spectrum, shift)

        # Phase perturbation: add noise to phase
        if random.random() < self.phase_noise_rate:
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            phase += np.random.normal(0, self.phase_noise_std, n)
            spectrum = magnitude * np.exp(1j * phase)

        # Amplitude noise: perturb magnitudes
        if random.random() < self.amplitude_noise_rate:
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            noise = 1.0 + np.random.normal(0, self.amplitude_noise_std, n)
            magnitude *= np.clip(noise, 0.1, 10.0)
            spectrum = magnitude * np.exp(1j * phase)

        child.spectrum = _enforce_hermitian(spectrum)
        return child


class SpectralFilter:
    """Spectral filtering for controlled band-limiting."""

    @staticmethod
    def low_pass(genome: SpectralGenome, cutoff: float) -> SpectralGenome:
        """Apply low-pass filter in frequency domain."""
        result = genome.copy()
        freqs = np.fft.fftfreq(genome.spectrum_size)
        mask = np.abs(freqs) <= cutoff
        result.spectrum = _enforce_hermitian(genome.spectrum * mask)
        return result

    @staticmethod
    def high_pass(genome: SpectralGenome, cutoff: float) -> SpectralGenome:
        """Apply high-pass filter in frequency domain."""
        result = genome.copy()
        freqs = np.fft.fftfreq(genome.spectrum_size)
        mask = np.abs(freqs) >= cutoff
        result.spectrum = _enforce_hermitian(genome.spectrum * mask)
        return result

    @staticmethod
    def band_pass(genome: SpectralGenome, low: float, high: float) -> SpectralGenome:
        """Apply band-pass filter in frequency domain."""
        result = genome.copy()
        freqs = np.fft.fftfreq(genome.spectrum_size)
        mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
        result.spectrum = _enforce_hermitian(genome.spectrum * mask)
        return result


@dataclass
class SpectralBreeder:
    """Main breeding orchestrator using spectral genomes."""

    population_size: int = 50
    spectrum_size: int = 64
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    elitism_count: int = 2
    max_age: int = 50
    band_limit: Optional[float] = None

    population: List[SpectralGenome] = field(default_factory=list, repr=False)
    generation: int = 0
    best_fitness: float = 0.0
    best_genome: Optional[SpectralGenome] = None

    _crossover: SpectralCrossover = field(
        default_factory=lambda: SpectralCrossover(), repr=False
    )
    _mutation: SpectralMutation = field(
        default_factory=lambda: SpectralMutation(), repr=False
    )

    def initialize(self) -> None:
        """Initialize population with random spectra."""
        self.population = [
            SpectralGenome.random(self.spectrum_size, self.band_limit)
            for _ in range(self.population_size)
        ]
        self.generation = 0
        self.best_fitness = 0.0
        self.best_genome = None

    def evaluate(self, task_fn: Callable[[np.ndarray], float]) -> None:
        """Evaluate all genomes via their phenotypes."""
        for genome in self.population:
            genome.age += 1
            phenotype = genome.phenotype
            genome.fitness = task_fn(phenotype)
            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.best_genome = genome.copy()

    def select_and_breed(self) -> None:
        """Selection (tournament), spectral crossover, mutation, replacement."""
        # Sort by fitness descending
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        # Elitism
        new_population = [g.copy() for g in self.population[:self.elitism_count]]

        # Fill rest with offspring
        while len(new_population) < self.population_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < self.mutation_rate:
                child1 = self._mutation.mutate(child1)
            if random.random() < self.mutation_rate:
                child2 = self._mutation.mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        # Age-based culling
        self.population = [g for g in new_population if g.age < self.max_age]
        while len(self.population) < self.population_size:
            self.population.append(
                SpectralGenome.random(self.spectrum_size, self.band_limit)
            )

        self.generation += 1

    def _tournament_select(self, tournament_size: int = 3) -> SpectralGenome:
        """Tournament selection."""
        contestants = random.sample(
            self.population, min(tournament_size, len(self.population))
        )
        return max(contestants, key=lambda g: g.fitness)

    def get_stats(self) -> dict:
        """Get population statistics."""
        fitnesses = [g.fitness for g in self.population]
        magnitudes = [np.mean(g.magnitude) for g in self.population]
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            "mean_magnitude": sum(magnitudes) / len(magnitudes) if magnitudes else 0,
            "population_size": len(self.population),
        }

    def get_spectral_diversity(self) -> float:
        """Compute spectral diversity as mean pairwise spectral distance."""
        if len(self.population) < 2:
            return 0.0
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Euclidean distance in magnitude spectrum
                d = np.linalg.norm(
                    self.population[i].magnitude - self.population[j].magnitude
                )
                distances.append(d)
        return sum(distances) / len(distances) if distances else 0.0
