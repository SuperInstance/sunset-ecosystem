"""Tests for Spectral Breeding — Fourier-domain evolutionary algorithm.

Covers SpectralGenome, SpectralCrossover, SpectralMutation, SpectralFilter,
and SpectralBreeder.
"""

import numpy as np
import pytest

from swarm.spectral_breeding import (
    SpectralGenome,
    SpectralCrossover,
    SpectralMutation,
    SpectralFilter,
    SpectralBreeder,
    _enforce_hermitian,
)


class TestEnforceHermitian:
    def test_real_output(self):
        spectrum = np.random.randn(64) + 1j * np.random.randn(64)
        enforced = _enforce_hermitian(spectrum)
        phenotype = np.fft.ifft(enforced).real
        # Check IFFT gives essentially real output
        imag_component = np.max(np.abs(np.fft.ifft(enforced).imag))
        assert imag_component < 1e-10

    def test_dc_real(self):
        spectrum = np.random.randn(64) + 1j * np.random.randn(64)
        enforced = _enforce_hermitian(spectrum)
        assert np.isreal(enforced[0])
        if len(enforced) % 2 == 0:
            assert np.isreal(enforced[len(enforced) // 2])


class TestSpectralGenome:
    def test_random(self):
        genome = SpectralGenome.random(64)
        assert genome.spectrum_size == 64
        assert len(genome.spectrum) == 64
        assert np.iscomplexobj(genome.spectrum)

    def test_random_band_limit(self):
        genome = SpectralGenome.random(64, band_limit=0.25)
        freqs = np.fft.fftfreq(64)
        high_freq_mask = np.abs(freqs) > 0.25
        high_freq_energy = np.sum(np.abs(genome.spectrum[high_freq_mask]))
        assert high_freq_energy < 1e-10

    def test_phenotype_real(self):
        genome = SpectralGenome.random(64)
        phenotype = genome.phenotype
        assert len(phenotype) == 64
        # Should be essentially real
        assert (
            np.max(np.abs(phenotype.imag)) < 1e-10
            if hasattr(phenotype, "imag")
            else True
        )

    def test_magnitude(self):
        genome = SpectralGenome.random(64)
        mag = genome.magnitude
        assert len(mag) == 64
        assert np.all(mag >= 0)

    def test_phase(self):
        genome = SpectralGenome.random(64)
        phase = genome.phase
        assert len(phase) == 64

    def test_copy(self):
        genome = SpectralGenome.random(64)
        genome.fitness = 1.5
        copy = genome.copy()
        assert copy.fitness == 1.5
        assert np.allclose(copy.spectrum, genome.spectrum)
        # Ensure independent
        copy.spectrum[0] = 999
        assert genome.spectrum[0] != 999

    def test_from_phenotype(self):
        phenotype = np.sin(np.linspace(0, 4 * np.pi, 64))
        genome = SpectralGenome.from_phenotype(phenotype)
        reconstructed = genome.phenotype
        assert np.allclose(reconstructed, phenotype, atol=1e-10)


class TestSpectralCrossover:
    def test_crossover(self):
        p1 = SpectralGenome.random(64)
        p2 = SpectralGenome.random(64)
        crossover = SpectralCrossover(alpha=0.5)
        c1, c2 = crossover.crossover(p1, p2)
        assert c1.spectrum_size == 64
        assert c2.spectrum_size == 64
        # Phenotypes should be real
        assert np.max(np.abs(c1.phenotype.imag)) < 1e-10
        assert np.max(np.abs(c2.phenotype.imag)) < 1e-10

    def test_crossover_hermitian(self):
        p1 = SpectralGenome.random(64)
        p2 = SpectralGenome.random(64)
        crossover = SpectralCrossover(alpha=0.5)
        c1, c2 = crossover.crossover(p1, p2)
        # Verify Hermitian symmetry
        for child in [c1, c2]:
            n = len(child.spectrum)
            assert np.isreal(child.spectrum[0])
            if n % 2 == 0:
                assert np.isreal(child.spectrum[n // 2])
            for i in range(1, (n + 1) // 2):
                assert np.isclose(
                    child.spectrum[n - i], child.spectrum[i].conjugate(), atol=1e-10
                )


class TestSpectralMutation:
    def test_mutate(self):
        genome = SpectralGenome.random(64)
        mutator = SpectralMutation(
            harmonic_shift_rate=0.5,
            phase_noise_rate=0.5,
            amplitude_noise_rate=0.5,
        )
        child = mutator.mutate(genome)
        assert child.spectrum_size == 64
        assert np.iscomplexobj(child.spectrum)

    def test_mutate_preserves_hermitian(self):
        genome = SpectralGenome.random(64)
        mutator = SpectralMutation(
            harmonic_shift_rate=1.0,
            phase_noise_rate=1.0,
            amplitude_noise_rate=1.0,
        )
        child = mutator.mutate(genome)
        n = len(child.spectrum)
        assert np.isreal(child.spectrum[0])
        if n % 2 == 0:
            assert np.isreal(child.spectrum[n // 2])

    def test_phenotype_real_after_mutation(self):
        genome = SpectralGenome.random(64)
        mutator = SpectralMutation(
            harmonic_shift_rate=1.0,
            phase_noise_rate=1.0,
            amplitude_noise_rate=1.0,
        )
        child = mutator.mutate(genome)
        phenotype = child.phenotype
        assert np.max(np.abs(phenotype.imag)) < 1e-10


class TestSpectralFilter:
    def test_low_pass(self):
        genome = SpectralGenome.random(64)
        filtered = SpectralFilter.low_pass(genome, cutoff=0.25)
        freqs = np.fft.fftfreq(64)
        high_freq_mask = np.abs(freqs) > 0.25
        high_freq_energy = np.sum(np.abs(filtered.spectrum[high_freq_mask]))
        assert high_freq_energy < 1e-10

    def test_high_pass(self):
        genome = SpectralGenome.random(64)
        filtered = SpectralFilter.high_pass(genome, cutoff=0.1)
        freqs = np.fft.fftfreq(64)
        low_freq_mask = np.abs(freqs) < 0.1
        low_freq_energy = np.sum(np.abs(filtered.spectrum[low_freq_mask]))
        assert low_freq_energy < 1e-10

    def test_band_pass(self):
        genome = SpectralGenome.random(64)
        filtered = SpectralFilter.band_pass(genome, low=0.1, high=0.3)
        freqs = np.fft.fftfreq(64)
        out_of_band = (np.abs(freqs) < 0.1) | (np.abs(freqs) > 0.3)
        oob_energy = np.sum(np.abs(filtered.spectrum[out_of_band]))
        assert oob_energy < 1e-10


class TestSpectralBreeder:
    def test_initialize(self):
        breeder = SpectralBreeder(population_size=10, spectrum_size=32)
        breeder.initialize()
        assert len(breeder.population) == 10
        for genome in breeder.population:
            assert genome.spectrum_size == 32

    def test_evaluate(self):
        breeder = SpectralBreeder(population_size=10, spectrum_size=32)
        breeder.initialize()
        breeder.evaluate(lambda p: float(np.sum(p**2)))
        assert breeder.best_fitness > 0
        assert breeder.best_genome is not None

    def test_select_and_breed(self):
        breeder = SpectralBreeder(population_size=10, spectrum_size=32)
        breeder.initialize()
        breeder.evaluate(lambda p: float(np.sum(p**2)))
        breeder.select_and_breed()
        assert len(breeder.population) == 10
        assert breeder.generation == 1

    def test_full_evolution(self):
        breeder = SpectralBreeder(
            population_size=20, spectrum_size=32, mutation_rate=0.3, crossover_rate=0.7
        )
        breeder.initialize()
        best_history = []
        for gen in range(10):
            breeder.evaluate(lambda p: float(np.sum(p**2)))
            breeder.select_and_breed()
            best_history.append(breeder.best_fitness)
        assert breeder.generation == 10
        assert breeder.best_fitness > 0

    def test_stats(self):
        breeder = SpectralBreeder(population_size=10, spectrum_size=32)
        breeder.initialize()
        stats = breeder.get_stats()
        assert stats["generation"] == 0
        assert stats["population_size"] == 10

    def test_spectral_diversity(self):
        breeder = SpectralBreeder(population_size=10, spectrum_size=32)
        breeder.initialize()
        diversity = breeder.get_spectral_diversity()
        assert diversity >= 0

    def test_band_limit(self):
        breeder = SpectralBreeder(population_size=10, spectrum_size=64, band_limit=0.25)
        breeder.initialize()
        for genome in breeder.population:
            freqs = np.fft.fftfreq(64)
            high_freq = np.abs(freqs) > 0.25
            assert np.sum(np.abs(genome.spectrum[high_freq])) < 1e-10

    def test_elitism(self):
        breeder = SpectralBreeder(population_size=10, spectrum_size=32, elitism_count=2)
        breeder.initialize()
        breeder.evaluate(lambda p: float(np.sum(p**2)))
        best_before = breeder.best_fitness
        breeder.select_and_breed()
        breeder.evaluate(lambda p: float(np.sum(p**2)))
        assert breeder.best_fitness >= best_before * 0.9

    def test_age_culling(self):
        breeder = SpectralBreeder(population_size=10, spectrum_size=32, max_age=2)
        breeder.initialize()
        for gen in range(5):
            breeder.evaluate(lambda p: float(np.sum(p**2)))
            breeder.select_and_breed()
        assert all(g.age < 2 for g in breeder.population)
