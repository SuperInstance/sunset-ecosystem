"""Tests for TDA Fitness Landscape — topological homology guidance.

Covers TDALandscape, PersistencePair, LandscapeGuide, and breeding guidance.
"""

import numpy as np
import pytest

from swarm.tda_landscape import (
    PersistencePair,
    TDALandscape,
    LandscapeGuide,
)


class TestPersistencePair:
    def test_persistence(self):
        pair = PersistencePair(birth=0.1, death=0.5, dimension=1)
        assert pair.persistence == 0.4

    def test_persistence_inf(self):
        pair = PersistencePair(birth=0.1, death=float('inf'), dimension=0)
        assert pair.persistence == 0.0

    def test_significant(self):
        pair = PersistencePair(birth=0.1, death=0.5, dimension=1)
        assert pair.is_significant(threshold=0.2) is True
        assert pair.is_significant(threshold=0.5) is False


class TestTDALandscape:
    def test_add_sample(self):
        tda = TDALandscape(dimension=2)
        tda.add_sample(np.array([0.0, 0.0]), fitness=1.0)
        assert len(tda._samples) == 1
        assert len(tda._fitnesses) == 1

    def test_get_samples(self):
        tda = TDALandscape(dimension=2)
        for i in range(5):
            tda.add_sample(np.array([float(i), float(i)]), fitness=float(i))
        samples = tda.get_samples()
        assert samples.shape == (5, 2)

    def test_get_fitnesses(self):
        tda = TDALandscape(dimension=2)
        for i in range(5):
            tda.add_sample(np.array([float(i), float(i)]), fitness=float(i))
        fitnesses = tda.get_fitnesses()
        assert len(fitnesses) == 5
        assert fitnesses[0] == 0.0

    def test_compute_homology_empty(self):
        tda = TDALandscape(dimension=2)
        result = tda.compute_homology()
        assert result["betti_0"] == 0
        assert result["betti_1"] == 0

    def test_compute_homology_few_samples(self):
        tda = TDALandscape(dimension=2)
        for i in range(2):
            tda.add_sample(np.array([float(i), float(i)]), fitness=float(i))
        result = tda.compute_homology()
        assert result["betti_0"] == 0
        assert result["betti_1"] == 0

    def test_compute_homology_many_samples(self):
        tda = TDALandscape(dimension=2)
        np.random.seed(42)
        for i in range(20):
            pos = np.random.randn(2)
            fitness = float(np.sum(pos ** 2))
            tda.add_sample(pos, fitness=fitness)
        result = tda.compute_homology()
        # Should have some components
        assert "betti_0" in result
        assert "betti_1" in result
        assert "significant_holes" in result

    def test_get_landscape_features(self):
        tda = TDALandscape(dimension=2)
        for i in range(10):
            tda.add_sample(np.array([float(i), float(i)]), fitness=float(i))
        features = tda.get_landscape_features()
        assert features["num_samples"] == 10
        assert "fitness_range" in features
        assert "fitness_std" in features

    def test_2d_samples(self):
        tda = TDALandscape(dimension=2)
        tda.add_sample(np.array([1.0, 2.0]), fitness=1.0)
        samples = tda.get_samples()
        assert samples.shape == (1, 2)

    def test_higher_dimension(self):
        tda = TDALandscape(dimension=3)
        for i in range(10):
            pos = np.random.randn(3)
            tda.add_sample(pos, fitness=float(np.sum(pos ** 2)))
        result = tda.compute_homology()
        assert "betti_0" in result
        assert "betti_1" in result


class TestLandscapeGuide:
    def test_recommend_explore_insufficient(self):
        tda = TDALandscape(dimension=2)
        guide = LandscapeGuide(tda)
        rec = guide.recommend_direction(np.array([0.0, 0.0]))
        assert rec["strategy"] == "explore"
        assert rec["confidence"] == 1.0

    def test_recommend_avoid(self):
        tda = TDALandscape(dimension=2)
        # Create a landscape with a hole: low fitness in center, high around it
        for i in range(8):
            angle = 2 * np.pi * i / 8
            pos = np.array([np.cos(angle), np.sin(angle)])
            tda.add_sample(pos, fitness=10.0)
        # Low fitness in center
        tda.add_sample(np.array([0.0, 0.0]), fitness=1.0)
        
        guide = LandscapeGuide(tda)
        rec = guide.recommend_direction(np.array([0.0, 0.0]))
        assert rec["strategy"] in ["avoid", "explore", "ridge"]

    def test_recommend_ridge(self):
        tda = TDALandscape(dimension=2)
        # High fitness at center
        for i in range(10):
            pos = np.random.randn(2) * 0.1
            tda.add_sample(pos, fitness=100.0)
        # Lower fitness around
        for i in range(10):
            pos = np.random.randn(2) + 2.0
            tda.add_sample(pos, fitness=10.0)
        
        guide = LandscapeGuide(tda)
        rec = guide.recommend_direction(np.array([0.0, 0.0]))
        assert rec["strategy"] in ["ridge", "exploit", "explore"]

    def test_get_avoidance_zones(self):
        tda = TDALandscape(dimension=2)
        for i in range(5):
            tda.add_sample(np.array([float(i), 0.0]), fitness=float(i))
        guide = LandscapeGuide(tda)
        zones = guide.get_avoidance_zones()
        assert isinstance(zones, list)

    def test_get_exploitation_zones(self):
        tda = TDALandscape(dimension=2)
        for i in range(10):
            tda.add_sample(np.array([float(i), 0.0]), fitness=float(i * 10))
        guide = LandscapeGuide(tda)
        zones = guide.get_exploitation_zones()
        assert isinstance(zones, list)
        assert len(zones) > 0

    def test_complex_landscape(self):
        """Test on a more complex, realistic landscape."""
        tda = TDALandscape(dimension=2)
        np.random.seed(42)
        
        # Create multiple peaks
        peaks = [(-2, -2), (2, 2), (-2, 2), (2, -2)]
        for px, py in peaks:
            for _ in range(15):
                pos = np.array([px, py]) + np.random.randn(2) * 0.5
                fitness = 100.0 - np.sum((pos - [px, py]) ** 2) * 10
                tda.add_sample(pos, fitness=max(0, fitness))
        
        # Valley points
        for _ in range(20):
            pos = np.random.randn(2) * 3
            fitness = 5.0
            tda.add_sample(pos, fitness=fitness)
        
        features = tda.get_landscape_features()
        assert features["num_samples"] == 80
        
        guide = LandscapeGuide(tda)
        # Check recommendations for different positions
        for pos in [np.array([0.0, 0.0]), np.array([2.0, 2.0]), np.array([-2.0, -2.0])]:
            rec = guide.recommend_direction(pos)
            assert rec["strategy"] in ["avoid", "exploit", "explore", "ridge", "unknown"]
            assert 0.0 <= rec["confidence"] <= 1.0
            assert len(rec["rationale"]) > 0
