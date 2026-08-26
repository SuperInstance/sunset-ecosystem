"""Tests for simd_ops.py — SIMD-style vector operations.

Run: python3 -m pytest tests/test_simd_ops.py -v --tb=short
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm.simd_ops import (
    SIMDOps,
    blockwise_cosine,
    blockwise_euclidean,
    blockwise_manhattan,
    batch_mutate,
    topk_select,
    vectorized_fitness,
)


class TestBlockwiseDistance:
    def test_euclidean_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert blockwise_euclidean(a, a) == pytest.approx(0.0, abs=1e-12)

    def test_euclidean_known(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert blockwise_euclidean(a, b) == pytest.approx(5.0)

    def test_euclidean_vs_numpy(self):
        a = np.random.randn(100)
        b = np.random.randn(100)
        simd = blockwise_euclidean(a, b)
        ref = np.linalg.norm(a - b)
        assert simd == pytest.approx(ref, rel=1e-10)

    def test_cosine_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert blockwise_cosine(a, a) == pytest.approx(0.0, abs=1e-12)

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert blockwise_cosine(a, b) == pytest.approx(1.0, abs=1e-12)

    def test_cosine_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert blockwise_cosine(a, b) == pytest.approx(2.0, abs=1e-12)

    def test_cosine_vs_numpy(self):
        a = np.random.randn(50)
        b = np.random.randn(50)
        simd = blockwise_cosine(a, b)
        # numpy cosine similarity
        dot = np.dot(a, b)
        ref = 1.0 - dot / (np.linalg.norm(a) * np.linalg.norm(b))
        assert simd == pytest.approx(ref, abs=1e-10)

    def test_manhattan_known(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert blockwise_manhattan(a, b) == pytest.approx(7.0)

    def test_manhattan_vs_numpy(self):
        a = np.random.randn(100)
        b = np.random.randn(100)
        assert blockwise_manhattan(a, b) == pytest.approx(
            np.sum(np.abs(a - b)), rel=1e-10
        )

    def test_different_block_sizes(self):
        a = np.random.randn(200)
        b = np.random.randn(200)
        d4 = blockwise_euclidean(a, b, block=4)
        d16 = blockwise_euclidean(a, b, block=16)
        d64 = blockwise_euclidean(a, b, block=64)
        assert d4 == pytest.approx(d16)
        assert d16 == pytest.approx(d64)


class TestBatchMutate:
    def test_shape_preserved(self):
        parents = np.random.randn(10, 20)
        children = batch_mutate(parents, noise_std=0.1)
        assert children.shape == parents.shape

    def test_values_change(self):
        parents = np.zeros((5, 10))
        children = batch_mutate(parents, noise_std=0.5, rng=np.random.default_rng(42))
        assert not np.allclose(children, parents)

    def test_deterministic_with_seed(self):
        parents = np.ones((3, 5))
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        c1 = batch_mutate(parents, noise_std=0.1, rng=rng1)
        c2 = batch_mutate(parents, noise_std=0.1, rng=rng2)
        np.testing.assert_allclose(c1, c2)


class TestVectorizedFitness:
    def test_dot(self):
        pop = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        obj = np.array([1.0, 0.0])
        scores = vectorized_fitness(pop, obj, metric="dot")
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)
        assert scores[2] == pytest.approx(1.0)

    def test_cosine(self):
        pop = np.array([[1.0, 0.0], [0.0, 1.0]])
        obj = np.array([1.0, 0.0])
        scores = vectorized_fitness(pop, obj, metric="cosine")
        # cosine similarity: 1.0 for [1,0], 0.0 for [0,1]
        assert scores[0] == pytest.approx(1.0, abs=1e-12)
        assert scores[1] == pytest.approx(0.0, abs=1e-12)

    def test_euclidean(self):
        pop = np.array([[0.0, 0.0], [3.0, 4.0]])
        obj = np.array([0.0, 0.0])
        scores = vectorized_fitness(pop, obj, metric="euclidean")
        assert scores[0] == pytest.approx(0.0, abs=1e-12)
        assert scores[1] == pytest.approx(-5.0)  # negative = farther


class TestTopKSelect:
    def test_selects_top(self):
        pop = np.array([[i] for i in range(10)])
        fitness = np.array([5.0, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0, 0.0, 6.0, 4.0])
        top, top_f = topk_select(pop, fitness, k=3)
        assert len(top) == 3
        np.testing.assert_allclose(top_f, np.array([9.0, 8.0, 7.0]))

    def test_k_larger_than_n(self):
        pop = np.array([[i] for i in range(5)])
        fitness = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        top, top_f = topk_select(pop, fitness, k=10)
        assert len(top) == 5

    def test_descending_order(self):
        pop = np.random.randn(20, 5)
        fitness = np.random.randn(20)
        top, top_f = topk_select(pop, fitness, k=5)
        assert np.all(np.diff(top_f) <= 0)  # descending


class TestSIMDOpsClass:
    def test_distance_matrix_euclidean(self):
        a = np.array([[0.0, 0.0], [3.0, 4.0]])
        b = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        D = SIMDOps.distance_matrix(a, b, metric="euclidean")
        assert D.shape == (2, 3)
        assert D[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert D[1, 1] == pytest.approx(0.0, abs=1e-12)
        assert D[0, 1] == pytest.approx(5.0)
        assert D[0, 2] == pytest.approx(10.0)

    def test_distance_matrix_cosine(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0, 0.0], [-1.0, 0.0]])
        D = SIMDOps.distance_matrix(a, b, metric="cosine")
        assert D[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert D[0, 1] == pytest.approx(2.0, abs=1e-12)

    def test_distance_matrix_dot(self):
        a = np.array([[1.0, 2.0]])
        b = np.array([[3.0, 4.0], [0.0, 1.0]])
        D = SIMDOps.distance_matrix(a, b, metric="dot")
        assert D[0, 0] == pytest.approx(11.0)
        assert D[0, 1] == pytest.approx(2.0)

    def test_batch_fitness_gradient(self):
        pop = np.array([[1.0, 2.0], [3.0, 4.0]])
        obj = np.array([5.0, 5.0])
        grads = SIMDOps.batch_fitness_gradient(pop, obj)
        assert grads.shape == (2, 2)
        np.testing.assert_allclose(grads[0], np.array([4.0, 3.0]))
        np.testing.assert_allclose(grads[1], np.array([2.0, 1.0]))

    def test_accelerate_crossover(self):
        a = np.array([0.0, 2.0, 4.0])
        b = np.array([2.0, 4.0, 6.0])
        child = SIMDOps.accelerate_crossover(a, b, alpha=0.5)
        np.testing.assert_allclose(child, np.array([1.0, 3.0, 5.0]))
