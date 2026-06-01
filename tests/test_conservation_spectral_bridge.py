#!/usr/bin/env python3
"""Tests for fleet/conservation_spectral_bridge.py."""
import numpy as np
import pytest

from fleet.conservation_spectral_bridge import (
    SpectralFingerprint,
    SpectralFleetRegistry,
    build_laplacian,
    collaborative_score,
    compute_fingerprint,
    conservation_ratio,
    eigenvalue_similarity,
    fiedler_alignment,
    flux_collaborative_intelligence,
    spectral_decompose,
)


class TestBuildLaplacian:
    def test_simple_triangle(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        l = build_laplacian(adj)
        expected = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
        assert np.allclose(l, expected)

    def test_isolated_node(self):
        adj = np.array([[0, 0], [0, 0]])
        l = build_laplacian(adj)
        assert np.allclose(l, np.zeros((2, 2)))


class TestSpectralDecompose:
    def test_triangle_eigenvalues(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        l = build_laplacian(adj)
        vals, vecs = spectral_decompose(l)
        assert len(vals) == 3
        assert vals[0] == pytest.approx(0, abs=1e-10)
        assert np.allclose(l @ vecs, vecs * vals)


class TestComputeFingerprint:
    def test_basic(self):
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        fp = compute_fingerprint("test", adj)
        assert fp.agent_name == "test"
        assert fp.dimension == 3
        assert fp.fiedler_value > 0

    def test_fiedler_vector(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        fp = compute_fingerprint("test", adj)
        assert len(fp.fiedler_vector) == 3


class TestEigenvalueSimilarity:
    def test_identical(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        a = compute_fingerprint("a", adj)
        b = compute_fingerprint("b", adj)
        sim = eigenvalue_similarity(a, b)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_different(self):
        a = compute_fingerprint("a", np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
        b = compute_fingerprint("b", np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        sim = eigenvalue_similarity(a, b)
        assert 0.0 < sim < 1.0


class TestConservationRatio:
    def test_triangle(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        fp = compute_fingerprint("t", adj)
        ratio = conservation_ratio(fp)
        assert ratio > 0

    def test_line_graph(self):
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        fp = compute_fingerprint("t", adj)
        ratio = conservation_ratio(fp)
        assert ratio > 0


class TestFiedlerAlignment:
    def test_same_graph(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        a = compute_fingerprint("a", adj)
        b = compute_fingerprint("b", adj)
        align = fiedler_alignment(a, b)
        assert abs(align) == pytest.approx(1.0, abs=1e-6)


class TestFluxCollaborativeIntelligence:
    def test_shape(self):
        a = compute_fingerprint("a", np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
        b = compute_fingerprint("b", np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        flux = flux_collaborative_intelligence(a, b)
        assert flux.shape == (3, 3)

    def test_collaborative_score(self):
        a = compute_fingerprint("a", np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
        b = compute_fingerprint("b", np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        score = collaborative_score(a, b)
        assert score >= 0


class TestSpectralFleetRegistry:
    def test_register_and_get(self):
        reg = SpectralFleetRegistry()
        fp = compute_fingerprint("A", np.array([[0, 1], [1, 0]]))
        reg.register(fp)
        assert reg.get("A") is not None
        assert reg.get("B") is None
        assert len(reg) == 1

    def test_alignment_matrix(self):
        reg = SpectralFleetRegistry()
        reg.register(compute_fingerprint("A", np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])))
        reg.register(compute_fingerprint("B", np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])))
        matrix = reg.alignment_matrix()
        assert len(matrix) == 3  # A-A, A-B, B-B

    def test_best_collaborator(self):
        reg = SpectralFleetRegistry()
        reg.register(compute_fingerprint("A", np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])))
        reg.register(compute_fingerprint("B", np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])))
        reg.register(compute_fingerprint("C", np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])))
        best = reg.best_collaborator("A", top_k=2)
        assert len(best) == 2
        assert best[0][0] in ("B", "C")

    def test_fiedler_route(self):
        reg = SpectralFleetRegistry()
        reg.register(compute_fingerprint("A", np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])))
        reg.register(compute_fingerprint("B", np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])))
        reg.register(compute_fingerprint("C", np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])))
        route = reg.fiedler_route("A", "B")
        assert isinstance(route, list)
