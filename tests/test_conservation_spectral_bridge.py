"""tests/test_conservation_spectral_bridge.py — Test suite for the Conservation Spectral Bridge.

Covers:
- SpectralFingerprint construction from adjacency and capabilities
- Pure-Python spectral math correctness (Laplacian, eigendecomposition)
- Conservation ratio computation
- Spectral alignment / diversity scoring
- Parent selection with diversity threshold
- Anomaly detection via conservation ratio monitoring
- Archive serialization/deserialization
- Integration with SpectralBreederDiversity
- ConservationSpectralEngine fallback behavior
"""

import numpy as np
import pytest

from fleet.conservation_spectral_bridge import (
    SpectralFingerprint,
    SpectralAlignmentScorer,
    ConservationRatioMonitor,
    SpectralBreederDiversity,
    ConservationSpectralEngine,
    _laplacian,
    _eigendecompose,
    _conservation_ratio,
    _spectral_alignment,
    _fiedler_vector,
)


# ---------------------------------------------------------------------------
# Core spectral math


class TestSpectralMath:
    """Test the pure-Python fallback implementations."""

    def test_laplacian_diagonal_is_degree(self):
        adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        L = _laplacian(adj)
        # Degrees: [2, 1, 1]
        assert np.allclose(np.diag(L), [2, 1, 1])

    def test_laplacian_off_diagonal_is_negative_adjacency(self):
        adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        L = _laplacian(adj)
        assert L[0, 1] == -1
        assert L[1, 0] == -1
        assert L[0, 2] == -1

    def test_eigendecompose_sorted_ascending(self):
        adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        L = _laplacian(adj)
        vals, vecs = _eigendecompose(L)
        # Eigenvalues should be sorted ascending
        assert np.all(np.diff(vals) >= -1e-10)
        # Smallest eigenvalue of Laplacian is always 0
        assert abs(vals[0]) < 1e-10

    def test_fiedler_vector_from_eigenvectors(self):
        adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        L = _laplacian(adj)
        vals, vecs = _eigendecompose(L)
        fiedler = _fiedler_vector(vecs)
        assert len(fiedler) == 3
        assert np.allclose(fiedler, vecs[:, 1])

    def test_conservation_ratio_non_negative(self):
        adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        L = _laplacian(adj)
        vals, vecs = _eigendecompose(L)
        cr = _conservation_ratio(vals, vecs, adj)
        assert cr >= 0.0

    def test_spectral_alignment_identical(self):
        adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        L = _laplacian(adj)
        vals, _ = _eigendecompose(L)
        alignment = _spectral_alignment(vals, vals)
        assert abs(alignment - 1.0) < 1e-10

    def test_spectral_alignment_orthogonal(self):
        # Two genuinely different graphs → low alignment
        # Triangle vs line graph
        adj_a = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)  # triangle
        adj_b = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)  # line/path
        vals_a, _ = _eigendecompose(_laplacian(adj_a))
        vals_b, _ = _eigendecompose(_laplacian(adj_b))
        alignment = _spectral_alignment(vals_a, vals_b)
        assert alignment < 0.95  # Should be less than identical

    def test_spectral_alignment_padded(self):
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([0.0, 1.0])
        alignment = _spectral_alignment(a, b)
        assert 0.0 <= alignment <= 1.0


# ---------------------------------------------------------------------------
# SpectralFingerprint


class TestSpectralFingerprint:
    def test_from_adjacency_basic(self):
        adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        fp = SpectralFingerprint.from_adjacency("test", adj)
        assert fp.agent_id == "test"
        assert len(fp.eigenvalues) == 3
        assert fp.conservation_ratio >= 0.0
        assert fp.spectral_gap >= 0.0
        assert len(fp.fiedler_vector) == 3

    def test_from_agent_capabilities(self):
        fp = SpectralFingerprint.from_agent(
            agent_id="agent_1",
            capabilities=["vision", "audio", "reasoning"],
        )
        assert fp.agent_id == "agent_1"
        assert len(fp.eigenvalues) == 3
        assert fp.metadata["capabilities"] == ["vision", "audio", "reasoning"]

    def test_from_agent_with_links(self):
        fp = SpectralFingerprint.from_agent(
            agent_id="agent_2",
            capabilities=["vision", "audio", "reasoning"],
            capability_links=[("vision", "reasoning", 0.8)],
        )
        assert fp.agent_id == "agent_2"
        # The adjacency should have the link weight
        adj = fp.adjacency_matrix
        i_vision = 0
        i_reasoning = 2
        assert adj[i_vision, i_reasoning] == 0.8
        assert adj[i_reasoning, i_vision] == 0.8

    def test_alignment_coefficient(self):
        adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        fp = SpectralFingerprint.from_adjacency("test", adj)
        alpha = fp.alignment_coefficient
        assert alpha >= 0.0 or alpha == float("inf")

    def test_to_dict_roundtrip(self):
        adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        fp = SpectralFingerprint.from_adjacency("test", adj)
        d = fp.to_dict()
        fp2 = SpectralFingerprint.from_dict(d)
        assert fp2.agent_id == fp.agent_id
        assert np.allclose(fp2.eigenvalues, fp.eigenvalues)
        assert fp2.conservation_ratio == fp.conservation_ratio
        assert fp2.spectral_gap == fp.spectral_gap

    def test_to_dict_jsonable(self):
        adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        fp = SpectralFingerprint.from_adjacency("test", adj)
        d = fp.to_dict()
        # All values should be JSON-serializable
        import json

        json_str = json.dumps(d)
        assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# SpectralAlignmentScorer


class TestSpectralAlignmentScorer:
    def test_score_identical(self):
        adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        fp1 = SpectralFingerprint.from_adjacency("a", adj)
        fp2 = SpectralFingerprint.from_adjacency("b", adj)
        score = SpectralAlignmentScorer.score(fp1, fp2)
        assert abs(score) < 1e-10  # identical graphs = 0 diversity

    def test_score_different(self):
        # Triangle vs line graph (genuinely different spectra)
        adj_a = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        adj_b = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        fp_a = SpectralFingerprint.from_adjacency("a", adj_a)
        fp_b = SpectralFingerprint.from_adjacency("b", adj_b)
        score = SpectralAlignmentScorer.score(fp_a, fp_b)
        assert score > 0.01  # Different graphs should have some diversity

    def test_score_batch_symmetric(self):
        adj_a = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        adj_b = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        adj_c = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        fp_a = SpectralFingerprint.from_adjacency("a", adj_a)
        fp_b = SpectralFingerprint.from_adjacency("b", adj_b)
        fp_c = SpectralFingerprint.from_adjacency("c", adj_c)
        mat = SpectralAlignmentScorer.score_batch([fp_a, fp_b, fp_c])
        assert mat.shape == (3, 3)
        assert np.allclose(np.diag(mat), 0.0)
        assert np.allclose(mat, mat.T)

    def test_select_diverse_parents_basic(self):
        adj_a = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        adj_b = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        adj_c = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        fp_a = SpectralFingerprint.from_adjacency("a", adj_a)
        fp_b = SpectralFingerprint.from_adjacency("b", adj_b)
        fp_c = SpectralFingerprint.from_adjacency("c", adj_c)
        parents = SpectralAlignmentScorer.select_diverse_parents(
            [fp_a, fp_b, fp_c], n_parents=2, min_diversity=0.1
        )
        assert len(parents) == 2
        assert "a" in parents or "b" in parents or "c" in parents

    def test_select_diverse_parents_insufficient_pool(self):
        adj_a = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        fp_a = SpectralFingerprint.from_adjacency("a", adj_a)
        parents = SpectralAlignmentScorer.select_diverse_parents([fp_a], n_parents=2)
        assert parents == ["a"]

    def test_select_diverse_parents_high_threshold(self):
        adj_a = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        adj_b = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        fp_a = SpectralFingerprint.from_adjacency("a", adj_a)
        fp_b = SpectralFingerprint.from_adjacency("b", adj_b)
        # With a very high threshold, should still pick the best pair
        parents = SpectralAlignmentScorer.select_diverse_parents(
            [fp_a, fp_b], n_parents=2, min_diversity=0.99
        )
        assert len(parents) == 2


# ---------------------------------------------------------------------------
# ConservationRatioMonitor


class TestConservationRatioMonitor:
    def test_record_and_history(self):
        mon = ConservationRatioMonitor()
        mon.record("agent_1", 0.8)
        mon.record("agent_1", 0.7)
        assert mon.get_history("agent_1") == [0.8, 0.7]

    def test_window_size(self):
        mon = ConservationRatioMonitor(window_size=3)
        for i in range(5):
            mon.record("agent_1", 0.5 + i * 0.1)
        hist = mon.get_history("agent_1")
        assert len(hist) == 3
        assert hist[0] == 0.7  # oldest in window

    def test_no_anomaly_when_stable(self):
        mon = ConservationRatioMonitor(threshold=0.5)
        for _ in range(5):
            mon.record("agent_1", 0.8)
        is_anom, severity = mon.is_anomaly("agent_1")
        assert not is_anom
        assert severity == 0.0

    def test_detects_drop(self):
        mon = ConservationRatioMonitor(threshold=0.3)
        for _ in range(5):
            mon.record("agent_1", 0.8)
        mon.record("agent_1", 0.4)  # 50% drop
        is_anom, severity = mon.is_anomaly("agent_1")
        assert is_anom
        assert severity > 0.3

    def test_insufficient_history(self):
        mon = ConservationRatioMonitor()
        mon.record("agent_1", 0.8)
        is_anom, severity = mon.is_anomaly("agent_1")
        assert not is_anom
        assert severity == 0.0

    def test_record_from_fingerprint(self):
        adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        fp = SpectralFingerprint.from_adjacency("test", adj)
        mon = ConservationRatioMonitor()
        mon.record_from_fingerprint(fp)
        assert len(mon.get_history("test")) == 1


# ---------------------------------------------------------------------------
# SpectralBreederDiversity


class TestSpectralBreederDiversity:
    def test_register_agent(self):
        sbd = SpectralBreederDiversity()
        fp = sbd.register_agent(
            agent_id="agent_1",
            capabilities=["vision", "audio"],
        )
        assert fp.agent_id == "agent_1"
        assert "agent_1" in sbd.fingerprints

    def test_score_diversity(self):
        sbd = SpectralBreederDiversity()
        sbd.register_agent(
            agent_id="a",
            capabilities=["vision", "audio"],
            capability_links=[("vision", "audio", 0.9)],
        )
        sbd.register_agent(
            agent_id="b",
            capabilities=["vision", "audio"],
            capability_links=[("vision", "audio", 0.1)],
        )
        score = sbd.score_diversity("a", "b")
        assert score >= 0.0

    def test_select_parents(self):
        sbd = SpectralBreederDiversity()
        sbd.register_agent("a", ["vision", "audio"])
        sbd.register_agent("b", ["reasoning", "planning"])
        sbd.register_agent("c", ["vision", "reasoning"])
        parents = sbd.select_parents(n=2, min_diversity=0.1)
        assert len(parents) == 2

    def test_detect_anomalies(self):
        sbd = SpectralBreederDiversity()
        # Register with high conservation ratio
        for i in range(5):
            sbd.register_agent(f"agent_{i}", ["vision", "audio", "reasoning"])
        # All registered agents have the same capabilities → same CR
        # No anomalies expected initially
        anomalies = sbd.detect_anomalies()
        # Actually, they all have the same CR so no drop
        assert len(anomalies) == 0

    def test_archive_roundtrip(self):
        sbd = SpectralBreederDiversity()
        sbd.register_agent("a", ["vision", "audio"])
        sbd.register_agent("b", ["reasoning", "planning"])
        archive = sbd.to_archive()
        assert len(archive) == 2
        sbd2 = SpectralBreederDiversity.from_archive(archive)
        assert len(sbd2.fingerprints) == 2
        assert "a" in sbd2.fingerprints
        assert "b" in sbd2.fingerprints


# ---------------------------------------------------------------------------
# ConservationSpectralEngine


class TestConservationSpectralEngine:
    def test_analyze_returns_metrics(self):
        engine = ConservationSpectralEngine()
        adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        result = engine.analyze(adj)
        assert "conservation_ratio" in result
        assert "spectral_gap" in result
        assert "fiedler_vector" in result
        assert result["conservation_ratio"] >= 0.0
        assert len(result["fiedler_vector"]) == 3

    def test_analyze_fallback_works(self):
        # The engine always works, even without the package
        engine = ConservationSpectralEngine()
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        result = engine.analyze(adj)
        assert result["conservation_ratio"] >= 0.0
        assert result["spectral_gap"] >= 0.0

    def test_is_available(self):
        engine = ConservationSpectralEngine()
        # May or may not be available depending on env
        assert isinstance(engine.is_available(), bool)


# ---------------------------------------------------------------------------
# Edge cases


class TestEdgeCases:
    def test_single_node_graph(self):
        adj = np.array([[0]], dtype=float)
        fp = SpectralFingerprint.from_adjacency("single", adj)
        assert len(fp.eigenvalues) == 1
        assert fp.eigenvalues[0] == 0.0
        assert fp.conservation_ratio == 0.0

    def test_empty_graph(self):
        adj = np.zeros((3, 3), dtype=float)
        fp = SpectralFingerprint.from_adjacency("empty", adj)
        assert len(fp.eigenvalues) == 3
        assert fp.conservation_ratio == 0.0

    def test_large_graph(self):
        n = 50
        adj = np.random.rand(n, n)
        adj = (adj + adj.T) / 2  # symmetric
        np.fill_diagonal(adj, 0)
        fp = SpectralFingerprint.from_adjacency("large", adj)
        assert len(fp.eigenvalues) == n
        assert fp.conservation_ratio >= 0.0

    def test_diversity_score_with_missing_agent(self):
        sbd = SpectralBreederDiversity()
        score = sbd.score_diversity("missing_a", "missing_b")
        assert score == 0.0
