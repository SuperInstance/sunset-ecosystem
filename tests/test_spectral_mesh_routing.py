"""Tests for SpectralMeshRouting — spectral graph theory mesh optimizer.

Run: python3 -m pytest tests/test_spectral_mesh_routing.py -v --tb=short
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm.spectral_mesh_routing import (
    GraphLaplacian,
    SpectralMeshRouter,
    effective_resistance,
    fiedler_vector,
)


# ── GraphLaplacian ──────────────────────────────────────────


class TestGraphLaplacian:
    def test_from_adjacency_path_graph(self):
        # Path graph: 0 — 1 — 2 — 3
        adj = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
        lap = GraphLaplacian.from_adjacency(adj)
        assert lap.adjacency.shape == (4, 4)
        assert lap.fiedler_value > 0  # connected
        assert len(lap.eigenvalues) == 4
        # Eigenvalues sorted ascending
        assert np.all(np.diff(lap.eigenvalues) >= -1e-10)

    def test_from_adjacency_cycle_graph(self):
        # Cycle C4
        adj = np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=np.float64,
        )
        lap = GraphLaplacian.from_adjacency(adj)
        assert lap.fiedler_value > 0
        # Cycle has better connectivity than path
        path_adj = np.array(
            [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.float64
        )
        path_lap = GraphLaplacian.from_adjacency(path_adj)
        assert lap.fiedler_value > path_lap.fiedler_value

    def test_disconnected_graph(self):
        # Two disconnected edges
        adj = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
        lap = GraphLaplacian.from_adjacency(adj)
        assert lap.fiedler_value == pytest.approx(0.0, abs=1e-10)
        assert lap.fiedler_value < 1e-10  # not connected

    def test_complete_graph(self):
        # K4 — fully connected
        adj = np.ones((4, 4), dtype=np.float64) - np.eye(4)
        lap = GraphLaplacian.from_adjacency(adj)
        assert lap.fiedler_value > 0
        assert lap.fiedler_value > 1e-10  # connected
        assert lap.spectral_gap() > 0.5

    def test_effective_resistance_symmetry(self):
        adj = np.array(
            [
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ],
            dtype=np.float64,
        )
        lap = GraphLaplacian.from_adjacency(adj)
        r01 = lap.effective_resistance(0, 1)
        r10 = lap.effective_resistance(1, 0)
        assert r01 == pytest.approx(r10)
        assert r01 > 0

    def test_cheeger_bound_positive(self):
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
        lap = GraphLaplacian.from_adjacency(adj)
        assert lap.cheeger_bound() > 0

    def test_spectral_clustering(self):
        # Two communities: 0-1-2 connected, 3-4-5 connected, one bridge
        adj = np.zeros((6, 6))
        adj[0, 1] = adj[1, 2] = adj[0, 2] = 1  # clique A
        adj[3, 4] = adj[4, 5] = adj[3, 5] = 1  # clique B
        adj[2, 3] = 1  # bridge
        adj = adj + adj.T
        np.fill_diagonal(adj, 0)
        lap = GraphLaplacian.from_adjacency(adj)
        clusters = lap.spectral_clustering(k=2)
        assert len(clusters) == 2
        # Each cluster should have 3 nodes
        assert all(len(c) == 3 for c in clusters)


# ── SpectralMeshRouter ────────────────────────────────────


class TestSpectralMeshRouter:
    def test_initially_disconnected(self):
        router = SpectralMeshRouter(["a", "b", "c"])
        assert not router.is_connected
        assert router.algebraic_connectivity == pytest.approx(0.0, abs=1e-10)

    def test_add_edge_connects(self):
        router = SpectralMeshRouter(["a", "b", "c"])
        router.add_edge(0, 1)
        router.add_edge(1, 2)
        assert router.is_connected
        assert router.algebraic_connectivity > 0

    def test_suggest_rewiring_improves_connectivity(self):
        router = SpectralMeshRouter(["a", "b", "c", "d"])
        router.add_edge(0, 1)
        router.add_edge(1, 2)
        router.add_edge(2, 3)
        # Path graph — suggest extra edges
        before = router.algebraic_connectivity
        suggestions = router.suggest_rewiring(num_edges=1)
        assert len(suggestions) > 0
        i, j, gain = suggestions[0]
        assert gain > 0
        # Apply and verify
        router.add_edge(i, j)
        after = router.algebraic_connectivity
        assert after > before

    def test_optimal_gossip_order(self):
        router = SpectralMeshRouter(["a", "b", "c", "d"])
        # Path: a-b-c-d
        router.add_edge(0, 1)
        router.add_edge(1, 2)
        router.add_edge(2, 3)
        order = router.optimal_gossip_order()
        assert set(order) == {"a", "b", "c", "d"}
        # Fiedler vector orders nodes by community

    def test_bottleneck_nodes(self):
        # Star graph: leaves have high resistance (only one path)
        router = SpectralMeshRouter(["center", "l1", "l2", "l3"])
        for i in range(1, 4):
            router.add_edge(0, i)
        bottlenecks = router.bottleneck_nodes(threshold_quantile=0.5)
        # Leaves are bottlenecks (high total resistance to others)
        assert len(bottlenecks) >= 1

    def test_cluster_report(self):
        router = SpectralMeshRouter(["a", "b", "c", "d"])
        router.add_edge(0, 1)
        router.add_edge(1, 2)
        router.add_edge(2, 3)
        report = router.cluster_report(k=2)
        assert report["k"] == 2
        assert len(report["clusters"]) == 2
        assert report["fiedler_value"] > 0


# ── standalone helpers ──────────────────────────────────────


class TestStandaloneHelpers:
    def test_effective_resistance_triangle(self):
        # Triangle: effective resistance between any two nodes is 2/3
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64)
        r = effective_resistance(adj, 0, 1)
        assert r == pytest.approx(2.0 / 3.0, rel=0.01)

    def test_fiedler_vector_sum_zero(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64)
        v = fiedler_vector(adj)
        # Fiedler vector is orthogonal to constant vector
        assert abs(v.sum()) < 1e-10
        assert np.linalg.norm(v) > 0
