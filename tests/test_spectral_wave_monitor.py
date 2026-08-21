"""Tests for spectral_wave_monitor.py — Spectral Graph Wave Coherence Monitoring.

Pattern 2 from the SuperInstance audit: model fleet as graph, monitor
spectral coherence via Laplacian eigenvalues, and control topology health.
"""

import numpy as np
import pytest

from fleet.spectral_wave_monitor import (
    WaveState,
    SpectralThermostat,
    SpectralThermostatConfig,
    ThermostatAction,
    conservation_ratio,
    coherence_halflife,
    fleet_coherence_forecast,
    detect_topology_change,
)


# ===================================================================
# Graph Construction
# ===================================================================


class TestGraphConstruction:
    def test_empty_graph(self) -> None:
        ws = WaveState()
        assert ws.n_nodes == 0
        assert ws.adjacency.shape == (0, 0)

    def test_from_edges_undirected(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 2.0)])
        assert ws.n_nodes == 3
        assert ws.adjacency[0, 1] == 1.0
        assert ws.adjacency[1, 0] == 1.0
        assert ws.adjacency[1, 2] == 2.0
        assert ws.adjacency[2, 1] == 2.0
        assert ws.adjacency[0, 2] == 0.0

    def test_from_fleet_topology(self) -> None:
        ws = WaveState.from_fleet_topology(
            agents=["alpha", "beta", "gamma"],
            links=[("alpha", "beta", 1.0), ("beta", "gamma", 2.0)],
        )
        assert ws.node_labels == ["alpha", "beta", "gamma"]
        assert ws.adjacency[0, 1] == 1.0
        assert ws.adjacency[1, 2] == 2.0

    def test_labels_auto_generated(self) -> None:
        ws = WaveState.from_edges(2, [(0, 1, 1.0)])
        assert ws.node_labels == ["node_0", "node_1"]


# ===================================================================
# Spectral Core
# ===================================================================


class TestSpectralCore:
    def test_laplacian_of_triangle(self) -> None:
        # Complete graph K3 (triangle)
        ws = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)])
        L = ws.laplacian
        expected = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
        np.testing.assert_allclose(L, expected)

    def test_degree_matrix(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 2.0)])
        D = ws.degree_matrix
        expected = np.diag([1.0, 3.0, 2.0])
        np.testing.assert_allclose(D, expected)

    def test_eigenvalues_zero_first(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)])
        vals, vecs = ws.compute_spectrum()
        assert len(vals) == 3
        assert vals[0] == pytest.approx(0.0, abs=1e-10)
        # All eigenvalues real and sorted ascending
        assert np.all(np.diff(vals) >= -1e-12)

    def test_fiedler_path_graph(self) -> None:
        # Path graph P4: λ₂ ≈ 0.586
        ws = WaveState.from_edges(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
        lam2 = ws.fiedler_eigenvalue()
        assert lam2 == pytest.approx(0.586, abs=0.01)

    def test_fiedler_complete_graph(self) -> None:
        # Complete graph K4: λ₂ = 4
        ws = WaveState.from_edges(
            4,
            [
                (0, 1, 1.0),
                (0, 2, 1.0),
                (0, 3, 1.0),
                (1, 2, 1.0),
                (1, 3, 1.0),
                (2, 3, 1.0),
            ],
        )
        lam2 = ws.fiedler_eigenvalue()
        assert lam2 == pytest.approx(4.0, abs=0.01)

    def test_fiedler_disconnected(self) -> None:
        # Two disconnected edges: λ₂ = 0
        ws = WaveState.from_edges(4, [(0, 1, 1.0), (2, 3, 1.0)])
        assert ws.fiedler_eigenvalue() == pytest.approx(0.0, abs=1e-10)

    def test_wave_speed(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)])
        speed = ws.wave_speed()
        assert speed == pytest.approx(np.sqrt(ws.fiedler_eigenvalue()), abs=1e-10)
        assert speed > 0

    def test_wave_speed_disconnected(self) -> None:
        ws = WaveState.from_edges(4, [(0, 1, 1.0), (2, 3, 1.0)])
        assert ws.wave_speed() == pytest.approx(0.0, abs=1e-10)

    def test_spectrum_caching(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0)])
        v1, _ = ws.compute_spectrum()
        ws._eigenvalues = None  # manual invalidate
        v2, _ = ws.compute_spectrum()
        np.testing.assert_array_equal(v1, v2)


# ===================================================================
# Frequency Sweep & Standing Waves
# ===================================================================


class TestFrequencySweep:
    def test_frequency_sweep_shape(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)])
        freqs = np.linspace(0.0, 3.0, 10)
        resp = ws.frequency_sweep(freqs)
        assert resp.shape == (10, 3)

    def test_frequency_sweep_no_nan(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)])
        freqs = np.linspace(0.0, 3.0, 50)
        resp = ws.frequency_sweep(freqs)
        assert np.isfinite(resp).all()

    def test_standing_wave_peaks_finds_eigenfrequencies(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)])
        peaks = ws.standing_wave_peaks(freq_resolution=0.01, max_freq=3.0)
        # Should find at least one peak near the eigenfrequency
        assert len(peaks) >= 1
        # First peak should be near the non-zero eigenfrequency
        vals, _ = ws.compute_spectrum()
        omega2 = np.sqrt(vals[1])
        freqs = [p[0] for p in peaks]
        assert any(abs(f - omega2) < 0.1 for f in freqs)

    def test_standing_wave_peaks_empty_for_high_threshold(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0)])
        peaks = ws.standing_wave_peaks(threshold_ratio=1.0)
        # Only the absolute maximum qualifies, so at most 1 peak
        assert len(peaks) <= 1

    def test_standing_wave_peaks_empty_graph(self) -> None:
        ws = WaveState.from_edges(1, [])
        peaks = ws.standing_wave_peaks()
        assert peaks == []


# ===================================================================
# Conservation Ratio & Coherence
# ===================================================================


class TestConservationRatio:
    def test_cr_complete_graph(self) -> None:
        # K4: λ₂ = 4, avg_deg = 3, CR = 4/3 ≈ 1.33 → clamped to 1.0
        ws = WaveState.from_edges(
            4,
            [
                (0, 1, 1.0),
                (0, 2, 1.0),
                (0, 3, 1.0),
                (1, 2, 1.0),
                (1, 3, 1.0),
                (2, 3, 1.0),
            ],
        )
        cr = conservation_ratio(ws)
        assert cr == pytest.approx(1.0, abs=1e-10)

    def test_cr_path_graph(self) -> None:
        # P4: λ₂ ≈ 0.586, avg_deg = 1.5, CR ≈ 0.39
        ws = WaveState.from_edges(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
        cr = conservation_ratio(ws)
        assert 0.0 < cr < 1.0
        assert cr == pytest.approx(0.3905, abs=0.01)

    def test_cr_star_graph(self) -> None:
        # S4: λ₂ = 1.0, avg_deg = 1.5, CR = 0.667
        ws = WaveState.from_edges(4, [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)])
        cr = conservation_ratio(ws)
        assert cr == pytest.approx(0.6667, abs=0.01)

    def test_cr_disconnected(self) -> None:
        ws = WaveState.from_edges(4, [(0, 1, 1.0), (2, 3, 1.0)])
        assert conservation_ratio(ws) == pytest.approx(0.0, abs=1e-10)

    def test_cr_empty_graph(self) -> None:
        ws = WaveState()
        assert conservation_ratio(ws) == pytest.approx(0.0, abs=1e-10)

    def test_coherence_halflife_at_target(self) -> None:
        # CR = 0.5 → halflife ≈ 1.0 time_unit
        hl = coherence_halflife(0.5)
        assert hl == pytest.approx(1.0, abs=1e-4)

    def test_coherence_halflife_perfect(self) -> None:
        # CR → 1.0 → halflife diverges (large)
        hl = coherence_halflife(0.9999)
        assert hl > 9000

    def test_coherence_halflife_zero(self) -> None:
        # CR → 0.0 → halflife → 0
        hl = coherence_halflife(0.0)
        assert hl == pytest.approx(0.0, abs=1e-6)

    def test_fleet_coherence_forecast(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)])
        forecast = fleet_coherence_forecast(ws, horizon_steps=5)
        assert len(forecast) == 5
        # Should decay monotonically
        assert all(forecast[i] >= forecast[i + 1] for i in range(4))
        assert forecast[0] > forecast[-1]

    def test_fleet_coherence_forecast_disconnected(self) -> None:
        ws = WaveState.from_edges(4, [(0, 1, 1.0), (2, 3, 1.0)])
        forecast = fleet_coherence_forecast(ws)
        assert all(v == 0.0 for v in forecast)


# ===================================================================
# Topology Change Detection
# ===================================================================


class TestTopologyChangeDetection:
    def test_no_change_same_graph(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)])
        ws.record_snapshot()
        assert ws.topology_change_detected() is False

    def test_change_when_node_added(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)])
        ws.record_snapshot()
        ws.add_node()
        ws.add_edge(2, 3, 1.0)
        assert ws.topology_change_detected() is True

    def test_change_when_edge_removed(self) -> None:
        ws = WaveState.from_edges(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
        ws.record_snapshot()
        ws.remove_edge(1, 2)
        assert ws.topology_change_detected() is True

    def test_detect_topology_change_standalone(self) -> None:
        before = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)])
        after = WaveState.from_edges(3, [(0, 1, 1.0)])
        assert detect_topology_change(before, after) is True

    def test_detect_topology_change_no_change(self) -> None:
        before = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)])
        after = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)])
        assert detect_topology_change(before, after) is False

    def test_history_window_size(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0)])
        ws.record_snapshot()  # snapshot 1
        ws.record_snapshot()  # snapshot 2
        ws.record_snapshot()  # snapshot 3
        # No change, but compare against last 3
        assert ws.topology_change_detected(window_size=3) is False


# ===================================================================
# Spectral Thermostat
# ===================================================================


class TestSpectralThermostat:
    def test_thermostat_noop_in_deadband(self) -> None:
        # Graph with CR ≈ 0.5, inside deadband [0.45, 0.55]
        ws = WaveState.from_edges(
            4, [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (2, 3, 1.0)]
        )
        tstat = SpectralThermostat(wave_state=ws)
        action = tstat.tick()
        assert action == ThermostatAction.NoOp

    def test_thermostat_increase_cr_when_low(self) -> None:
        # P4 graph: CR ≈ 0.39, below deadband [0.45, 0.55] → IncreaseCR
        ws = WaveState.from_edges(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
        tstat = SpectralThermostat(wave_state=ws)
        action = tstat.tick()
        assert action == ThermostatAction.IncreaseCR

    def test_thermostat_decrease_cr_when_high(self) -> None:
        # Complete graph K5: CR = 1.0, above deadband → DecreaseCR
        edges = [(i, j, 1.0) for i in range(5) for j in range(i + 1, 5)]
        ws = WaveState.from_edges(5, edges)
        tstat = SpectralThermostat(wave_state=ws)
        action = tstat.tick()
        assert action == ThermostatAction.DecreaseCR

    def test_thermostat_topology_change_triggers_increase(self) -> None:
        # Graph with CR ≈ 0.5, then remove edge → topology change
        ws = WaveState.from_edges(
            4, [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (2, 3, 1.0)]
        )
        tstat = SpectralThermostat(wave_state=ws)
        tstat.tick()  # establish baseline
        ws.remove_edge(0, 1)
        action = tstat.tick()
        assert action == ThermostatAction.IncreaseCR

    def test_thermostat_predicted_halflife(self) -> None:
        ws = WaveState.from_edges(
            4, [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (2, 3, 1.0)]
        )
        tstat = SpectralThermostat(wave_state=ws)
        tstat.tick()
        hl = tstat.predicted_halflife()
        assert hl > 0

    def test_thermostat_config_custom(self) -> None:
        # target=0.7, deadband=0.1 → range [0.6, 0.8]
        # Use P4 with CR ≈ 0.39 → below 0.6 → IncreaseCR
        cfg = SpectralThermostatConfig(target_cr=0.7, deadband=0.1)
        ws = WaveState.from_edges(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
        tstat = SpectralThermostat(config=cfg, wave_state=ws)
        action = tstat.tick()
        assert action == ThermostatAction.IncreaseCR

    def test_thermostat_update_graph(self) -> None:
        ws1 = WaveState.from_edges(3, [(0, 1, 1.0)])
        ws2 = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)])
        tstat = SpectralThermostat(wave_state=ws1)
        tstat.update_graph(ws2)
        assert tstat.wave_state is ws2


# ===================================================================
# Mutations
# ===================================================================


class TestMutations:
    def test_add_node(self) -> None:
        ws = WaveState.from_edges(2, [(0, 1, 1.0)])
        idx = ws.add_node("newbie")
        assert idx == 2
        assert ws.n_nodes == 3
        assert ws.node_labels[-1] == "newbie"
        assert ws.adjacency[2, 2] == 0.0

    def test_remove_node(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)])
        ws.remove_node(1)
        assert ws.n_nodes == 2
        assert ws.adjacency[0, 1] == 0.0  # edge (0,1) and (1,2) removed

    def test_remove_node_out_of_range(self) -> None:
        ws = WaveState.from_edges(2, [(0, 1, 1.0)])
        with pytest.raises(IndexError):
            ws.remove_node(5)

    def test_add_edge(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0)])
        ws.add_edge(1, 2, 3.0)
        assert ws.adjacency[1, 2] == 3.0
        assert ws.adjacency[2, 1] == 3.0

    def test_add_edge_out_of_range(self) -> None:
        ws = WaveState.from_edges(2, [(0, 1, 1.0)])
        with pytest.raises(IndexError):
            ws.add_edge(0, 5, 1.0)

    def test_remove_edge(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)])
        ws.remove_edge(0, 1)
        assert ws.adjacency[0, 1] == 0.0
        assert ws.adjacency[1, 0] == 0.0

    def test_cache_invalidation_on_mutation(self) -> None:
        ws = WaveState.from_edges(3, [(0, 1, 1.0)])
        ws.compute_spectrum()  # cache
        ws.add_edge(1, 2, 1.0)
        assert ws._eigenvalues is None
        assert ws._eigenvectors is None


# ===================================================================
# Edge Cases & Regression
# ===================================================================


class TestEdgeCases:
    def test_single_node(self) -> None:
        ws = WaveState.from_edges(1, [])
        assert ws.n_nodes == 1
        assert ws.fiedler_eigenvalue() == 0.0
        assert ws.wave_speed() == 0.0

    def test_two_node_edge(self) -> None:
        ws = WaveState.from_edges(2, [(0, 1, 1.0)])
        assert ws.fiedler_eigenvalue() == pytest.approx(2.0, abs=1e-10)
        assert ws.wave_speed() == pytest.approx(np.sqrt(2.0), abs=1e-10)

    def test_weighted_edges(self) -> None:
        ws = WaveState.from_edges(2, [(0, 1, 5.0)])
        assert ws.fiedler_eigenvalue() == pytest.approx(10.0, abs=1e-10)

    def test_negative_weight_ignored(self) -> None:
        # Negative weights are allowed by the API but create non-physical
        # Laplacians; verify the code handles them gracefully.
        ws = WaveState.from_edges(2, [(0, 1, -1.0)])
        L = ws.laplacian
        assert np.isfinite(L).all()
        vals, _ = ws.compute_spectrum()
        assert np.isfinite(vals).all()

    def test_fleet_topology_missing_agent(self) -> None:
        ws = WaveState.from_fleet_topology(
            agents=["alpha", "beta"],
            links=[("alpha", "beta", 1.0), ("beta", "gamma", 1.0)],  # gamma missing
        )
        assert ws.n_nodes == 2
        assert ws.adjacency[0, 1] == 1.0
