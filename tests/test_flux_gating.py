"""Tests for FLUX Path A gating (swarm/flux_gating.py).

No Rust dependency required — tests exercise both the Python fallback
and the public FluxGatingChecker API.
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm.flux_gating import (
    FluxGatingConfig,
    FluxGatingChecker,
    FluxCheckResult,
    PythonFluxFallback,
)


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def config():
    return FluxGatingConfig(
        weight_bounds=(-2.0, 2.0),
        max_l2_norm=50.0,
        max_variance=5.0,
        max_chaos=0.95,
        thermal_budget_gate=0.90,
        pass_threshold=0.05,  # lowered so single chaos/thermal violations fail
    )


@pytest.fixture
def checker(config):
    return FluxGatingChecker(config=config, vm_path="/nonexistent/so")


@pytest.fixture
def zero_weights():
    """All-zero weights — perfectly compliant."""
    return np.zeros(64, dtype=np.float32)


@pytest.fixture
def extreme_weights():
    """Weights that violate every constraint."""
    return np.full(64, 100.0, dtype=np.float32)


# ── PythonFluxFallback unit tests ───────────────────────────

class TestPythonFluxFallback:
    """Direct tests for the pure-Python backend."""

    def test_perfect_candidate_passes(self, config, zero_weights):
        backend = PythonFluxFallback(config)
        res = backend.check_candidate(zero_weights, chaos=0.1, thermal_pressure=0.0)
        assert res.passed is True
        assert res.score == 0.0
        assert res.violations == {}

    def test_bounds_violation_fails(self, config):
        backend = PythonFluxFallback(config)
        weights = np.full(64, 5.0, dtype=np.float32)  # exceeds max=2.0
        res = backend.check_candidate(weights, chaos=0.1, thermal_pressure=0.0)
        assert res.passed is False
        assert "bounds" in res.violations
        assert res.score > 0.0

    def test_chaos_violation_fails(self, config, zero_weights):
        backend = PythonFluxFallback(config)
        res = backend.check_candidate(zero_weights, chaos=2.0, thermal_pressure=0.0)
        assert res.passed is False
        assert "chaos" in res.violations

    def test_thermal_violation_fails(self, config, zero_weights):
        backend = PythonFluxFallback(config)
        res = backend.check_candidate(zero_weights, chaos=0.1, thermal_pressure=0.99)
        assert res.passed is False
        assert "thermal" in res.violations

    def test_batch_all_pass(self, config):
        backend = PythonFluxFallback(config)
        batch = np.zeros((5, 64), dtype=np.float32)
        chaos = np.zeros(5, dtype=np.float32)
        thermal = np.zeros(5, dtype=np.float32)
        results = backend.check_batch(batch, chaos, thermal)
        assert len(results) == 5
        for r in results:
            assert r.passed is True
            assert r.score == 0.0

    def test_batch_mixed(self, config):
        backend = PythonFluxFallback(config)
        batch = np.zeros((3, 64), dtype=np.float32)
        batch[1] = 100.0  # violates bounds + l2
        chaos = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        thermal = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        results = backend.check_batch(batch, chaos, thermal)
        assert results[0].passed is True
        assert results[1].passed is False
        assert results[2].passed is True

    def test_l2_norm_violation(self, config):
        backend = PythonFluxFallback(config)
        weights = np.ones(64, dtype=np.float32) * 10.0  # l2 = sqrt(64*100) = 80 > 50
        res = backend.check_candidate(weights, chaos=0.1, thermal_pressure=0.0)
        assert res.passed is False
        assert "l2_norm" in res.violations

    def test_variance_violation(self, config):
        backend = PythonFluxFallback(config)
        weights = np.zeros(64, dtype=np.float32)
        weights[0] = 100.0  # high variance
        res = backend.check_candidate(weights, chaos=0.1, thermal_pressure=0.0)
        assert res.passed is False
        assert "variance" in res.violations

    def test_multiple_violations_score_clamped(self, config, extreme_weights):
        backend = PythonFluxFallback(config)
        res = backend.check_candidate(
            extreme_weights, chaos=0.99, thermal_pressure=0.99
        )
        assert res.passed is False
        assert len(res.violations) >= 3  # bounds, l2, variance at minimum
        assert 0.0 < res.score <= 1.0


# ── FluxGatingChecker API tests ─────────────────────────────

class TestFluxGatingChecker:
    """Tests for the public checker API (always uses Python fallback here)."""

    def test_check_candidate_pass(self, checker, zero_weights):
        res = checker.check_candidate(zero_weights, chaos=0.1, thermal_pressure=0.0)
        assert isinstance(res, FluxCheckResult)
        assert res.passed is True
        assert res.score == 0.0

    def test_check_candidate_fail(self, checker):
        weights = np.full(64, 100.0, dtype=np.float32)
        res = checker.check_candidate(weights, chaos=0.1, thermal_pressure=0.0)
        assert res.passed is False
        assert res.score > 0.0

    def test_check_batch(self, checker):
        batch = np.zeros((4, 64), dtype=np.float32)
        batch[2] = 100.0
        results = checker.check_batch(batch)
        assert len(results) == 4
        assert results[0].passed is True
        assert results[2].passed is False

    def test_score_for_breeding_perfect(self, checker, zero_weights):
        score = checker.score_for_breeding(
            zero_weights, zero_weights, chaos_a=0.1, chaos_b=0.1
        )
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_score_for_breeding_bad_parent_penalizes(self, checker, zero_weights):
        bad = np.full(64, 100.0, dtype=np.float32)
        score = checker.score_for_breeding(
            zero_weights, bad, chaos_a=0.1, chaos_b=0.1
        )
        assert score < 0.5  # one parent fails → 0.5 penalty

    def test_score_for_breeding_both_bad(self, checker):
        bad = np.full(64, 100.0, dtype=np.float32)
        score = checker.score_for_breeding(
            bad, bad, chaos_a=0.99, chaos_b=0.99
        )
        assert score == 0.0  # worst case

    def test_get_violating_rooms(self, checker):
        batch = np.zeros((3, 64), dtype=np.float32)
        batch[1] = 100.0
        results = checker.get_violating_rooms(batch)
        assert len(results) == 1
        assert results[0][0] == 1
        assert results[0][1].passed is False

    def test_default_chaos_and_thermal(self, checker, zero_weights):
        # When not provided, chaos defaults to 0.3 and thermal to 0.0
        res = checker.check_candidate(zero_weights)
        assert res.passed is True

    def test_config_attribute(self, config):
        c = FluxGatingChecker(config=config, vm_path="/nonexistent")
        assert c.config is config

    def test_numpy_only_skips_ffi(self, config):
        c = FluxGatingChecker(config=FluxGatingConfig(numpy_only=True))
        assert isinstance(c._backend, PythonFluxFallback)


# ── Edge-case tests ─────────────────────────────────────────

class TestEdgeCases:
    """Boundary and stress tests."""

    def test_empty_weights(self, config):
        backend = PythonFluxFallback(config)
        res = backend.check_candidate(
            np.array([], dtype=np.float32), chaos=0.1, thermal_pressure=0.0
        )
        # Empty array: no bounds violation, l2=0, variance skipped because size<=1
        assert res.passed is True
        assert res.score == 0.0

    def test_single_dim_weights(self, config):
        backend = PythonFluxFallback(config)
        res = backend.check_candidate(
            np.array([5.0], dtype=np.float32), chaos=0.1, thermal_pressure=0.0
        )
        assert res.passed is False  # bounds violation
        assert "bounds" in res.violations

    def test_batch_with_none_chaos_thermal(self, checker):
        batch = np.zeros((2, 64), dtype=np.float32)
        results = checker.check_batch(batch, chaos_vec=None, thermal_vec=None)
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_1d_batch_auto_reshape(self, checker):
        weights = np.zeros(64, dtype=np.float32)
        results = checker.check_batch(weights)
        assert len(results) == 1
        assert results[0].passed is True

    def test_severity_alias(self, checker, zero_weights):
        res = checker.check_candidate(zero_weights)
        assert res.severity == res.score

    def test_repr(self, checker):
        bad = np.full(64, 100.0, dtype=np.float32)
        res = checker.check_candidate(bad)
        s = repr(res)
        assert "FAIL" in s
        assert "score=" in s

    def test_different_severity_weights(self):
        config = FluxGatingConfig(
            weight_bounds=(0.0, 1.0),  # tight bounds
            severity_weights={"bounds": 2.0, "l2_norm": 0.1},
            pass_threshold=0.5,
        )
        checker = FluxGatingChecker(config=config, vm_path="/nonexistent")
        # Only bounds violation — high weight should push score above threshold
        weights = np.full(64, 5.0, dtype=np.float32)
        res = checker.check_candidate(weights, chaos=0.1, thermal_pressure=0.0)
        assert res.passed is False
        assert res.violations["bounds"] > 0.0


# ── BreederDaemonV2 integration tests ───────────────────────

class TestBreederIntegration:
    """Verify the gating module can be attached to BreederDaemonV2."""

    def test_attach_flux_gating(self):
        from swarm.breeder_daemon_v2 import BreederDaemonV2, DiversityConfig, ThermalConfig
        from nerve.room_grid import RoomGrid
        from swarm.thermal import ThermalBudget, DeviceType

        grid = RoomGrid(n=10)
        thermal = ThermalBudget({DeviceType.GPU: 5})
        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            wal_path=":memory:",
            diversity=DiversityConfig(),
            thermal_cfg=ThermalConfig(),
        )
        checker = FluxGatingChecker(config=FluxGatingConfig(), vm_path="/nonexistent")
        daemon.attach_flux_gating(checker=checker)
        assert daemon._flux_checker is checker

    def test_attach_flux_gating_auto_create(self):
        from swarm.breeder_daemon_v2 import BreederDaemonV2
        from nerve.room_grid import RoomGrid
        from swarm.thermal import ThermalBudget, DeviceType

        grid = RoomGrid(n=10)
        thermal = ThermalBudget({DeviceType.GPU: 5})
        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            wal_path=":memory:",
        )
        daemon.attach_flux_gating()
        assert daemon._flux_checker is not None
        assert isinstance(daemon._flux_checker._backend, PythonFluxFallback)

    def test_flux_config_parameter(self):
        from swarm.breeder_daemon_v2 import BreederDaemonV2
        from nerve.room_grid import RoomGrid
        from swarm.thermal import ThermalBudget, DeviceType

        grid = RoomGrid(n=10)
        thermal = ThermalBudget({DeviceType.GPU: 5})
        cfg = FluxGatingConfig(max_chaos=0.5)
        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            wal_path=":memory:",
            flux_config=cfg,
        )
        daemon.attach_flux_gating()
        assert daemon._flux_checker.config.max_chaos == 0.5
