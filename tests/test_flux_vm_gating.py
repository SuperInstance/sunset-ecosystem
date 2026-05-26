"""Tests for FLUX Path B VM-based gating integration."""

from __future__ import annotations

import os

_dev_so = "/root/.openclaw/workspace/flux-vm-v3-temp/target/release/libflux_vm_v3.so"
if os.path.exists(_dev_so) and not os.environ.get("FLUX_VM_SO"):
    os.environ["FLUX_VM_SO"] = _dev_so

import numpy as np
import pytest

from swarm.flux_gating import FluxGatingConfig, FluxCheckResult
from swarm.flux_vm_gating import FluxVMGatingChecker, FluxVMConfig


class TestFluxVMGatingChecker:
    """End-to-end: VM gating checker produces proof-carrying results."""

    def test_checker_initializes(self):
        cfg = FluxGatingConfig(weight_bounds=(-5.0, 5.0), max_l2_norm=50.0)
        vm_cfg = FluxVMConfig(scale=1000)
        checker = FluxVMGatingChecker(flux_config=cfg, vm_config=vm_cfg)
        assert checker._bridge is not None

    def test_passing_candidate(self):
        cfg = FluxGatingConfig(
            weight_bounds=(-5.0, 5.0),
            max_l2_norm=100.0,
            max_variance=10.0,
            max_chaos=1.0,
        )
        checker = FluxVMGatingChecker(flux_config=cfg)
        weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = checker.check_candidate(weights, chaos=0.3, thermal_pressure=0.0)
        assert isinstance(result, FluxCheckResult)
        assert result.passed is True
        assert result.score < 0.5

    def test_failing_bounds(self):
        cfg = FluxGatingConfig(weight_bounds=(-1.0, 1.0), max_l2_norm=100.0)
        checker = FluxVMGatingChecker(flux_config=cfg)
        weights = np.array([5.0, 0.0, 0.0], dtype=np.float32)
        result = checker.check_candidate(weights)
        assert result.passed is False
        assert "bounds" in result.violations

    def test_proof_certificate_present(self):
        cfg = FluxGatingConfig(weight_bounds=(-10.0, 10.0), max_l2_norm=100.0)
        vm_cfg = FluxVMConfig(collect_proof=True)
        checker = FluxVMGatingChecker(flux_config=cfg, vm_config=vm_cfg)
        weights = np.array([1.0, 2.0], dtype=np.float32)
        result = checker.check_candidate(weights)
        assert result.proof_hash is not None
        assert len(result.proof_hash) == 32

    def test_vm_cycles_recorded(self):
        cfg = FluxGatingConfig(weight_bounds=(-10.0, 10.0), max_l2_norm=100.0)
        checker = FluxVMGatingChecker(flux_config=cfg)
        weights = np.array([1.0, 2.0], dtype=np.float32)
        result = checker.check_candidate(weights)
        assert result.vm_cycles > 0

    def test_batch_check(self):
        cfg = FluxGatingConfig(weight_bounds=(-5.0, 5.0), max_l2_norm=100.0)
        checker = FluxVMGatingChecker(flux_config=cfg)
        candidates = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([10.0, 0.0], dtype=np.float32),  # fails bounds
        ]
        results = checker.check_batch(candidates)
        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False

    def test_thermal_pressure_gates(self):
        cfg = FluxGatingConfig(
            weight_bounds=(-1.0, 1.0),  # tight bounds so weights fail too
            max_l2_norm=100.0,
            thermal_budget_gate=0.1,
        )
        checker = FluxVMGatingChecker(flux_config=cfg)
        weights = np.array([5.0, 0.0], dtype=np.float32)  # bounds violation
        result = checker.check_candidate(weights, thermal_pressure=5.0)
        assert result.passed is False
        assert "thermal" in result.violations

    def test_chaos_gates(self):
        cfg = FluxGatingConfig(
            weight_bounds=(-1.0, 1.0),  # tight bounds so weights fail too
            max_l2_norm=100.0,
            max_chaos=0.1,
        )
        checker = FluxVMGatingChecker(flux_config=cfg)
        weights = np.array([5.0, 0.0], dtype=np.float32)  # bounds violation
        result = checker.check_candidate(weights, chaos=2.0)
        assert result.passed is False
        assert "chaos" in result.violations
