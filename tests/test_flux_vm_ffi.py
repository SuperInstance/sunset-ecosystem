"""Tests for flux_vm FFI bindings.

Requires libflux_vm.so built via cargo build --release.
Run: pytest tests/test_flux_vm_ffi.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from flux_vm.ffi import FluxVM


@pytest.fixture
def vm():
    return FluxVM()


class TestFluxVMAvailability:
    def test_vm_detects_so(self):
        vm = FluxVM()
        assert vm.is_available() is True

    def test_vm_repr(self):
        vm = FluxVM()
        assert "FluxVM" in repr(vm)
        assert "loaded=True" in repr(vm)


class TestCheckBatch:
    def test_all_pass(self, vm):
        latents = np.zeros((2, 4), dtype=np.float32)
        vio = vm.check_batch(latents, min_bound=-10.0, max_bound=10.0,
                             max_l2=100.0, max_var=10.0)
        assert vio.shape == (2,)
        assert np.all(vio == 0)

    def test_bounds_violation(self, vm):
        latents = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        vio = vm.check_batch(latents, min_bound=-10.0, max_bound=10.0,
                             max_l2=100.0, max_var=10.0)
        assert vio[0] == 0
        assert vio[1] == 1

    def test_l2_violation(self, vm):
        latents = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0],
        ], dtype=np.float32)
        vio = vm.check_batch(latents, min_bound=-50.0, max_bound=50.0,
                             max_l2=15.0, max_var=100.0)
        assert vio[0] == 0
        assert vio[1] == 1

    def test_dtype_conversion(self, vm):
        latents = np.zeros((2, 4), dtype=np.float64)
        vio = vm.check_batch(latents)
        assert vio.dtype == np.uint8

    def test_wrong_ndim_raises(self, vm):
        latents = np.zeros(4, dtype=np.float32)
        with pytest.raises(ValueError, match="2D"):
            vm.check_batch(latents)


class TestErrorHandling:
    def test_missing_so_raises(self):
        with pytest.raises(FileNotFoundError):
            FluxVM("/nonexistent/libflux_vm.so")
