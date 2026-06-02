"""Tests for superinstance_ffi.py — ctypes bindings.

These tests mock the Rust shared library since it requires
cargo build --release in superinstance-ffi/.
"""

import ctypes
import pytest
from unittest.mock import MagicMock, patch


class TestSuperinstanceFFIBindings:
    """Verify FFI function signatures are wired correctly."""

    @pytest.fixture
    def mock_lib(self):
        """Return a mocked CDLL with callable attributes."""
        lib = MagicMock()
        # Configure return values for each function
        lib.eisenstein_norm.return_value = 3
        lib.laman_check_subset.return_value = 1
        lib.laman_is_rigid.return_value = 1
        lib.holonomy_check.return_value = 1.0
        lib.pythagorean48_encode.return_value = 12
        lib.constraint_check.return_value = 1
        lib.constraint_violation.return_value = 0.0
        lib.spline_interpolate.return_value = 0.5
        lib.deadband_filter.return_value = 0.0
        lib.manhattan_distance.return_value = 5.0
        lib.cascade_match.return_value = 1
        return lib

    @pytest.fixture
    def ffi_module(self, mock_lib):
        """Import superinstance_ffi with a mocked CDLL."""
        with patch("ctypes.CDLL", return_value=mock_lib):
            with patch("os.path.exists", return_value=True):
                import importlib
                import swarm.superinstance_ffi as ffi
                importlib.reload(ffi)
                return ffi

    def test_eisenstein_norm(self, ffi_module):
        assert ffi_module.eisenstein_norm(2, 1) == 3

    def test_laman_check_subset(self, ffi_module):
        assert ffi_module.laman_check_subset(3, 3) is True

    def test_laman_is_rigid(self, ffi_module):
        assert ffi_module.laman_is_rigid(3, 3) is True

    def test_holonomy_check_consistent(self, ffi_module):
        assert ffi_module.holonomy_check([0.0, 0.0, 0.0], 1e-6) == 1.0

    def test_holonomy_check_inconsistent(self, ffi_module):
        """Mock returns 1.0 regardless; test the wiring."""
        assert ffi_module.holonomy_check([0.0, 1.0, 2.0], 0.1) == 1.0

    def test_pythagorean48_encode(self, ffi_module):
        assert ffi_module.pythagorean48_encode(2, 1) == 12

    def test_constraint_check_pass(self, ffi_module):
        assert ffi_module.constraint_check(0.5, 0.0, 1.0) is True

    def test_constraint_check_fail(self, ffi_module):
        """Mock returns 1 regardless; test the wiring."""
        assert ffi_module.constraint_check(1.1, 0.0, 1.0) is True

    def test_constraint_violation(self, ffi_module):
        assert ffi_module.constraint_violation(0.5, 0.0, 1.0) == 0.0

    def test_spline_interpolate(self, ffi_module):
        assert ffi_module.spline_interpolate(0.0, 1.0, 0.0, 0.0, 0.5) == 0.5

    def test_deadband_filter(self, ffi_module):
        val, last = ffi_module.deadband_filter(0.05, 0.0, 0.1)
        # Mock returns 0.0 for filtered value
        assert val == 0.0

    def test_manhattan_distance(self, ffi_module):
        assert ffi_module.manhattan_distance([1.0, 2.0, 3.0], [4.0, 0.0, 3.0]) == 5.0

    def test_manhattan_distance_mismatched(self, ffi_module):
        with pytest.raises(ValueError, match="same length"):
            ffi_module.manhattan_distance([1.0, 2.0], [3.0])

    def test_cascade_match(self, ffi_module):
        idx = ffi_module.cascade_match(
            [1.0, 1.0, 1.0],
            [[0.0, 0.0, 0.0], [1.1, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [0.5, 1.5],
        )
        assert idx == 1

    def test_cascade_match_empty(self, ffi_module):
        idx = ffi_module.cascade_match([1.0], [], [0.5])
        assert idx == -1

    def test_cascade_match_mismatched(self, ffi_module):
        with pytest.raises(ValueError, match="same dimension"):
            ffi_module.cascade_match([1.0, 2.0], [[1.0]], [0.5])

    def test_library_not_found(self):
        """Import should raise RuntimeError when .so is missing."""
        with patch("os.path.exists", return_value=False):
            with pytest.raises(RuntimeError, match="libsuperinstance_ffi.so not found"):
                import importlib
                import swarm.superinstance_ffi as ffi
                importlib.reload(ffi)
