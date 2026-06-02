"""Tests for jepa_rust.py — PersistentGrid ctypes wrapper.

Loads the module fresh for each test with a mocked ctypes.CDLL,
since the real libjepa_kernel.so lacks the expected symbols.
"""

import ctypes
import importlib.util
import numpy as np
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_jepa_rust(mock_lib: MagicMock):
    """Import nerve.jepa_rust with ctypes.CDLL mocked to return *mock_lib*."""
    spec = importlib.util.spec_from_file_location(
        "nerve.jepa_rust",
        Path(__file__).parent.parent / "nerve" / "jepa_rust.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nerve.jepa_rust"] = mod
    with patch("ctypes.CDLL", return_value=mock_lib):
        spec.loader.exec_module(mod)
    return mod


def _unload_jepa_rust():
    sys.modules.pop("nerve.jepa_rust", None)


class TestPersistentGrid:
    """Verify PersistentGrid FFI wiring and memory management."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Remove our injected module after every test."""
        yield
        _unload_jepa_rust()

    @pytest.fixture
    def mock_lib(self):
        lib = MagicMock()
        lib.jepa_grid_create.return_value = ctypes.c_void_p(0xDEADBEEF)
        lib.jepa_grid_tick.return_value = None
        lib.jepa_grid_tick_batch.return_value = None
        lib.jepa_grid_destroy.return_value = None
        return lib

    @pytest.fixture
    def weights(self):
        return {
            "w1": np.random.randn(3, 64, 32).astype(np.float32),
            "w2": np.random.randn(3, 32, 16).astype(np.float32),
            "w3": np.random.randn(3, 16, 16).astype(np.float32),
            "b1": np.random.randn(3, 32).astype(np.float32),
            "b2": np.random.randn(3, 16).astype(np.float32),
            "b3": np.random.randn(3, 16).astype(np.float32),
        }

    @pytest.fixture
    def jepa_mod(self, mock_lib):
        return _load_jepa_rust(mock_lib)

    @pytest.fixture
    def grid(self, jepa_mod, weights):
        return jepa_mod.PersistentGrid(n=3, weights=weights)

    def test_init_creates_handle(self, grid):
        assert grid.n == 3
        assert grid._handle is not None

    def test_init_raises_when_create_fails(self, jepa_mod, mock_lib, weights):
        mock_lib.jepa_grid_create.return_value = None
        with pytest.raises(RuntimeError, match="jepa_grid_create failed"):
            jepa_mod.PersistentGrid(n=3, weights=weights)

    def test_tick_returns_preallocated_buffer(self, grid, mock_lib):
        signal = np.random.randn(64).astype(np.float32)
        out = grid.tick(signal)
        assert out.shape == (3, 16)
        assert out.dtype == np.float32
        mock_lib.jepa_grid_tick.assert_called_once()

    def test_tick_trims_signal(self, grid, mock_lib):
        """Signal longer than 64 should be trimmed."""
        signal = np.random.randn(128).astype(np.float32)
        grid.tick(signal)
        mock_lib.jepa_grid_tick.assert_called_once()

    def test_tick_batch_shape(self, grid, mock_lib):
        signals = np.random.randn(5, 64).astype(np.float32)
        out = grid.tick_batch(signals)
        assert out.shape == (5, 3, 16)
        assert out.dtype == np.float32
        mock_lib.jepa_grid_tick_batch.assert_called_once()

    def test_tick_batch_reshapes_signals(self, grid, mock_lib):
        """Signals shaped (batch, 64) are handled correctly."""
        signals = np.random.randn(5, 64).astype(np.float32)
        grid.tick_batch(signals)
        mock_lib.jepa_grid_tick_batch.assert_called_once()

    def test_repr(self, grid):
        r = repr(grid)
        assert "PersistentGrid" in r
        assert "n=3" in r

    def test_del_calls_destroy(self, grid, mock_lib):
        handle = grid._handle
        grid.__del__()
        mock_lib.jepa_grid_destroy.assert_called_once()
        assert grid._handle is None
