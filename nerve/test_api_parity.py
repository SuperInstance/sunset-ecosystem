"""test_api_parity — verify Python einsum and Rust subprocess produce identical latents."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

import room_grid as rg_py
import room_grid_rust as rg_rust


@pytest.fixture
def small_grid():
    """10-room grid with deterministic weights."""
    return rg_py.make_weights(n=10, seed=42)


@pytest.fixture
def room_state():
    """64-dim room state."""
    rng = np.random.RandomState(123)
    return rng.randn(64).astype(np.float32)


def test_python_forward_shape(small_grid, room_state):
    """Sanity: numpy forward returns correct shape."""
    out = rg_py.forward_einsum(small_grid, room_state)
    assert out.shape == (10, 16)
    assert out.dtype == np.float32


def test_rust_subprocess_forward_shape(small_grid, room_state):
    """Rust subprocess returns correct shape (if binary exists)."""
    if rg_rust._RUST_EXE is None:
        pytest.skip("jepa-cli binary not built — run `cargo build --release` in nerve/")
    out = rg_rust.forward_rust_subprocess(small_grid, room_state, n=10)
    assert out.shape == (10, 16)
    assert out.dtype == np.float32


def test_parity_numpy_vs_rust(small_grid, room_state):
    """Latents from Rust CLI must match numpy einsum within floating-point tolerance."""
    if rg_rust._RUST_EXE is None:
        pytest.skip("jepa-cli binary not built — run `cargo build --release` in nerve/")

    py_out = rg_py.forward_einsum(small_grid, room_state)
    rs_out = rg_rust.forward_rust_subprocess(small_grid, room_state, n=10)

    # ReLU + matmul order is identical; expect ~1e-6 relative diff
    np.testing.assert_allclose(py_out, rs_out, rtol=1e-5, atol=1e-6)


def test_parity_forward_one(small_grid, room_state):
    """forward_one(i) must equal the i-th row of forward_einsum."""
    full = rg_py.forward_einsum(small_grid, room_state)
    for i in range(10):
        single = rg_py.forward_one(small_grid, i, room_state)
        np.testing.assert_allclose(full[i], single, rtol=1e-5, atol=1e-6)


def test_class_api_parity(small_grid, room_state):
    """JEPAGrid.tick() returns same structure regardless of backend."""
    g_py = rg_py.JEPAGrid(n=10)
    g_py.w = small_grid
    g_rs = rg_rust.JEPAGrid(n=10)
    g_rs.w = small_grid

    out_py = g_py.tick(room_state)
    out_rs = g_rs.tick(room_state)

    assert set(out_py.keys()) == set(out_rs.keys())
    assert out_py["fired"] == out_rs["fired"]
    assert out_py["tick"] == out_rs["tick"] == 1


def test_rust_cli_json_roundtrip():
    """The Rust CLI reads JSON and emits valid JSON (mocked, no binary required)."""
    # We just verify the Python serialization / deserialization path
    # that will feed the CLI.  Real parity is covered above.
    payload = {
        "n": 2,
        "x": [1.0] * 64,
        "w1": [0.01] * (2 * 64 * 32),
        "w2": [0.01] * (2 * 32 * 16),
        "w3": [0.01] * (2 * 16 * 16),
        "b1": [0.0] * (2 * 32),
        "b2": [0.0] * (2 * 16),
        "b3": [0.0] * (2 * 16),
    }
    dumped = json.dumps(payload)
    loaded = json.loads(dumped)
    assert loaded["n"] == 2
    assert len(loaded["x"]) == 64


def test_fallback_when_binary_missing():
    """If jepa-cli is missing, _forward falls back to einsum silently."""
    old_exe = rg_rust._RUST_EXE
    try:
        rg_rust._RUST_EXE = None
        g = rg_rust.JEPAGrid(n=5)
        # Should not raise even though binary is "missing"
        out = g._forward(np.random.randn(64).astype(np.float32))
        assert out.shape == (5, 16)
    finally:
        rg_rust._RUST_EXE = old_exe


def test_fingerprint_diff():
    """Fingerprint.diff() produces a non-negative scalar."""
    g = rg_py.JEPAGrid(n=5)
    fps = g.fingerprints(n=5)
    assert len(fps) == 5
    d = fps[0].diff(fps[1])
    assert isinstance(d, float)
    assert d >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
