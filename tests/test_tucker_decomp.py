#!/usr/bin/env python3
"""tests/test_tucker_decomp.py — Tucker decomposition correctness suite.

Five gates:
1. factorize + reconstruct ≈ original
2. compression ratio > 2×
3. forward pass produces correct output shape
4. parameter count is strictly reduced vs dense
5. gradient flows through the factored form

Design eye: Gilbert Strang — test the math, not the plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.tucker_decomp import TuckerLayer

# ── Fixtures ─────────────────────────────────────────────────────────────

RNG = np.random.default_rng(2026)


def _make_smooth_tensor(dims: tuple[int, int, int]) -> np.ndarray:
    """Create a low-rank-ish synthetic weight tensor."""
    d1, d2, d3 = dims
    # Sum of outer products of smooth 1-D signals → inherently low-rank in each mode
    W = np.zeros(dims, dtype=np.float32)
    for k in range(4):
        a = np.sin(np.linspace(0, (k + 1) * np.pi, d1)).astype(np.float32)
        b = np.cos(np.linspace(0, (k + 2) * np.pi, d2)).astype(np.float32)
        c = np.sin(np.linspace(0, (k + 3) * np.pi, d3)).astype(np.float32)
        W += np.outer(np.outer(a, b), c).reshape(d1, d2, d3)
    # Small noise so it isn't *too* trivial
    W += 0.01 * RNG.standard_normal(dims).astype(np.float32)
    return W


@pytest.fixture
def dims():
    return (32, 32, 32)


@pytest.fixture
def ranks():
    return (8, 8, 8)


@pytest.fixture
def dense_weight(dims):
    return _make_smooth_tensor(dims)


@pytest.fixture
def layer(dims, ranks, dense_weight):
    return TuckerLayer(dims, ranks, weight=dense_weight)


# ── Test 1: Factorize + Reconstruct ≈ Original ──────────────────────────


class TestFactorizeReconstruct:
    def test_reconstruction_error_small(self, layer, dense_weight):
        """HOSVD reconstruction should be close to the original tensor."""
        W_hat = layer.reconstruct()
        rel_err = float(
            np.linalg.norm(dense_weight - W_hat) / np.linalg.norm(dense_weight)
        )
        # For a smooth, low-rank-ish tensor and ranks=16 on dims=64,
        # we expect < 5% relative error.
        assert rel_err < 0.05, f"Relative reconstruction error {rel_err:.4f} too high"

    def test_reconstruction_shape_matches(self, layer, dims):
        """Reconstructed tensor must match original shape."""
        W_hat = layer.reconstruct()
        assert W_hat.shape == dims


# ── Test 2: Compression Ratio > 2× ───────────────────────────────────────


class TestCompressionRatio:
    def test_ratio_exceeds_2x(self, layer):
        """FM targeted 4×; we gate at 2× as a hard floor."""
        ratio = layer.compression_ratio()
        assert ratio > 2.0, f"Compression ratio {ratio:.2f}× ≤ 2×"

    def test_ratio_matches_param_counts(self, layer):
        """Ratio from param_counts() must agree with compression_ratio()."""
        counts = layer.param_counts()
        assert counts["ratio"] == pytest.approx(layer.compression_ratio())
        assert counts["ratio"] == pytest.approx(counts["dense"] / counts["tucker"])

    def test_64_16_compression(self):
        """Specific expectation: (64,64,64) → (16,16,16) yields ~36.5×."""
        W = _make_smooth_tensor((64, 64, 64))
        layer = TuckerLayer((64, 64, 64), (16, 16, 16), weight=W)
        assert layer.compression_ratio() == pytest.approx(
            64**3 / (16**3 + 3 * 64 * 16), rel=1e-3
        )


# ── Test 3: Forward Pass — Correct Output Shape ──────────────────────────


class TestForwardPass:
    def test_single_sample_shape(self, layer, dims):
        """Forward with single input sample produces (d1,)."""
        x = RNG.standard_normal((dims[1], dims[2])).astype(np.float32)
        y = layer(x)
        assert y.shape == (dims[0],)

    def test_batch_shape(self, layer, dims):
        """Forward with batch produces (batch, d1)."""
        batch = 8
        x = RNG.standard_normal((batch, dims[1], dims[2])).astype(np.float32)
        y = layer(x)
        assert y.shape == (batch, dims[0])

    def test_multidim_batch_shape(self, layer, dims):
        """Arbitrary leading dimensions are preserved."""
        x = RNG.standard_normal((2, 3, dims[1], dims[2])).astype(np.float32)
        y = layer(x)
        assert y.shape == (2, 3, dims[0])

    def test_forward_matches_dense_reference(self, layer, dense_weight, dims):
        """Factored forward must equal dense contraction with *reconstructed* weights."""
        x = RNG.standard_normal((4, dims[1], dims[2])).astype(np.float32)
        y_tucker = layer(x)
        # Dense reference using the reconstructed (not original) weight
        W_hat = layer.reconstruct()
        y_dense = np.einsum("ijk,bjk->bi", W_hat, x)
        np.testing.assert_allclose(y_tucker, y_dense, rtol=1e-3, atol=1e-4)


# ── Test 4: Parameter Count Strictly Reduced ──────────────────────────────


class TestParamCount:
    def test_tucker_less_than_dense(self, layer):
        """Tucker parameter count must be strictly smaller than dense."""
        counts = layer.param_counts()
        assert counts["tucker"] < counts["dense"]

    def test_core_and_factors_accounted(self, layer):
        """The reported count must equal the sum of individual components."""
        counts = layer.param_counts()
        manual = layer.core.size + layer.A.size + layer.B.size + layer.C.size
        assert counts["tucker"] == manual

    def test_asymmetric_ranks(self):
        """Asymmetric ranks still reduce parameters."""
        W = _make_smooth_tensor((32, 48, 64))
        layer = TuckerLayer((32, 48, 64), (8, 12, 16), weight=W)
        counts = layer.param_counts()
        dense = 32 * 48 * 64
        assert counts["dense"] == dense
        assert counts["tucker"] < dense


# ── Test 5: Gradient Flow ─────────────────────────────────────────────────


class TestGradientFlow:
    def test_numpy_numerical_gradient(self, layer, dims):
        """NumPy finite-difference check: perturbations in core propagate to output."""
        x = RNG.standard_normal((2, dims[1], dims[2])).astype(np.float32)
        eps = 1e-4

        # Pick a scalar entry in the core and finite-difference it
        y0 = layer(x)
        layer.core[0, 0, 0] += eps
        y1 = layer(x)
        layer.core[0, 0, 0] -= eps  # restore

        dy = (y1 - y0) / eps
        # The output must have changed (non-zero gradient signal)
        assert np.linalg.norm(dy) > 0, "Core perturbation did not propagate to output"

    def test_torch_autograd_flow(self, dims, dense_weight):
        """PyTorch autograd must compute non-zero gradients for all factors."""
        pytest.importorskip("torch")
        import torch

        layer = TuckerLayer(
            dims, (16, 16, 16), weight=torch.from_numpy(dense_weight), backend="torch"
        )
        mod = layer.to_torch_module()

        x = torch.randn(2, dims[1], dims[2])
        y = mod(x)
        loss = y.sum()
        loss.backward()

        assert mod.core.grad is not None and mod.core.grad.abs().sum() > 0
        assert mod.A.grad is not None and mod.A.grad.abs().sum() > 0
        assert mod.B.grad is not None and mod.B.grad.abs().sum() > 0
        assert mod.C.grad is not None and mod.C.grad.abs().sum() > 0

    def test_gradient_wrt_input(self, layer, dims):
        """Output must change when input changes (layer is not constant)."""
        x0 = RNG.standard_normal((2, dims[1], dims[2])).astype(np.float32)
        x1 = x0 + 0.1 * RNG.standard_normal(x0.shape).astype(np.float32)
        y0 = layer(x0)
        y1 = layer(x1)
        assert not np.allclose(y0, y1, atol=1e-6)


# ── Torch Module Wrapper ─────────────────────────────────────────────────


class TestTorchModule:
    def test_wrapper_produces_same_output(self, dims, dense_weight):
        """to_torch_module() must match the standalone forward."""
        pytest.importorskip("torch")
        import torch

        layer = TuckerLayer(
            dims, (16, 16, 16), weight=torch.from_numpy(dense_weight), backend="torch"
        )
        x = torch.randn(3, dims[1], dims[2])

        y_standalone = layer(x)
        mod = layer.to_torch_module()
        y_wrapped = mod(x)

        np.testing.assert_allclose(
            y_standalone.detach().cpu().numpy(),
            y_wrapped.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_wrapper_is_nn_module(self, dims):
        """Returned object must be an instance of torch.nn.Module."""
        torch = pytest.importorskip("torch")
        layer = TuckerLayer(dims, (16, 16, 16), backend="torch")
        mod = layer.to_torch_module()
        assert isinstance(mod, torch.nn.Module)


# ── Edge Cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_random_init_no_weight(self, dims):
        """Layer works when initialized without a dense weight to factorize."""
        layer = TuckerLayer(dims, (16, 16, 16))
        x = RNG.standard_normal((2, dims[1], dims[2])).astype(np.float32)
        y = layer(x)
        assert y.shape == (2, dims[0])

    def test_backend_detection(self, dims, dense_weight):
        """Backend auto-detects torch Tensor input."""
        pytest.importorskip("torch")
        import torch

        layer = TuckerLayer(dims, (16, 16, 16), weight=torch.from_numpy(dense_weight))
        assert layer.backend == "torch"

    def test_different_dims_and_ranks(self):
        """Non-cubic dimensions work correctly."""
        dims = (16, 32, 48)
        ranks = (4, 8, 12)
        W = _make_smooth_tensor(dims)
        layer = TuckerLayer(dims, ranks, weight=W)
        x = RNG.standard_normal((5, dims[1], dims[2])).astype(np.float32)
        y = layer(x)
        assert y.shape == (5, dims[0])
        assert layer.compression_ratio() > 1.0
