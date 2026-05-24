#!/usr/bin/env python3
"""tucker_decomp.py — Tucker decomposition layer for tensor-compressed neural weights.

Factorizes a dense weight tensor (e.g., 64×64×64) into a small core tensor
(e.g., 16×16×16) + 3 factor matrices. Forward pass contracts through the
factored form instead of materializing the dense tensor.

Reference: T. Kolda & B. Bader, "Tensor Decompositions and Applications", 2009.
Design eye: Dieter Rams meets Gilbert Strang — clean, but the math is rigorous.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Literal, Tuple

import numpy as np

# Prefer PyTorch if available; fall back to pure NumPy.
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False


# ── Helpers ───────────────────────────────────────────────────────────────

Array = "np.ndarray | torch.Tensor"


def _get_array_mod(x: Array) -> type:
    """Return the module (numpy or torch) backing *x*."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return torch
    return np


def _hosvd_factorize(
    W: np.ndarray,
    ranks: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Higher-Order SVD (HOSVD) for a 3-D tensor.

    Returns (core, A, B, C) where W ≈ core ×₁ A ×₂ B ×₃ C.
    """
    d1, d2, d3 = W.shape
    r1, r2, r3 = ranks

    # Mode-1 unfolding: (d1, d2*d3)
    unfold1 = W.reshape(d1, d2 * d3)
    U1, _, _ = np.linalg.svd(unfold1, full_matrices=False)
    A = U1[:, :r1]

    # Mode-2 unfolding: (d2, d1*d3)
    unfold2 = W.transpose(1, 0, 2).reshape(d2, d1 * d3)
    U2, _, _ = np.linalg.svd(unfold2, full_matrices=False)
    B = U2[:, :r2]

    # Mode-3 unfolding: (d3, d1*d2)
    unfold3 = W.transpose(2, 0, 1).reshape(d3, d1 * d2)
    U3, _, _ = np.linalg.svd(unfold3, full_matrices=False)
    C = U3[:, :r3]

    # Core tensor: G = W ×₁ Aᵀ ×₂ Bᵀ ×₃ Cᵀ
    # np.einsum('ijk,ia,jb,kc->abc', W, A, B, C)
    # where A is (d1,r1), B is (d2,r2), C is (d3,r3)
    core = np.einsum("ijk,ia,jb,kc->abc", W, A, B, C)

    return core, A, B, C


def _reconstruct_dense(core: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Reconstruct the dense tensor from Tucker factors.

    W_hat[i1,i2,i3] = Σ_{r1,r2,r3} G[r1,r2,r3] · A[i1,r1] · B[i2,r2] · C[i3,r3]
    """
    return np.einsum("abc,ia,jb,kc->ijk", core, A, B, C)


def _tucker_forward_numpy(
    x: np.ndarray,
    core: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    """Efficient forward pass through Tucker factors (NumPy).

    Given input x of shape (..., d2, d3), returns output of shape (..., d1).

    Contraction chain:
        z1 = einsum('...ij,jc->...ic',  x, C)      # (..., d2, r3)
        z2 = einsum('...ic,ib->...cb',  z1, B)      # (..., r3, r2)
        z3 = einsum('...cb,abc->...a',  z2, core)   # (..., r1)
        y  = einsum('...a,ia->...i',    z3, A)      # (..., d1)
    """
    z1 = np.einsum("...ij,jc->...ic", x, C)
    z2 = np.einsum("...ic,ib->...cb", z1, B)
    z3 = np.einsum("...cb,abc->...a", z2, core)
    y = np.einsum("...a,ia->...i", z3, A)
    return y


def _tucker_forward_torch(
    x: "torch.Tensor",
    core: "torch.Tensor",
    A: "torch.Tensor",
    B: "torch.Tensor",
    C: "torch.Tensor",
) -> "torch.Tensor":
    """Efficient forward pass through Tucker factors (PyTorch).

    Same contraction chain as _tucker_forward_numpy but with torch.einsum.
    """
    z1 = torch.einsum("...ij,jc->...ic", x, C)
    z2 = torch.einsum("...ic,ib->...cb", z1, B)
    z3 = torch.einsum("...cb,abc->...a", z2, core)
    y = torch.einsum("...a,ia->...i", z3, A)
    return y


# ── TuckerLayer ──────────────────────────────────────────────────────────

class TuckerLayer:
    """A 3-D tensor-contraction layer compressed via Tucker decomposition.

    Dense weight shape: (d1, d2, d3) — e.g., (64, 64, 64)
    Core tensor shape:  (r1, r2, r3)   — e.g., (16, 16, 16)
    Factor matrices:     A(d1,r1), B(d2,r2), C(d3,r3)

    Forward pass contracts input ``x`` of shape ``(..., d2, d3)`` through the
    Tucker factors, producing output of shape ``(..., d1)``.

    Parameters
    ----------
    dims :
        Tuple ``(d1, d2, d3)`` giving the dense tensor shape.
    ranks :
        Tuple ``(r1, r2, r3)`` giving the Tucker core shape.
    weight :
        Optional dense tensor to factorize. If ``None``, random init.
    backend :
        ``"numpy"`` or ``"torch"``. Auto-detected if not provided.
    """

    def __init__(
        self,
        dims: Tuple[int, int, int],
        ranks: Tuple[int, int, int],
        weight: Array | None = None,
        backend: Literal["numpy", "torch"] | None = None,
    ):
        self.dims = dims
        self.ranks = ranks
        self.d1, self.d2, self.d3 = dims
        self.r1, self.r2, self.r3 = ranks

        # Resolve backend
        if backend is None:
            if _HAS_TORCH and weight is not None and isinstance(weight, torch.Tensor):
                self.backend = "torch"
            else:
                self.backend = "numpy"
        else:
            self.backend = backend

        xp = torch if self.backend == "torch" else np

        if weight is not None:
            # Factorize provided dense tensor via HOSVD
            if self.backend == "torch":
                W_np = weight.detach().cpu().numpy()
            else:
                W_np = np.asarray(weight)
            core_np, A_np, B_np, C_np = _hosvd_factorize(W_np, ranks)
            if self.backend == "torch":
                self.core = torch.from_numpy(core_np).float()
                self.A = torch.from_numpy(A_np).float()
                self.B = torch.from_numpy(B_np).float()
                self.C = torch.from_numpy(C_np).float()
            else:
                self.core = core_np.astype(np.float32)
                self.A = A_np.astype(np.float32)
                self.B = B_np.astype(np.float32)
                self.C = C_np.astype(np.float32)
        else:
            # Random initialization (Glorot-ish scaling)
            scale = math.sqrt(2.0 / (self.d2 * self.d3 + self.d1))
            if self.backend == "torch":
                self.core = torch.randn(ranks) * scale
                self.A = torch.randn(self.d1, self.r1) * scale
                self.B = torch.randn(self.d2, self.r2) * scale
                self.C = torch.randn(self.d3, self.r3) * scale
            else:
                self.core = np.random.randn(*ranks).astype(np.float32) * scale
                self.A = np.random.randn(self.d1, self.r1).astype(np.float32) * scale
                self.B = np.random.randn(self.d2, self.r2).astype(np.float32) * scale
                self.C = np.random.randn(self.d3, self.r3).astype(np.float32) * scale

    # ── Forward ────────────────────────────────────────────────────────

    def forward(self, x: Array) -> Array:
        """Contract ``x`` (shape ``(..., d2, d3)``) through Tucker factors."""
        if self.backend == "torch":
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(np.asarray(x)).float()
            return _tucker_forward_torch(x, self.core, self.A, self.B, self.C)
        else:
            x = np.asarray(x, dtype=np.float32)
            return _tucker_forward_numpy(x, self.core, self.A, self.B, self.C)

    __call__ = forward

    # ── Diagnostics ────────────────────────────────────────────────────

    def compression_ratio(self) -> float:
        """Dense parameter count / Tucker parameter count."""
        dense_params = self.d1 * self.d2 * self.d3
        tucker_params = (
            self.core.size
            + self.A.size
            + self.B.size
            + self.C.size
        )
        return float(dense_params) / float(tucker_params)

    def param_counts(self) -> dict:
        """Return dict with dense and tucker parameter counts."""
        dense_params = self.d1 * self.d2 * self.d3
        tucker_params = (
            self.core.size
            + self.A.size
            + self.B.size
            + self.C.size
        )
        return {
            "dense": int(dense_params),
            "tucker": int(tucker_params),
            "ratio": float(dense_params) / float(tucker_params),
        }

    def reconstruct(self) -> np.ndarray:
        """Reconstruct dense weight from factors (NumPy array always)."""
        if self.backend == "torch":
            core_np = self.core.detach().cpu().numpy()
            A_np = self.A.detach().cpu().numpy()
            B_np = self.B.detach().cpu().numpy()
            C_np = self.C.detach().cpu().numpy()
        else:
            core_np, A_np, B_np, C_np = self.core, self.A, self.B, self.C
        return _reconstruct_dense(core_np, A_np, B_np, C_np)

    def to_torch_module(self) -> "nn.Module":
        """Return a PyTorch ``nn.Module`` wrapper with registered parameters."""
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is not available")

        class _TuckerTorchModule(nn.Module):
            def __init__(
                self,
                core: "torch.Tensor",
                A: "torch.Tensor",
                B: "torch.Tensor",
                C: "torch.Tensor",
            ):
                super().__init__()
                self.register_parameter("core", nn.Parameter(core.clone()))
                self.register_parameter("A", nn.Parameter(A.clone()))
                self.register_parameter("B", nn.Parameter(B.clone()))
                self.register_parameter("C", nn.Parameter(C.clone()))

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                return _tucker_forward_torch(x, self.core, self.A, self.B, self.C)

        return _TuckerTorchModule(self.core, self.A, self.B, self.C)


# ── CLI / quick test ──────────────────────────────────────────────────────

def _demo() -> None:
    """Run a quick sanity check from the command line."""
    dims = (64, 64, 64)
    ranks = (16, 16, 16)

    # Create a synthetic dense weight with structure (smooth + noise)
    rng = np.random.default_rng(42)
    W = np.zeros(dims, dtype=np.float32)
    for i in range(dims[0]):
        for j in range(dims[1]):
            W[i, j, :] = np.sin(0.1 * i + 0.05 * j) + 0.1 * rng.standard_normal(dims[2])

    layer = TuckerLayer(dims, ranks, weight=W)
    counts = layer.param_counts()
    print(f"Dense params:   {counts['dense']:,}")
    print(f"Tucker params:  {counts['tucker']:,}")
    print(f"Compression:    {counts['ratio']:.2f}×")

    # Reconstruction quality
    W_hat = layer.reconstruct()
    rel_error = float(np.linalg.norm(W - W_hat) / np.linalg.norm(W))
    print(f"Relative error: {rel_error:.4f}")

    # Forward pass
    x = rng.standard_normal((4, dims[1], dims[2])).astype(np.float32)
    y = layer(x)
    print(f"Input shape:    {x.shape}")
    print(f"Output shape:   {y.shape}")

    # Torch variant
    if _HAS_TORCH:
        layer_t = TuckerLayer(dims, ranks, weight=torch.from_numpy(W), backend="torch")
        x_t = torch.from_numpy(x)
        y_t = layer_t(x_t)
        print(f"Torch output:   {tuple(y_t.shape)}")


if __name__ == "__main__":
    _demo()
