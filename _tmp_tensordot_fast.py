import numpy as np
from nerve.room_grid import forward_einsum
import time

def forward_tensordot(w, x):
    x = x.ravel().astype(np.float32)
    # layer 1: tensordot over axis 1 of w1 and axis 0 of x
    h = np.tensordot(w["w1"], x, axes=([1], [0])) + w["b1"][0]
    np.maximum(h, 0, out=h)
    # layer 2: h(n,32) tensordot w2(n,32,16) over axis 1 of h and axis 1 of w2? No.
    # h[i] @ w2[i] for each i. We want h[i](32) dot w2[i](32,16) -> h2[i](16)
    # tensordot over the 32 dim: axes=([1],[1]) would give (n,n,16) — wrong.
    # Use einsum for layer 2 and 3.
    h2 = np.einsum("nh,nhl->nl", h, w["w2"], optimize=False) + w["b2"][0]
    np.maximum(h2, 0, out=h2)
    out = np.einsum("nl,nll->nl", h2, w["w3"], optimize=False) + w["b3"][0]
    return out

for n in [100, 250, 500, 1000]:
    np.random.seed(42)
    w = {
        "w1": np.random.randn(n, 64, 32).astype(np.float32) * 0.01,
        "b1": np.zeros((1, n, 32), dtype=np.float32),
        "w2": np.random.randn(n, 32, 16).astype(np.float32) * 0.01,
        "b2": np.zeros((1, n, 16), dtype=np.float32),
        "w3": np.broadcast_to(np.eye(16, dtype=np.float32) * 0.99, (n, 16, 16)).copy(),
        "b3": np.zeros((1, n, 16), dtype=np.float32),
    }
    x = np.random.randn(64).astype(np.float32)
    trials = 50
    t0 = time.perf_counter()
    for _ in range(trials):
        forward_einsum(w, x)
    t_orig = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    for _ in range(trials):
        forward_tensordot(w, x)
    t_opt = (time.perf_counter() - t0) * 1000
    print(f"n={n:4d}  einsum={t_orig/trials:.3f}ms  tensordot={t_opt/trials:.3f}ms  speedup={t_orig/t_opt:.2f}x")
    expected = forward_einsum(w, x)
    actual = forward_tensordot(w, x)
    diff = np.max(np.abs(expected - actual))
    print(f"  diff={diff:.2e}")
