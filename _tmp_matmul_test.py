import numpy as np
from nerve.room_grid import forward_einsum
import time

def forward_matmul(w, x):
    x = x.ravel().astype(np.float32)
    # x (64,) @ w1 (n, 64, 32) -> (n, 32)
    h = np.matmul(x, w["w1"]) + w["b1"][0]
    np.maximum(h, 0, out=h)
    h2 = np.matmul(h, w["w2"]) + w["b2"][0]
    np.maximum(h2, 0, out=h2)
    out = np.matmul(h2, w["w3"]) + w["b3"][0]
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
        forward_matmul(w, x)
    t_opt = (time.perf_counter() - t0) * 1000

    print(f"n={n:4d}  einsum={t_orig/trials:.3f}ms  matmul={t_opt/trials:.3f}ms  speedup={t_orig/t_opt:.2f}x")
    expected = forward_einsum(w, x)
    actual = forward_matmul(w, x)
    diff = np.max(np.abs(expected - actual))
    print(f"  diff={diff:.2e}")
