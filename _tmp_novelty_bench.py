import numpy as np
from nerve.room_grid import batch_novelty, _batch_novelty_numpy
import time

n = 1000
np.random.seed(42)
latents = np.random.randn(n, 16).astype(np.float32)
hist = np.random.randn(20, n, 16).astype(np.float32)
hist_count = np.full(n, 5, dtype=np.int32)
hist_idx = 5
hist_max = 20

# warmup
batch_novelty(latents, hist, hist_count, hist_idx, hist_max)
_batch_novelty_numpy(latents, hist, hist_count, hist_idx, hist_max)

trials = 50
t0 = time.perf_counter()
for _ in range(trials):
    batch_novelty(latents, hist, hist_count, hist_idx, hist_max)
t = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
for _ in range(trials):
    _batch_novelty_numpy(latents, hist, hist_count, hist_idx, hist_max)
t2 = (time.perf_counter() - t0) * 1000

print(f"batch_novelty (auto): {t/trials:.3f}ms")
print(f"_batch_novelty_numpy: {t2/trials:.3f}ms")
print(f"speedup of auto vs numpy: {t2/t:.2f}x")
