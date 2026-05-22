import numpy as np
from numba import njit
from nerve.room_grid import batch_novelty, _batch_novelty_numpy
import time

@njit(cache=True, fastmath=True)
def _routing_numba_core(nv, chaos, n):
    fired_mask = np.empty(n, dtype=np.bool_)
    new_chaos = np.empty(n, dtype=np.float32)
    fired_count = 0
    for i in range(n):
        chaos_fire = np.random.random() < chaos[i]
        fire = (nv[i] > 0.5) or chaos_fire
        fired_mask[i] = fire
        if fire:
            new_chaos[i] = max(0.01, chaos[i] * 0.99)
            fired_count += 1
        else:
            new_chaos[i] = chaos[i]
    return fired_mask, new_chaos, fired_count

def _compiled_routing(latents, chaos, n, hist, hist_count, hist_idx, hist_max):
    nv = batch_novelty(latents, hist, hist_count, hist_idx, hist_max)
    return _routing_numba_core(nv, chaos, n)

def _original_routing(latents, chaos, n, hist, hist_count, hist_idx, hist_max):
    nv = batch_novelty(latents, hist, hist_count, hist_idx, hist_max)
    chaos_fire = np.random.random(n) < chaos
    fired_mask = (nv > 0.5) | chaos_fire
    new_chaos = np.where(fired_mask, np.maximum(0.01, chaos * 0.99), chaos)
    fired_count = int(fired_mask.sum())
    return fired_mask, new_chaos, fired_count

for n in [100, 250, 500, 1000]:
    np.random.seed(42)
    latents = np.random.randn(n, 16).astype(np.float32)
    chaos = np.full(n, 0.3, dtype=np.float32)
    hist = np.zeros((20, n, 16), dtype=np.float32)
    hist_count = np.full(n, 5, dtype=np.int32)
    hist_idx = 5
    hist_max = 20

    _compiled_routing(latents, chaos, n, hist, hist_count, hist_idx, hist_max)
    _original_routing(latents, chaos, n, hist, hist_count, hist_idx, hist_max)

    trials = 50
    t0 = time.perf_counter()
    for _ in range(trials):
        _compiled_routing(latents, chaos, n, hist, hist_count, hist_idx, hist_max)
    t_numba = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(trials):
        _original_routing(latents, chaos, n, hist, hist_count, hist_idx, hist_max)
    t_numpy = (time.perf_counter() - t0) * 1000

    print(f"n={n:4d}  numba={t_numba/trials:.3f}ms  numpy={t_numpy/trials:.3f}ms  speedup={t_numpy/t_numba:.2f}x")
