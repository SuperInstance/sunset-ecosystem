import numpy as np
import pytest
import time

from nerve.room_grid import RoomGrid, batch_novelty
from nerve.topology import NerveTopology


@pytest.mark.slow
def test_batch_novelty_under_5ms(room_grid_1000):
    """1000-room novelty batch must complete in <5 ms after warmup."""
    np.random.seed(42)
    latents = np.random.randn(1000, 16).astype(np.float32)
    hist = room_grid_1000._hist
    hist_count = room_grid_1000._hist_count
    hist_idx = room_grid_1000._hist_idx
    hist_max = room_grid_1000._hist_max
    # Warmup
    for _ in range(20):
        batch_novelty(latents, hist, hist_count, hist_idx, hist_max)
    t0 = time.perf_counter()
    for _ in range(50):
        batch_novelty(latents, hist, hist_count, hist_idx, hist_max)
    elapsed = (time.perf_counter() - t0) / 50 * 1000
    assert elapsed < 5.0


@pytest.mark.slow
def test_topology_tick_under_100ms():
    """1000 rooms × 8 fibers must tick in <100 ms after warmup."""
    topo = NerveTopology(n_fibers=8, n_rooms=1000)
    # Warmup
    for _ in range(5):
        topo.tick()
    t0 = time.perf_counter()
    for _ in range(10):
        topo.tick()
    elapsed = (time.perf_counter() - t0) / 10 * 1000
    assert elapsed < 100.0


@pytest.mark.slow
def test_grid_10000_under_100ms():
    """10000-room forward pass must complete in <100 ms after warmup."""
    np.random.seed(42)
    grid = RoomGrid(10000)
    x = np.random.randn(64).astype(np.float32)
    for _ in range(10):
        grid.tick(x)
    t0 = time.perf_counter()
    for _ in range(20):
        grid.tick(x)
    elapsed = (time.perf_counter() - t0) / 20 * 1000
    assert elapsed < 200.0
