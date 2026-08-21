"""Benchmark the full topology tick cycle."""

import time
import numpy as np
from nerve.topology import NerveTopology


def bench_topology(n_fibers=8, n_rooms=250, ticks=100):
    topo = NerveTopology(n_fibers=n_fibers, n_rooms=n_rooms)

    # Warmup
    for _ in range(10):
        topo.tick()

    t0 = time.perf_counter()
    for _ in range(ticks):
        topo.tick()
    elapsed = time.perf_counter() - t0

    avg_ms = elapsed / ticks * 1000
    print(f"NerveTopology({n_fibers} fibers, {n_rooms} rooms):")
    print(f"  {ticks} ticks in {elapsed:.3f}s")
    print(f"  {avg_ms:.2f} ms/tick")
    print(f"  Stats: {topo.stats}")

    return avg_ms


if __name__ == "__main__":
    # Small
    bench_topology(4, 50, 200)
    # Medium
    bench_topology(8, 250, 200)
    # Large
    bench_topology(8, 1000, 100)
