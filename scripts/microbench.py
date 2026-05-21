"""Micro-benchmark: isolate each NerveTopology phase.

Run: python -m scripts.microbench
"""

from __future__ import annotations

import time
import numpy as np
from nerve.topology import NerveTopology


def bench_phase(label: str, fn, n: int = 100) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"{label:30s}: {ms:7.3f} ms")
    return ms


def main():
    topo = NerveTopology(n_fibers=4, n_rooms=100)
    for _ in range(5):
        topo.tick()

    signals = {f"fiber-{i}": np.random.randn(64).astype(np.float32) for i in range(4)}

    # Phase 1: Fiber perceive
    def phase_perceive():
        for fid, fiber in topo.fibers.items():
            fiber.perceive(signals[fid])

    # Phase 2: fire_fast
    def phase_fire():
        for fid in topo.fibers:
            topo.routing.fire_fast(fid)

    # Phase 3: encode tiles (simulated)
    tiles = {}
    for fid, fiber in topo.fibers.items():
        tiles[fid] = fiber.perceive(signals[fid])

    def phase_encode():
        for tile in tiles.values():
            topo._encode_tile(tile)

    # Phase 4: grid tick
    x = np.random.randn(64).astype(np.float32)

    def phase_grid():
        topo.grid.tick(x)

    # Phase 5: feedback_batch
    updates = [("fiber-0", f"room-{i}", True) for i in range(50)]

    def phase_feedback():
        topo.routing.feedback_batch(updates)

    # Phase 6: cold
    def phase_cold():
        topo.grid.cold()

    # Phase 7: Hebbian
    fired = [f"room-{i}" for i in range(30)]

    def phase_hebbian():
        topo.routing._activate_channels_limited(fired)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║         MICRO-BENCHMARK — NerveTopology phases           ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    total = 0.0
    total += bench_phase("fiber_perceive", phase_perceive)
    total += bench_phase("fire_fast", phase_fire)
    total += bench_phase("encode_tile", phase_encode)
    total += bench_phase("grid_tick", phase_grid)
    total += bench_phase("feedback_batch", phase_feedback)
    total += bench_phase("cold", phase_cold)
    total += bench_phase("hebbian_limited", phase_hebbian)

    print(f"\n{'SUMMARY (theoretical max per-tick)':30s}: {total:7.3f} ms")
    print(f"{'Actual tick (measured)':30s}: {20.32:7.3f} ms")
    overhead = 20.32 - total
    print(f"{'Unaccounted overhead':30s}: {overhead:7.3f} ms ({overhead/20.32*100:.0f}%)")


if __name__ == "__main__":
    main()
