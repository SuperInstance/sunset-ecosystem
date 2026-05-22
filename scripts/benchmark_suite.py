#!/usr/bin/env python3
"""
Sunset Ecosystem Benchmark Suite
==================================

Systematic performance measurement across all backends and scales.

Usage:
    PYTHONPATH=$(pwd) python3 scripts/benchmark_suite.py [output.json]

Measures:
  - Room count scaling: 100, 500, 1000, 5000, 10000
  - Backends: numpy, rust_oneshot, rust_persistent
  - Signal types: structured (sine), random, mixed
  - Metrics: latency p50/p99, throughput, memory

Output:
  JSON file with all measurements for regression detection.
"""
from __future__ import annotations

import json, sys, time, os
import numpy as np

from nerve.topology import NerveTopology


def benchmark_grid_backends():
    """Benchmark RoomGrid._forward() across backends and room counts."""
    from nerve.room_grid import RoomGrid, _select_backend

    results = []
    room_counts = [100, 500, 1000, 5000, 10000]

    for n in room_counts:
        grid = RoomGrid(n)
        backend = _select_backend(n)

        # Warmup
        for _ in range(5):
            grid.tick(np.random.randn(64))

        # Benchmark
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            grid.tick(np.random.randn(64))
            latencies.append((time.perf_counter() - t0) * 1000)

        results.append({
            "n_rooms": n,
            "backend": backend,
            "avg_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "ticks_per_sec": 50 / (sum(latencies) / 1000),
        })

    return results


def benchmark_topology_scaling():
    """Benchmark full topology at different scales."""
    results = []
    configs = [
        {"n_rooms": 100, "n_fibers": 4},
        {"n_rooms": 500, "n_fibers": 8},
        {"n_rooms": 1000, "n_fibers": 8},
        {"n_rooms": 2000, "n_fibers": 16},
    ]

    for cfg in configs:
        topo = NerveTopology(**cfg)
        n_ticks = 50

        # Warmup
        for _ in range(5):
            signals = {fid: np.random.randn(64).astype(np.float32) for fid in topo.fibers}
            topo.tick(signals)

        # Structured signals (sine waves)
        latencies_structured = []
        for _ in range(n_ticks):
            signals = {
                fid: np.sin(np.linspace(0, 4*np.pi, 64) + i*0.5).astype(np.float32)
                for i, fid in enumerate(topo.fibers)
            }
            t0 = time.perf_counter()
            topo.tick(signals)
            latencies_structured.append((time.perf_counter() - t0) * 1000)

        # Random signals
        latencies_random = []
        for _ in range(n_ticks):
            signals = {fid: np.random.randn(64).astype(np.float32) for fid in topo.fibers}
            t0 = time.perf_counter()
            topo.tick(signals)
            latencies_random.append((time.perf_counter() - t0) * 1000)

        results.append({
            "config": cfg,
            "backend": topo.grid.__repr__().split("backend=")[1].rstrip(")"),
            "structured": {
                "avg_ms": float(np.mean(latencies_structured)),
                "p50_ms": float(np.percentile(latencies_structured, 50)),
                "p99_ms": float(np.percentile(latencies_structured, 99)),
                "ticks_per_sec": n_ticks / (sum(latencies_structured) / 1000),
            },
            "random": {
                "avg_ms": float(np.mean(latencies_random)),
                "p50_ms": float(np.percentile(latencies_random, 50)),
                "p99_ms": float(np.percentile(latencies_random, 99)),
                "ticks_per_sec": n_ticks / (sum(latencies_random) / 1000),
            },
        })

    return results


def benchmark_novelty():
    """Benchmark batch_novelty in isolation."""
    from nerve.room_grid import batch_novelty

    results = []
    for n in [1000, 5000, 10000]:
        latents = np.random.randn(n, 16).astype(np.float32)
        hist = np.random.randn(20, n, 16).astype(np.float32)
        hist_count = np.ones(n, dtype=np.int32) * 5

        # Warmup
        for _ in range(5):
            batch_novelty(latents, hist, hist_count, 5, 20)

        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            batch_novelty(latents, hist, hist_count, 5, 20)
            latencies.append((time.perf_counter() - t0) * 1000)

        results.append({
            "n": n,
            "avg_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "calls_per_sec": 100 / (sum(latencies) / 1000),
        })

    return results


def main():
    output_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sunset_benchmark.json"

    print("=== Sunset Ecosystem Benchmark Suite ===")
    print()

    print("[1/3] Grid backend scaling...")
    grid_results = benchmark_grid_backends()
    for r in grid_results:
        print(f"  {r['n_rooms']:5d} rooms: {r['avg_ms']:6.2f}ms/tick ({r['backend']})")
    print()

    print("[2/3] Topology full-stack scaling...")
    topo_results = benchmark_topology_scaling()
    for r in topo_results:
        cfg = r["config"]
        print(f"  {cfg['n_rooms']:4d} rooms × {cfg['n_fibers']:2d} fibers:")
        print(f"    structured: {r['structured']['avg_ms']:6.2f}ms/tick ({r['structured']['ticks_per_sec']:.0f}/s)")
        print(f"    random:     {r['random']['avg_ms']:6.2f}ms/tick ({r['random']['ticks_per_sec']:.0f}/s)")
        print(f"    backend:    {r['backend']}")
    print()

    print("[3/3] Novelty function isolation...")
    novelty_results = benchmark_novelty()
    for r in novelty_results:
        print(f"  {r['n']:5d} rooms: {r['avg_ms']:.3f}ms/call ({r['calls_per_sec']:,.0f}/s)")
    print()

    report = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hostname": os.uname().nodename,
            "python": sys.version.split()[0],
        },
        "grid_backends": grid_results,
        "topology_scaling": topo_results,
        "novelty_isolation": novelty_results,
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[report] written to {output_file}")
    print("=== Benchmark Complete ===")


if __name__ == "__main__":
    main()
