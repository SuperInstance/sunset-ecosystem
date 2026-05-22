#!/usr/bin/env python3
"""CUDA Kernel Profiler — Break down tick latency for FM.

Run on RTX 4050 (or any CUDA-capable machine) to measure:
    1. Memory transfer time (H2D signal + D2H output)
    2. Kernel execution time
    3. Total tick time
    4. Batch tick amortization

Usage::

    cd sunset-ecosystem
    python3 scripts/profile_cuda.py --rooms 10000 --batch 1,4,8,16

Requirements:
    - libjepa_cuda.so compiled (nvcc -O3 -shared -Xcompiler -fPIC ...)
    - PyCUDA or simple ctypes timing (this script uses ctypes + time.perf_counter)

Output:
    Markdown table written to stdout + saved to docs/CUDA_PROFILE.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make sure we can import nerve
sys.path.insert(0, str(Path(__file__).parent.parent))

from nerve.cuda_bridge import PersistentCUDAGrid
from nerve.room_grid import make_weights


def profile_single(grid: PersistentCUDAGrid, n: int, warmup: int = 10, runs: int = 100) -> dict:
    """Profile single-tick latency."""
    signal = np.random.randn(64).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        grid.tick(signal)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        grid.tick(signal)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    times = np.array(times)
    return {
        "n": n,
        "batch": 1,
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "min_ms": float(times.min()),
        "p99_ms": float(np.percentile(times, 99)),
        "throughput_rooms_per_sec": n / (times.mean() / 1000.0),
    }


def profile_batch(grid: PersistentCUDAGrid, n: int, batch: int, warmup: int = 5, runs: int = 50) -> dict:
    """Profile batch-tick latency."""
    signals = np.random.randn(batch, 64).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        grid.tick_batch(signals)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        grid.tick_batch(signals)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    times = np.array(times)
    total_rooms = n * batch
    return {
        "n": n,
        "batch": batch,
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "min_ms": float(times.min()),
        "p99_ms": float(np.percentile(times, 99)),
        "throughput_rooms_per_sec": total_rooms / (times.mean() / 1000.0),
        "amortized_ms_per_room": float(times.mean()) / batch,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile CUDA JEPA kernel")
    parser.add_argument("--rooms", type=int, default=10000, help="Number of rooms")
    parser.add_argument("--batch", type=str, default="1,4,8,16", help="Batch sizes to test (comma-separated)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Timed iterations")
    parser.add_argument("--output", type=str, default="docs/CUDA_PROFILE.md", help="Output markdown file")
    args = parser.parse_args()

    n = args.rooms
    batch_sizes = [int(b.strip()) for b in args.batch.split(",")]

    # Build weights
    weights = make_weights(n, d=64, h=32, l=16, seed=42)

    try:
        grid = PersistentCUDAGrid(n, weights)
    except RuntimeError as exc:
        print(f"❌ CUDA backend not available: {exc}")
        print("Compile with:")
        print("  nvcc -O3 -shared -Xcompiler -fPIC -o nerve/libjepa_cuda.so nerve/src/jepa_kernel.cu")
        sys.exit(1)

    print(f"🚀 Profiling CUDA kernel: {n} rooms")
    print(f"   Batch sizes: {batch_sizes}")
    print(f"   Warmup: {args.warmup}, Runs: {args.runs}")
    print()

    results = []
    for batch in batch_sizes:
        if batch == 1:
            r = profile_single(grid, n, warmup=args.warmup, runs=args.runs)
        else:
            r = profile_batch(grid, n, batch, warmup=args.warmup, runs=args.runs)
        results.append(r)
        print(f"  batch={batch:2d}: {r['mean_ms']:.2f}ms  "
              f"({r['throughput_rooms_per_sec']/1e6:.2f}M rooms/sec)")

    # Generate markdown report
    lines = [
        "# CUDA Kernel Profile Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Device:** {n} rooms, RTX 4050 laptop (or detected GPU)",
        f"**Kernel:** `jepa_forward_kernel` / `jepa_forward_batch_kernel`",
        "",
        "## Results",
        "",
        "| Batch | Mean (ms) | Median (ms) | Min (ms) | P99 (ms) | Throughput (M rooms/sec) |",
        "|-------|-----------|-------------|----------|----------|---------------------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['batch']:5d} | {r['mean_ms']:9.2f} | {r['median_ms']:11.2f} | "
            f"{r['min_ms']:8.2f} | {r['p99_ms']:8.2f} | "
            f"{r['throughput_rooms_per_sec']/1e6:25.2f} |"
        )

    lines += [
        "",
        "## Bottleneck Analysis",
        "",
        "If `batch=1` is significantly slower than `batch>1` amortized per tick,",
        "the bottleneck is **kernel launch overhead**, not compute.",
        "",
        "If all batch sizes scale linearly with total rooms, the bottleneck is",
        "**compute-bound** (not enough SMs or memory bandwidth).",
        "",
        "If `batch=1` is ~same as `batch=4` divided by 4, the bottleneck is",
        "**memory-transfer-bound** (PCIe bandwidth limited).",
        "",
        "## Recommendations",
        "",
        "- **If launch overhead dominates:** Use `RoomGrid.tick_batch()` with batch ≥ 4.",
        "- **If compute dominates:** Increase occupancy (more threads per block) or",
        "  use Tensor Cores (wmma) for the 64×32 and 32×16 matmuls.",
        "- **If memory dominates:** Keep weights in GPU memory persistently",
        "  (persistent CUDA grid, not per-tick H2D copy).",
        "",
        "---",
        "*Generated by scripts/profile_cuda.py*",
    ]

    md = "\n".join(lines)
    print()
    print(md)

    # Write to file
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"\n💾 Report saved to {out_path}")


if __name__ == "__main__":
    main()
