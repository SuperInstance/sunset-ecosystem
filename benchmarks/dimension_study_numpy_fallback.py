#!/usr/bin/env python3
"""Dimension Study — numpy-only fallback (turbovec blocked by missing libopenblas-dev).

Uses brute-force numpy cosine similarity to compare dimensions.
Not production-representative for turbovec, but gives relative
memory and latency trends across 128/256/384/512.

Usage::

    python benchmarks/dimension_study_numpy_fallback.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

DIMENSIONS = [128, 256, 384, 512]
POPULATION = 10_000  # Smaller for brute-force
K = 5
WARMUP = 10


def brute_search(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Brute-force top-k cosine similarity."""
    # vectors: (N, D), query: (D,)
    dots = vectors @ query
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
    scores = dots / (norms + 1e-8)
    top_k = np.argpartition(scores, -k)[-k:]
    return top_k[np.argsort(scores[top_k])][::-1]


def benchmark_dim(dim: int) -> dict:
    print(f"\n{'='*60}")
    print(f"Dimension: {dim} (numpy brute-force fallback)")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Build population
    print(f"Building {POPULATION} agents...")
    t0 = time.perf_counter()
    vectors = rng.standard_normal((POPULATION, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    build_time = time.perf_counter() - t0
    print(f"  Build: {build_time:.2f}s")

    # Warmup
    for _ in range(WARMUP):
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-8
        brute_search(vectors, q, K)

    # Benchmark
    latencies: list[float] = []
    for _ in range(100):
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-8
        t0 = time.perf_counter()
        brute_search(vectors, q, K)
        latencies.append(time.perf_counter() - t0)

    avg_ms = np.mean(latencies) * 1000
    p99_ms = np.percentile(latencies, 99) * 1000

    # Memory
    naive_mb = (POPULATION * dim * 4) / (1024 * 1024)
    # Simulated 4-bit turbovec
    turbovec_mb = (POPULATION * dim * 0.5) / (1024 * 1024) + (POPULATION * 40) / (1024 * 1024)

    print(f"  Avg latency: {avg_ms:.3f}ms")
    print(f"  P99 latency: {p99_ms:.3f}ms")
    print(f"  Naive memory: {naive_mb:.1f}MB")
    print(f"  Simulated turbovec: {turbovec_mb:.1f}MB ({naive_mb/turbovec_mb:.1f}x compression)")

    return {
        "dim": dim,
        "build_time": build_time,
        "avg_latency_ms": avg_ms,
        "p99_latency_ms": p99_ms,
        "naive_memory_mb": naive_mb,
        "simulated_turbovec_mb": turbovec_mb,
        "compression_ratio": naive_mb / turbovec_mb,
    }


def main() -> None:
    print("=" * 60)
    print("Dimension Study — Numpy Fallback (turbovec blocked)")
    print(f"Population: {POPULATION:,} agents")
    print("WARNING: Latency numbers are brute-force, NOT turbovec SIMD")
    print("=" * 60)

    results: list[dict] = []
    for dim in DIMENSIONS:
        try:
            results.append(benchmark_dim(dim))
        except Exception as exc:
            print(f"  ❌ FAILED: {exc}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Dim':>6} {'Build':>8} {'Avg ms':>10} {'P99 ms':>10} {'Naive MB':>10} {'TVec MB':>10} {'Compress':>10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['dim']:>6} "
            f"{r['build_time']:>7.2f}s "
            f"{r['avg_latency_ms']:>9.2f} "
            f"{r['p99_latency_ms']:>9.2f} "
            f"{r['naive_memory_mb']:>9.1f} "
            f"{r['simulated_turbovec_mb']:>9.1f} "
            f"{r['compression_ratio']:>9.1f}x"
        )

    # Recommendation based on memory compression (latency is brute-force)
    if results:
        best = max(results, key=lambda r: r["compression_ratio"])
        print(f"\n✅ Best compression: dim={best['dim']} ({best['compression_ratio']:.1f}x)")
        print("⚠️  Latency numbers are NOT turbovec — real SIMD will be 10-100x faster")

    import json
    out = Path("/tmp/sunset-ecosystem/benchmarks/dimension_study_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
