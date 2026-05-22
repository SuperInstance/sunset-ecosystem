from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Preload openblas if needed
try:
    from turbovec import IdMapIndex
except ImportError:
    os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0"
    print("⚠️  Setting LD_PRELOAD for openblas — turbovec wheel linking issue")
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, "/tmp/sunset-ecosystem")

from swarm.vector_table import FluxVectorTable, AgentVector

DIMENSIONS = [128, 256, 384, 512]
POPULATION = 10_000
K = 5
WARMUP = 50
QUERIES = 100


def benchmark_dim(dim: int) -> dict:
    print(f"\n{'='*60}")
    print(f"Dimension: {dim} (TURBOVEC SIMD)")
    print("=" * 60)

    table = FluxVectorTable(dim=dim, bit_width=4)
    rng = np.random.default_rng(42)

    print(f"Building {POPULATION} agents...")
    t0 = time.perf_counter()
    for i in range(POPULATION):
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        table.add(
            AgentVector(
                agent_id=i,
                vector=vec.tolist(),
                fitness=float(rng.random()),
                generation=i // 100,
                capability_mask=int(rng.integers(0, 65536)),
                thermal_pressure=float(rng.random() * 0.5),
            )
        )
    build_time = time.perf_counter() - t0
    print(f"  Build: {build_time:.2f}s ({POPULATION/build_time:.0f} agents/s)")

    # Warmup
    query = rng.standard_normal(dim).astype(np.float32)
    query /= np.linalg.norm(query) + 1e-8
    for _ in range(WARMUP):
        table.search(query.tolist(), k=K)

    # Latency benchmark
    latencies: list[float] = []
    for _ in range(QUERIES):
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-8
        t0 = time.perf_counter()
        results = table.search(q.tolist(), k=K)
        latencies.append(time.perf_counter() - t0)

    avg_latency = np.mean(latencies) * 1000
    p99_latency = np.percentile(latencies, 99) * 1000

    # Memory
    vector_bytes = POPULATION * dim * 0.5
    meta_bytes = POPULATION * 40
    total_mb = (vector_bytes + meta_bytes) / (1024 * 1024)
    naive_mb = (POPULATION * dim * 4) / (1024 * 1024)
    compression = naive_mb / total_mb

    print(f"  Avg latency: {avg_latency:.3f}ms")
    print(f"  P99 latency: {p99_latency:.3f}ms")
    print(f"  Memory: {total_mb:.1f}MB (naive {naive_mb:.1f}MB, {compression:.1f}x compression)")

    return {
        "dim": dim,
        "build_time": build_time,
        "avg_latency_ms": avg_latency,
        "p99_latency_ms": p99_latency,
        "memory_mb": total_mb,
        "compression_ratio": compression,
    }


def main() -> None:
    print("=" * 60)
    print("Turbovec DNA Dimension Study — REAL SIMD (Light)")
    print(f"Population: {POPULATION:,} agents")
    print(f"Quantization: 4-bit per dimension")
    print(f"Backend: turbovec {IdMapIndex}")
    print("=" * 60)

    results: list[dict] = []
    for dim in DIMENSIONS:
        try:
            results.append(benchmark_dim(dim))
        except Exception as exc:
            print(f"  ❌ FAILED: {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Dim':>6} {'Build':>8} {'Avg ms':>10} {'P99 ms':>10} {'Memory':>10} {'Compress':>10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['dim']:>6} "
            f"{r['build_time']:>7.1f}s "
            f"{r['avg_latency_ms']:>9.3f} "
            f"{r['p99_latency_ms']:>9.3f} "
            f"{r['memory_mb']:>9.1f}MB "
            f"{r['compression_ratio']:>9.1f}x"
        )

    if results:
        best = min(results, key=lambda r: r["avg_latency_ms"] + r["memory_mb"] * 0.1)
        print(f"\n✅ Recommended: dim={best['dim']} (best latency/memory tradeoff)")

    out = Path("/tmp/sunset-ecosystem/benchmarks/dimension_study_turbovec_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
