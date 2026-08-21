"""Benchmark: turbovec vs numpy for agent DNA similarity search.

Measures query latency and memory usage for fleet-sized agent populations.
Run with: python benchmarks/turbovec_vs_numpy.py
"""

from __future__ import annotations

import random
import sys
import time

import numpy as np

try:
    from swarm.vector_table import AgentVector, FluxVectorTable

    HAS_TURBOVEC = True
except ImportError:
    HAS_TURBOVEC = False
    print("turbovec not installed; install with: pip install -e '.[vecsearch]'")
    sys.exit(1)


def make_random_vector(dim: int) -> list[float]:
    """Generate a random unit vector."""
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def benchmark_turbovec(n_agents: int, dim: int, bit_width: int, k: int) -> dict:
    """Run turbovec benchmark."""
    table = FluxVectorTable(dim=dim, bit_width=bit_width)

    # Insert
    t0 = time.perf_counter()
    for i in range(n_agents):
        table.add(
            AgentVector(
                agent_id=i,
                vector=make_random_vector(dim),
                fitness=random.random(),
                generation=random.randint(0, 100),
                capability_mask=random.randint(1, 0xFFFF),
                thermal_pressure=random.random(),
            )
        )
    insert_time = time.perf_counter() - t0

    # Query
    query = make_random_vector(dim)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = table.search(query, k=k)
    query_time = (time.perf_counter() - t0) / 100

    return {
        "backend": "turbovec",
        "n_agents": n_agents,
        "dim": dim,
        "bit_width": bit_width,
        "insert_ms": insert_time * 1000,
        "query_ms": query_time * 1000,
        "memory_mb": "unknown",  # turbovec internal; estimate below
    }


def benchmark_numpy(n_agents: int, dim: int, k: int) -> dict:
    """Run naive numpy benchmark."""
    vectors = np.array(
        [make_random_vector(dim) for _ in range(n_agents)], dtype=np.float32
    )
    ids = np.arange(n_agents, dtype=np.uint64)
    meta = {
        i: {
            "fitness": random.random(),
            "generation": random.randint(0, 100),
            "capability_mask": random.randint(1, 0xFFFF),
            "thermal_pressure": random.random(),
        }
        for i in range(n_agents)
    }

    # Insert (trivial for numpy — already built)
    insert_time = 0.0

    # Query: brute-force top-k via dot product
    query = np.array(make_random_vector(dim), dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(100):
        scores = vectors @ query
        top_k = np.argpartition(scores, -k)[-k:]
        _ = [(int(ids[i]), float(scores[i]), meta[int(ids[i])]) for i in top_k]
    query_time = (time.perf_counter() - t0) / 100

    raw_bytes = vectors.nbytes
    meta_bytes = sys.getsizeof(meta)

    return {
        "backend": "numpy",
        "n_agents": n_agents,
        "dim": dim,
        "bit_width": 32,
        "insert_ms": insert_time,
        "query_ms": query_time * 1000,
        "memory_mb": (raw_bytes + meta_bytes) / (1024 * 1024),
    }


def run_all():
    configs = [
        (1000, 256, 4),
        (10000, 256, 4),
        (50000, 384, 4),
        (100000, 512, 4),
    ]
    k = 10

    print("=" * 70)
    print("Turbovec vs Numpy — Agent DNA Similarity Search Benchmark")
    print("=" * 70)

    for n, dim, bw in configs:
        print(f"\n--- Config: n={n}, dim={dim}, bit_width={bw}, k={k} ---")

        np_result = benchmark_numpy(n, dim, k)
        tv_result = benchmark_turbovec(n, dim, bw, k)

        # Estimate turbovec memory: compressed vectors + scales + overhead
        bytes_per_vec = dim * bw // 8
        scale_bytes = 4
        overhead = 0.1  # 10% index overhead
        tv_memory = (n * (bytes_per_vec + scale_bytes) * (1 + overhead)) / (1024 * 1024)
        tv_result["memory_mb"] = tv_memory

        # Speedup
        speedup = np_result["query_ms"] / tv_result["query_ms"]
        compression = np_result["memory_mb"] / tv_result["memory_mb"]

        print(
            f"  numpy:  query={np_result['query_ms']:.3f} ms, memory={np_result['memory_mb']:.1f} MB"
        )
        print(
            f"  turbovec: query={tv_result['query_ms']:.3f} ms, memory={tv_result['memory_mb']:.1f} MB"
        )
        print(f"  speedup: {speedup:.1f}×, compression: {compression:.1f}×")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    run_all()
