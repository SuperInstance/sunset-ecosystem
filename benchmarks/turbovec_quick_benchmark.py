#!/usr/bin/env python3
"""Quick turbovec benchmark — 10K agents, 4 dims, 50 queries each.
Writes progress to stdout for monitoring."""

import sys
import time
sys.path.insert(0, "/tmp/sunset-ecosystem")

import numpy as np
from swarm.vector_table import FluxVectorTable, AgentVector

DIMS = [128, 256, 384, 512]
POP = 10_000
QUERIES = 50
K = 5

results = []
for dim in DIMS:
    print(f"\n[dim={dim}] Building {POP} agents...", flush=True)
    t0 = time.perf_counter()
    table = FluxVectorTable(dim=dim, bit_width=4)
    rng = np.random.default_rng(42)
    for i in range(POP):
        v = rng.standard_normal(dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        table.add(AgentVector(agent_id=i, vector=v.tolist(), fitness=rng.random()))
    build = time.perf_counter() - t0
    print(f"[dim={dim}] Build: {build:.2f}s", flush=True)

    # Warmup
    q = rng.standard_normal(dim).astype(np.float32)
    q /= np.linalg.norm(q) + 1e-8
    for _ in range(10):
        table.search(q.tolist(), k=K)

    # Benchmark
    latencies = []
    for _ in range(QUERIES):
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-8
        t0 = time.perf_counter()
        table.search(q.tolist(), k=K)
        latencies.append(time.perf_counter() - t0)

    avg_ms = sum(latencies) / len(latencies) * 1000
    p99_ms = sorted(latencies)[int(len(latencies)*0.99)] * 1000
    mem_mb = (POP * dim * 0.5 + POP * 40) / (1024*1024)
    naive_mb = (POP * dim * 4) / (1024*1024)

    print(f"[dim={dim}] Avg={avg_ms:.3f}ms P99={p99_ms:.3f}ms Mem={mem_mb:.1f}MB ({naive_mb/mem_mb:.1f}x)", flush=True)

    results.append({
        "dim": dim,
        "build_s": build,
        "avg_ms": avg_ms,
        "p99_ms": p99_ms,
        "mem_mb": mem_mb,
        "compress": naive_mb / mem_mb,
    })

print("\n=== SUMMARY ===")
print(f"{'Dim':>5} {'Build':>7} {'Avg ms':>8} {'P99 ms':>8} {'Mem MB':>8} {'Compress':>8}")
for r in results:
    print(f"{r['dim']:>5} {r['build_s']:>6.1f}s {r['avg_ms']:>7.2f} {r['p99_ms']:>7.2f} {r['mem_mb']:>7.1f} {r['compress']:>7.1f}x")

best = min(results, key=lambda r: r['avg_ms'] + r['mem_mb'] * 0.1)
print(f"\nRecommended: dim={best['dim']}")

import json
from pathlib import Path
Path("/tmp/sunset-ecosystem/benchmarks/turbovec_real_results.json").write_text(json.dumps(results, indent=2))
print("Saved to benchmarks/turbovec_real_results.json")
