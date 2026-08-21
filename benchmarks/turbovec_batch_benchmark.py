#!/usr/bin/env python3
"""Batch-add benchmark — turbovec is more efficient with batch adds."""

import sys
import time

sys.path.insert(0, "/tmp/sunset-ecosystem")

import numpy as np
from swarm.vector_table import FluxVectorTable, AgentMeta

for dim in [128, 256]:
    print(f"\n[dim={dim}] Batch-building 1000 agents...", flush=True)
    table = FluxVectorTable(dim=dim, bit_width=4)
    rng = np.random.default_rng(42)

    # Batch add all at once
    ids = np.arange(1000, dtype=np.uint64)
    vecs = np.random.randn(1000, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8

    t0 = time.perf_counter()
    table._index.add_with_ids(vecs, ids)
    for i in range(1000):
        table._meta[int(i)] = AgentMeta(fitness=float(rng.random()))
    build = time.perf_counter() - t0
    print(f"[dim={dim}] Batch build: {build:.3f}s", flush=True)

    # Search
    latencies = []
    for _ in range(10):
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-8
        t0 = time.perf_counter()
        table.search(q.tolist(), k=5)
        latencies.append(time.perf_counter() - t0)
    avg_ms = sum(latencies) / len(latencies) * 1000
    print(f"[dim={dim}] Avg search: {avg_ms:.3f}ms", flush=True)

print("\n✅ Batch benchmark complete")
