#!/usr/bin/env python3
"""Minimal turbovec benchmark — 1K agents, 2 dims, to avoid OOM kills."""

import sys
import time

sys.path.insert(0, "/tmp/sunset-ecosystem")

import numpy as np
from swarm.vector_table import FluxVectorTable, AgentVector

for dim in [128, 256]:
    print(f"\n[dim={dim}] Building 1000 agents...", flush=True)
    table = FluxVectorTable(dim=dim, bit_width=4)
    rng = np.random.default_rng(42)
    for i in range(1000):
        v = rng.standard_normal(dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        table.add(AgentVector(agent_id=i, vector=v.tolist(), fitness=rng.random()))
    print(f"[dim={dim}] Build complete, table={len(table)}", flush=True)

    # 10 searches
    latencies = []
    for _ in range(10):
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-8
        t0 = time.perf_counter()
        table.search(q.tolist(), k=5)
        latencies.append(time.perf_counter() - t0)
    avg_ms = sum(latencies) / len(latencies) * 1000
    print(f"[dim={dim}] Avg search: {avg_ms:.3f}ms", flush=True)

print("\n✅ turbovec benchmark complete (1K agents)")
