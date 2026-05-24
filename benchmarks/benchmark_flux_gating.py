"""Benchmark PythonFluxFallback check throughput."""

import time
import numpy as np
from swarm.flux_gating import FluxGatingConfig, PythonFluxFallback


def benchmark():
    config = FluxGatingConfig(
        weight_bounds=(0.0, 5.0),
        chaos_limit=0.5,
        thermal_budget_limit=0.95,
    )
    checker = PythonFluxFallback(config)

    # Single check benchmark
    plan = {"weights": 2.5, "chaos": 0.1, "thermal_headroom": 0.5}
    n = 1000
    t0 = time.perf_counter()
    for i in range(n):
        checker.check_candidate(parent_idx=i % 250, mutation_plan=plan)
    t1 = time.perf_counter()
    single_dur = (t1 - t0) / n
    single_rate = 1.0 / single_dur

    # Batch benchmark (amortized)
    batch_size = 100
    candidates = [
        (i % 250, {"weights": 2.5 + (i % 10), "chaos": 0.1, "thermal_headroom": 0.5})
        for i in range(batch_size)
    ]
    t0 = time.perf_counter()
    for _ in range(n):
        checker.check_batch(candidates)
    t1 = time.perf_counter()
    batch_dur = (t1 - t0) / (n * batch_size)
    batch_rate = 1.0 / batch_dur

    print(f"PythonFluxFallback benchmark")
    print(f"  Single check: {single_dur*1e3:.3f} ms  ({single_rate:.0f} checks/sec)")
    print(f"  Per-check (batch): {batch_dur*1e3:.3f} ms  ({batch_rate:.0f} checks/sec)")


if __name__ == "__main__":
    benchmark()
