# Turbovec Benchmark Results — Real SIMD (with LD_PRELOAD)

**Date:** 2026-05-22 01:50 UTC
**Environment:** x86_64, OpenBLAS (libopenblas-dev), numpy 1.26.4
**Note:** turbovec wheel requires `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0` due to missing dynamic link to openblas in the wheel.

## Results: 1,000 Agents

| Dim | Build Time | Search Latency | Memory |
|---|---|---|---|
| 128 | 1.06s | 105.8ms | ~0.5MB |
| 256 | 1.06s | 82.1ms | ~0.9MB |

**Observation:** Search latency is higher than expected for 1K agents. Likely because `FluxVectorTable.search()` builds allowlists and metadata lookups per query, and the turbovec index may not have been `prepare()`'d. With `prepare()` and without the Python overhead, turbovec SIMD should achieve sub-millisecond latencies.

## Comparison: Numpy Brute-Force Fallback

| Dim | Agents | Latency | Source |
|---|---|---|---|
| 128 | 10,000 | 16ms | `benchmarks/dimension_study_numpy_fallback.py` |
| 256 | 10,000 | 31ms | `benchmarks/dimension_study_numpy_fallback.py` |

The numpy fallback was faster at 10K agents because it had no per-query metadata overhead. The turbovec path needs optimization in the `FluxVectorTable.search()` method to be competitive.

## Recommendation

1. **Call `index.prepare()` after bulk loading** to build the search structure
2. **Batch metadata filtering** instead of per-query allowlist building
3. **Profile `FluxVectorTable.search()`** to find the Python overhead bottleneck

## LD_PRELOAD Fix

Add to systemd service or shell wrapper:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0
python3 -m swarm.breeder_daemon
```

Or patch the turbovec wheel to link libopenblas dynamically.
