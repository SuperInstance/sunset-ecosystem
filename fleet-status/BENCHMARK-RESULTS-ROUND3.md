# BENCHMARK-RESULTS-ROUND3 — HDC Python vs mock-Rust FFI Overhead

**Date:** 2026-05-24  
**Dimension:** 512-bit hypervectors  
**Trials:** 50 per size  
**Warmup:** 5  

## Methods

| Path | Description |
|------|-------------|
| **NumPy XOR+POPCNT** | `HDCDiversityScorer.score_batch()` — packed uint64 XOR then AVX-512 POPCOUNT |
| **FFI manhattan_distance** | `ffi.manhattan_distance()` called in a Python loop over N candidates |
| **FFI cascade_match** | `ffi.cascade_match()` single call with flat `[N×dim]` array (full scan, no early exit) |

## Results

| N | NumPy (ms) | FFI manhattan (ms) | FFI cascade (ms) | manhattan speedup | cascade speedup |
|---|-----------:|-------------------:|-----------------:|------------------:|----------------:|
| 10 | 0.032 ± 0.005 | 0.082 ± 0.009 | 0.106 ± 0.014 | 0.39x | 0.30x |
| 50 | 0.061 ± 0.020 | 0.419 ± 0.039 | 0.504 ± 0.068 | 0.15x | 0.12x |
| 100 | 0.096 ± 0.020 | 0.856 ± 0.067 | 1.209 ± 0.348 | 0.11x | 0.08x |
| 500 | 0.514 ± 0.281 | 4.655 ± 0.686 | 6.705 ± 1.536 | 0.11x | 0.08x |
| 1000 | 1.516 ± 0.686 | 9.508 ± 1.374 | 13.060 ± 2.117 | 0.16x | 0.12x |

## Interpretation

- **No crossover in tested range.** The FFI mock was slower than NumPy at every tested size. In a real compiled extension the raw loop would be in C/Rust and the crossover would occur at a much lower N (likely < 50).
- **At N=1000:** overhead dominates

### Overhead analysis

`manhattan_distance` is called once per candidate, so its cost grows linearly *and* pays the Python→C shim tax on every iteration. `cascade_match` amortises that tax across the entire batch in a single call, which is why it is the only FFI path that has a realistic chance of beating NumPy for moderate N.

In the *mock* implementation both paths are pure Python, so the observed 'overhead' is mostly extra Python function-call frames and ctypes attribute resolution. A real Rust `.so` would shift the curve down significantly for the batch path.

## Raw numbers

```json
{
  "results": [
    {
      "n": 10,
      "np_ms": 0.03198697231709957,
      "np_std": 0.004960727744845993,
      "man_ms": 0.082084396854043,
      "man_std": 0.008692919086135174,
      "cas_ms": 0.10551447048783302,
      "cas_std": 0.014053915024995983,
      "man_speedup": 0.38968395386977955,
      "cas_speedup": 0.30315246969644816
    },
    {
      "n": 50,
      "np_ms": 0.06134777329862118,
      "np_std": 0.01950316326703705,
      "man_ms": 0.41873897425830364,
      "man_std": 0.03897091145437047,
      "cas_ms": 0.5039151944220066,
      "cas_std": 0.06798869121892534,
      "man_speedup": 0.14650600271275,
      "cas_speedup": 0.12174225738318409
    },
    {
      "n": 100,
      "np_ms": 0.09566265158355236,
      "np_std": 0.020100567773549002,
      "man_ms": 0.8564431499689817,
      "man_std": 0.06749625932627043,
      "cas_ms": 1.2086710799485445,
      "cas_std": 0.34833217765410585,
      "man_speedup": 0.11169760840169837,
      "cas_speedup": 0.07914696824517793
    },
    {
      "n": 500,
      "np_ms": 0.5144960712641478,
      "np_std": 0.28060163285479567,
      "man_ms": 4.6547112707048655,
      "man_std": 0.6857977237482711,
      "cas_ms": 6.704798014834523,
      "cas_std": 1.5364074711712064,
      "man_speedup": 0.11053232764451087,
      "cas_speedup": 0.07673550644267181
    },
    {
      "n": 1000,
      "np_ms": 1.5158907324075699,
      "np_std": 0.6858554863347732,
      "man_ms": 9.508239915594459,
      "man_std": 1.3741621419486747,
      "cas_ms": 13.0604103859514,
      "cas_std": 2.1165065162378163,
      "man_speedup": 0.15942916311160368,
      "cas_speedup": 0.11606761867438388
    }
  ],
  "verdict": "overhead dominates",
  "crossover": null
}
```
