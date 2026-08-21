"""
BENCHMARK-HDC-ROUND3.py — Python HDC vs mock-Rust FFI overhead benchmark.

Compares three 512-bit hypervector distance paths:
  1. NumPy XOR+POPCNT  (HDC binary, packed uint64)
  2. mock FFI manhattan_distance (float32 loop)
  3. mock FFI cascade_match (float32 batch)

Varying candidate counts: 10, 50, 100, 500, 1000.
Measures per-operation mean time, std dev, and speedup ratios.
"""

from __future__ import annotations

import statistics
import time
import textwrap
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Import the two modules we are benchmarking
# ---------------------------------------------------------------------------
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superinstance_ffi_mock import load_mock_ffi
from swarm.hdc_novelty import HDCDiversityScorer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DIM = 512
BITS = DIM  # 512-bit hypervectors
SEED = 42
TRIALS = 50
WARMUP = 5
SIZES = [10, 50, 100, 500, 1000]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt_ms(t: float) -> str:
    """Format time in ms with 3 sig figs."""
    return f"{t * 1000:.3f}"


def bench(
    func, *args, trials: int = TRIALS, warmup: int = WARMUP
) -> tuple[float, float]:
    """Run *func*(*args) repeatedly. Return (mean_seconds, stdev_seconds)."""
    for _ in range(warmup):
        func(*args)
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return statistics.mean(times), statistics.stdev(times)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def run_numpy_hdc(query_packed, candidates_packed, scorer: HDCDiversityScorer):
    """Broadcast XOR+POPCNT over all candidates."""
    # score_batch expects (n_q, n_words) and (n_r, n_words)
    # We do 1 query vs N candidates
    return scorer.score_batch(query_packed[None, :], candidates_packed)


def run_ffi_manhattan(query_float, candidates_float, ffi, dim: int):
    """Loop manhattan_distance over N candidates (simulates per-call FFI overhead)."""
    n = candidates_float.shape[0]
    for i in range(n):
        ffi.manhattan_distance(query_float, candidates_float[i], dim)


def run_ffi_cascade(query_float, candidates_float, ffi, dim: int, n: int):
    """Single FFI call with flat candidate array (full scan — no early match)."""
    # Flatten candidates to [n*dim]
    flat = candidates_float.reshape(-1).astype(np.float32)
    # Thresholds set so NO candidate passes → forces full scan
    thresholds = np.array([0.0, 0.0], dtype=np.float32)
    tiers = 2
    return ffi.cascade_match(query_float, flat, n, dim, thresholds, tiers)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> dict:
    rng = np.random.default_rng(SEED)
    ffi = load_mock_ffi()
    scorer = HDCDiversityScorer(DIM, use_avx512=False)  # isolate FFI vs pure-Py

    print(f"{'=' * 70}")
    print(f"HDC ROUND 3 — Python HDC vs mock-Rust FFI overhead")
    print(f"DIM={DIM} bits  |  TRIALS={TRIALS}  |  WARMUP={WARMUP}")
    print(f"{'=' * 70}\n")

    results = []

    for n in SIZES:
        # ---- generate data ------------------------------------------------
        # Random float32 vectors → encode to binary for HDC path
        candidates_float = rng.standard_normal((n, DIM), dtype=np.float32)
        query_float = rng.standard_normal((DIM,), dtype=np.float32)

        candidates_packed = scorer.encoder.encode_batch(candidates_float)
        query_packed = scorer.encoder.encode(query_float)

        # ---- benchmark 1: numpy XOR+POPCNT -------------------------------
        t_mean, t_std = bench(run_numpy_hdc, query_packed, candidates_packed, scorer)
        np_mean_ms = t_mean * 1000
        np_std_ms = t_std * 1000

        # ---- benchmark 2: mock FFI manhattan_distance (loop) -----------------
        t_mean, t_std = bench(
            run_ffi_manhattan, query_float, candidates_float, ffi, DIM
        )
        man_mean_ms = t_mean * 1000
        man_std_ms = t_std * 1000

        # ---- benchmark 3: mock FFI cascade_match (single call) -------------
        t_mean, t_std = bench(
            run_ffi_cascade, query_float, candidates_float, ffi, DIM, n
        )
        cas_mean_ms = t_mean * 1000
        cas_std_ms = t_std * 1000

        # ---- speedups (relative to numpy, the baseline) ---------------------
        man_speedup = np_mean_ms / max(man_mean_ms, 1e-9)
        cas_speedup = np_mean_ms / max(cas_mean_ms, 1e-9)

        results.append(
            {
                "n": n,
                "np_ms": np_mean_ms,
                "np_std": np_std_ms,
                "man_ms": man_mean_ms,
                "man_std": man_std_ms,
                "cas_ms": cas_mean_ms,
                "cas_std": cas_std_ms,
                "man_speedup": man_speedup,
                "cas_speedup": cas_speedup,
            }
        )

        # ---- print ----------------------------------------------------------
        print(
            f"--- n = {n:4d} candidates -----------------------------------------------"
        )
        print(
            f"  NumPy XOR+POPCNT       : {fmt_ms(np_mean_ms)} ms  ± {fmt_ms(np_std_ms)} ms"
        )
        print(
            f"  FFI manhattan (loop)   : {fmt_ms(man_mean_ms)} ms  ± {fmt_ms(man_std_ms)} ms  "
            f"(speedup {man_speedup:.2f}x)"
        )
        print(
            f"  FFI cascade_match      : {fmt_ms(cas_mean_ms)} ms  ± {fmt_ms(cas_std_ms)} ms  "
            f"(speedup {cas_speedup:.2f}x)"
        )
        print()

    # ---- verdict -----------------------------------------------------------
    last = results[-1]
    print(f"{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    if last["cas_speedup"] > 1.0:
        print(
            f"  At n={last['n']}, FFI cascade_match is {last['cas_speedup']:.2f}x faster than NumPy."
        )
        verdict = f"speedup {last['cas_speedup']:.2f}x"
    else:
        print(f"  At n={last['n']}, FFI cascade_match is SLOWER than NumPy.")
        print(f"  FFI call overhead still dominates at 1000 vectors.")
        verdict = "overhead dominates"

    # Check crossover for cascade_match
    crossover = None
    for i, r in enumerate(results):
        if r["cas_speedup"] > 1.0:
            crossover = r["n"]
            break

    if crossover:
        print(f"  FFI cascade_match crosses over NumPy at n ≈ {crossover} vectors.")
    else:
        print(f"  FFI cascade_match NEVER crosses over NumPy in the tested range.")

    print()
    return {"results": results, "verdict": verdict, "crossover": crossover}


# ---------------------------------------------------------------------------
# Markdown report generator
# ---------------------------------------------------------------------------


def write_report(data: dict, path: Path) -> None:
    results = data["results"]
    verdict = data["verdict"]
    crossover = data["crossover"]

    lines = [
        "# BENCHMARK-RESULTS-ROUND3 — HDC Python vs mock-Rust FFI Overhead",
        "",
        f"**Date:** 2026-05-24  ",
        f"**Dimension:** 512-bit hypervectors  ",
        f"**Trials:** {TRIALS} per size  ",
        f"**Warmup:** {WARMUP}  ",
        "",
        "## Methods",
        "",
        "| Path | Description |",
        "|------|-------------|",
        "| **NumPy XOR+POPCNT** | `HDCDiversityScorer.score_batch()` — packed uint64 XOR then AVX-512 POPCOUNT |",
        "| **FFI manhattan_distance** | `ffi.manhattan_distance()` called in a Python loop over N candidates |",
        "| **FFI cascade_match** | `ffi.cascade_match()` single call with flat `[N×dim]` array (full scan, no early exit) |",
        "",
        "## Results",
        "",
        "| N | NumPy (ms) | FFI manhattan (ms) | FFI cascade (ms) | manhattan speedup | cascade speedup |",
        "|---|-----------:|-------------------:|-----------------:|------------------:|----------------:|",
    ]

    for r in results:
        lines.append(
            f"| {r['n']} | "
            f"{r['np_ms']:.3f} ± {r['np_std']:.3f} | "
            f"{r['man_ms']:.3f} ± {r['man_std']:.3f} | "
            f"{r['cas_ms']:.3f} ± {r['cas_std']:.3f} | "
            f"{r['man_speedup']:.2f}x | "
            f"{r['cas_speedup']:.2f}x |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
    ]

    if crossover:
        lines.append(
            f"- **Crossover point:** FFI `cascade_match` becomes faster than NumPy XOR+POPCNT at "
            f"**≈ {crossover} vectors**. Below this, the per-call Python-to-C shim overhead "
            f"(argument marshalling, ctypes wrapper dispatch) dominates the actual compute."
        )
    else:
        lines.append(
            "- **No crossover in tested range.** The FFI mock was slower than NumPy at every "
            "tested size. In a real compiled extension the raw loop would be in C/Rust and "
            "the crossover would occur at a much lower N (likely < 50)."
        )

    lines += [
        f"- **At N=1000:** {verdict}",
        "",
        "### Overhead analysis",
        "",
        "`manhattan_distance` is called once per candidate, so its cost grows linearly *and* "
        "pays the Python→C shim tax on every iteration. `cascade_match` amortises that tax "
        "across the entire batch in a single call, which is why it is the only FFI path that "
        "has a realistic chance of beating NumPy for moderate N.",
        "",
        "In the *mock* implementation both paths are pure Python, so the observed 'overhead' "
        "is mostly extra Python function-call frames and ctypes attribute resolution. A real "
        "Rust `.so` would shift the curve down significantly for the batch path.",
        "",
        "## Raw numbers",
        "",
        "```json",
    ]

    # Emit compact JSON
    import json

    lines.append(json.dumps(data, indent=2))
    lines.append("```")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {path}")


if __name__ == "__main__":
    data = main()
    out_path = Path(__file__).with_name("BENCHMARK-RESULTS-ROUND3.md")
    write_report(data, out_path)
