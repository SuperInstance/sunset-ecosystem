#!/usr/bin/env python3
"""Benchmark CUDA Einsum Kernel across room counts and dimensions.

Outputs CSV to stdout. Usage::

    python scripts/benchmark_cuda.py > benchmarks/cuda_einsum_$(date +%Y%m%d).csv

Room counts: [50, 100, 250, 500, 1000]
Dimensions:  [32, 64, 128]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# Add repo root to path if running standalone
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from sunset.cuda_kernels import CudaEinsumKernel


def main():
    room_counts = [50, 100, 250, 500, 1000]
    dimensions = [32, 64, 128]
    n_iterations = 100

    kernel = CudaEinsumKernel(n_sms=20, shared_mem_kb=16)

    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "n_rooms",
            "d_latent",
            "n_selected",
            "n_iterations",
            "ms_per_call_cuda",
            "ms_per_call_np",
            "speedup",
            "cuda_available",
            "shape_ok",
            "max_diff",
        ]
    )

    for n_rooms in room_counts:
        for d_latent in dimensions:
            bench = kernel.benchmark(
                n_rooms=n_rooms,
                d_latent=d_latent,
                n_iterations=n_iterations,
            )
            writer.writerow(
                [
                    bench["n_rooms"],
                    bench["d_latent"],
                    bench["n_selected"],
                    bench["n_iterations"],
                    bench["ms_per_call"],
                    bench["ms_per_call_np"],
                    bench["speedup"],
                    bench["cuda_available"],
                    bench["shape_ok"],
                    bench["max_diff"],
                ]
            )
            # Flush so partial output is visible even if interrupted
            sys.stdout.flush()


if __name__ == "__main__":
    main()
