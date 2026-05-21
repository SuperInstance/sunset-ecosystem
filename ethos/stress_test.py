"""Micro-benchmarks to stress-test available compute devices."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = ["StressReport", "DeviceBenchmark", "run_stress_test"]


@dataclass
class MatrixBenchmark:
    """Results of matrix multiply benchmark at a given size."""

    size: int
    avg_ms: float
    gflops: float

    def __repr__(self) -> str:
        return f"MatMul({self.size}x{self.size}: {self.gflops:.1f} GFLOPS, {self.avg_ms:.2f}ms)"


@dataclass
class TokenThroughput:
    """Token generation throughput estimate."""

    tokens_per_second: float
    avg_latency_ms: float

    def __repr__(self) -> str:
        return f"TokenTP({self.tokens_per_second:.1f} tok/s, {self.avg_latency_ms:.1f}ms)"


@dataclass
class MemoryBandwidth:
    """Memory bandwidth measurement."""

    bandwidth_gb_s: float
    size_mb: float

    def __repr__(self) -> str:
        return f"MemBW({self.bandwidth_gb_s:.1f} GB/s, {self.size_mb:.0f}MB)"


@dataclass
class DeviceBenchmark:
    """Full benchmark suite for a single device."""

    device_name: str
    device_type: str  # "cuda", "cpu", "igpu"
    device_index: int = 0
    matrix_benchmarks: List[MatrixBenchmark] = field(default_factory=list)
    token_throughput: Optional[TokenThroughput] = None
    memory_bandwidth: Optional[MemoryBandwidth] = None

    def __repr__(self) -> str:
        return f"DeviceBenchmark({self.device_name!r}, {self.device_type}, {len(self.matrix_benchmarks)} mats)"


@dataclass
class StressReport:
    """Complete stress test report across all devices."""

    benchmarks: List[DeviceBenchmark] = field(default_factory=list)
    max_parallel_agents: int = 1
    total_duration_s: float = 0.0

    def __repr__(self) -> str:
        return (
            f"StressReport({len(self.benchmarks)} devices, "
            f"max_agents={self.max_parallel_agents}, {self.total_duration_s:.1f}s)"
        )

    def best_gpu_gflops(self) -> Optional[float]:
        """Return best GPU GFLOPS from 1024x1024 matmul, or None."""
        for b in self.benchmarks:
            if b.device_type == "cuda":
                for m in b.matrix_benchmarks:
                    if m.size == 1024:
                        return m.gflops
        return None

    def cpu_gflops(self) -> Optional[float]:
        """Return CPU GFLOPS from 1024x1024 matmul."""
        for b in self.benchmarks:
            if b.device_type == "cpu":
                for m in b.matrix_benchmarks:
                    if m.size == 1024:
                        return m.gflops
        return None


def _bench_matrix_cpu(sizes: List[int], warmup: int = 2, runs: int = 5) -> List[MatrixBenchmark]:
    """Matrix multiply benchmark using numpy (CPU)."""
    try:
        import numpy as np
    except ImportError:
        return []

    results: List[MatrixBenchmark] = []
    for size in sizes:
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        # Warmup
        for _ in range(warmup):
            _ = a @ b
        times: List[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = a @ b
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg_s = sum(times) / len(times)
        # 2*size^3 FLOPs for matmul
        gflops = (2 * size ** 3) / (avg_s * 1e9)
        results.append(
            MatrixBenchmark(size=size, avg_ms=avg_s * 1000, gflops=gflops)
        )
    return results


def _bench_matrix_cuda(
    sizes: List[int], device_index: int = 0, warmup: int = 5, runs: int = 10
) -> List[MatrixBenchmark]:
    """Matrix multiply benchmark using torch on CUDA."""
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        return []

    if not torch.cuda.is_available():
        return []

    results: List[MatrixBenchmark] = []
    dev = torch.device(f"cuda:{device_index}")

    for size in sizes:
        a = torch.randn(size, size, dtype=torch.float32, device=dev)
        b = torch.randn(size, size, dtype=torch.float32, device=dev)
        # Warmup
        for _ in range(warmup):
            _ = a @ b
        torch.cuda.synchronize(dev)

        times: List[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = a @ b
            torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg_s = sum(times) / len(times)
        gflops = (2 * size ** 3) / (avg_s * 1e9)
        results.append(
            MatrixBenchmark(size=size, avg_ms=avg_s * 1000, gflops=gflops)
        )
    return results


def _bench_memory_bandwidth_cpu(size_mb: float = 256.0, runs: int = 5) -> Optional[MemoryBandwidth]:
    """Simple memory bandwidth test on CPU."""
    try:
        import numpy as np
    except ImportError:
        return None

    n = int(size_mb * 1024 * 1024 / 4)  # float32 elements
    a = np.ones(n, dtype=np.float32)
    b = np.zeros(n, dtype=np.float32)

    times: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        b[:] = a[:]
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_s = sum(times) / len(times)
    # 2 copies of size_mb
    gb_s = (2 * size_mb) / (avg_s * 1024)
    return MemoryBandwidth(bandwidth_gb_s=gb_s, size_mb=size_mb)


def _bench_memory_bandwidth_cuda(
    size_mb: float = 256.0, device_index: int = 0, runs: int = 10
) -> Optional[MemoryBandwidth]:
    """Memory bandwidth test on CUDA device."""
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    dev = torch.device(f"cuda:{device_index}")
    n = int(size_mb * 1024 * 1024 / 4)
    a = torch.ones(n, dtype=torch.float32, device=dev)
    b = torch.zeros(n, dtype=torch.float32, device=dev)

    times: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        b.copy_(a)
        torch.cuda.synchronize(dev)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_s = sum(times) / len(times)
    gb_s = (2 * size_mb) / (avg_s * 1024)
    return MemoryBandwidth(bandwidth_gb_s=gb_s, size_mb=size_mb)


def _estimate_token_throughput(gflops_1024: float, device_type: str) -> TokenThroughput:
    """Estimate token generation throughput from GFLOPS.

    Rough model: a 7B model needs ~14 GFLOPs per token (2 * params).
    Adjust heuristically.
    """
    # Assume inference workload targeting ~7B model scale
    flops_per_token = 14e9  # ~7B params * 2
    if device_type == "cpu":
        flops_per_token *= 0.5  # CPUs are less efficient for this
    tps = gflops_1024 * 1e9 / flops_per_token if flops_per_token > 0 else 0
    latency = 1000 / tps if tps > 0 else float("inf")
    return TokenThroughput(tokens_per_second=tps, avg_latency_ms=latency)


def _estimate_parallel_agents(report: "StressReport") -> int:
    """Estimate max parallel agents from benchmark data."""
    # Heuristic: base 1 agent per CPU core (clamped), plus 1 per GPU
    import os

    base = max(1, min(os.cpu_count() or 4, 8))
    gpu_count = sum(1 for b in report.benchmarks if b.device_type == "cuda")
    return base + gpu_count * 2


# Sizes to benchmark
_DEFAULT_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096]
_QUICK_SIZES = [128, 512, 1024, 2048]


def run_stress_test(
    cuda_device_indices: Optional[List[int]] = None,
    sizes: Optional[List[int]] = None,
    quick: bool = False,
) -> StressReport:
    """Run micro-benchmarks across available compute devices.

    Benchmarks matrix multiply at various sizes, estimates token generation
    throughput, and measures memory bandwidth. Returns a full
    :class:`StressReport`.

    Args:
        cuda_device_indices: Specific CUDA devices to test. None = all.
        sizes: Matrix sizes to benchmark. None = defaults.
        quick: If True, run fewer sizes and iterations for speed.

    Returns:
        StressReport with benchmarks for each available device.
    """
    t_start = time.perf_counter()
    bench_sizes = sizes or (_QUICK_SIZES if quick else _DEFAULT_SIZES)
    report = StressReport()
    import os

    # CPU benchmark
    cpu_mats = _bench_matrix_cpu(bench_sizes)
    cpu_bench = DeviceBenchmark(
        device_name=os.cpu_count() and f"CPU ({os.cpu_count()} threads)" or "CPU",
        device_type="cpu",
        matrix_benchmarks=cpu_mats,
        memory_bandwidth=_bench_memory_bandwidth_cpu(),
    )
    if cpu_mats:
        for m in reversed(cpu_mats):
            cpu_bench.token_throughput = _estimate_token_throughput(m.gflops, "cpu")
            break
    report.benchmarks.append(cpu_bench)

    # CUDA benchmarks
    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            indices = cuda_device_indices or list(range(torch.cuda.device_count()))
            for idx in indices:
                name = torch.cuda.get_device_name(idx)
                mats = _bench_matrix_cuda(bench_sizes, device_index=idx)
                bench = DeviceBenchmark(
                    device_name=name,
                    device_type="cuda",
                    device_index=idx,
                    matrix_benchmarks=mats,
                    memory_bandwidth=_bench_memory_bandwidth_cuda(device_index=idx),
                )
                if mats:
                    for m in reversed(mats):
                        bench.token_throughput = _estimate_token_throughput(
                            m.gflops, "cuda"
                        )
                        break
                report.benchmarks.append(bench)
    except ImportError:
        pass

    report.max_parallel_agents = _estimate_parallel_agents(report)
    report.total_duration_s = time.perf_counter() - t_start
    return report
