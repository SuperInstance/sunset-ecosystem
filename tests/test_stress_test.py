"""Tests for stress_test — CPU benchmarks, memory bandwidth, and report aggregation.

Focuses on CPU-only paths that don't require CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest

from ethos.stress_test import (
    DeviceBenchmark,
    MatrixBenchmark,
    MemoryBandwidth,
    StressReport,
    TokenThroughput,
    _bench_matrix_cpu,
    _bench_memory_bandwidth_cpu,
    run_stress_test,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_matrix_benchmark_repr(self):
        mb = MatrixBenchmark(size=1024, avg_ms=50.0, gflops=100.0)
        assert "1024x1024" in repr(mb)
        assert "100.0 GFLOPS" in repr(mb)

    def test_token_throughput_repr(self):
        tt = TokenThroughput(tokens_per_second=50.0, avg_latency_ms=20.0)
        assert "50.0 tok/s" in repr(tt)

    def test_memory_bandwidth_repr(self):
        mb = MemoryBandwidth(bandwidth_gb_s=50.0, size_mb=256.0)
        assert "50.0 GB/s" in repr(mb)

    def test_device_benchmark_repr(self):
        db = DeviceBenchmark(device_name="CPU", device_type="cpu", matrix_benchmarks=[MatrixBenchmark(size=512, avg_ms=10.0, gflops=50.0)])
        assert "CPU" in repr(db)
        assert "1 mats" in repr(db)

    def test_stress_report_repr(self):
        sr = StressReport(benchmarks=[DeviceBenchmark(device_name="CPU", device_type="cpu")], max_parallel_agents=4, total_duration_s=5.0)
        assert "1 devices" in repr(sr)
        assert "max_agents=4" in repr(sr)

    def test_stress_report_best_gpu(self):
        sr = StressReport()
        assert sr.best_gpu_gflops() is None

        gpu = DeviceBenchmark(
            device_name="GPU", device_type="cuda",
            matrix_benchmarks=[MatrixBenchmark(size=1024, avg_ms=10.0, gflops=200.0)],
        )
        sr2 = StressReport(benchmarks=[gpu])
        assert sr2.best_gpu_gflops() == 200.0

    def test_stress_report_cpu_gflops(self):
        cpu = DeviceBenchmark(
            device_name="CPU", device_type="cpu",
            matrix_benchmarks=[MatrixBenchmark(size=1024, avg_ms=100.0, gflops=50.0)],
        )
        sr = StressReport(benchmarks=[cpu])
        assert sr.cpu_gflops() == 50.0

    def test_stress_report_no_cpu_gflops(self):
        sr = StressReport()
        assert sr.cpu_gflops() is None


# ---------------------------------------------------------------------------
# CPU matrix benchmark
# ---------------------------------------------------------------------------

class TestBenchMatrixCpu:
    def test_small_sizes(self):
        results = _bench_matrix_cpu(sizes=[64, 128], warmup=1, runs=2)
        assert len(results) == 2
        for r in results:
            assert r.gflops > 0
            assert r.avg_ms > 0


# ---------------------------------------------------------------------------
# CPU memory bandwidth
# ---------------------------------------------------------------------------

class TestBenchMemoryBandwidth:
    def test_basic(self):
        result = _bench_memory_bandwidth_cpu(size_mb=64.0, runs=2)
        assert result is not None
        assert result.bandwidth_gb_s > 0
        assert result.size_mb == 64.0


# ---------------------------------------------------------------------------
# run_stress_test
# ---------------------------------------------------------------------------

class TestRunStressTest:
    def test_runs_quick(self):
        report = run_stress_test(quick=True)
        assert isinstance(report, StressReport)
        assert report.total_duration_s > 0
        assert len(report.benchmarks) >= 1  # at least CPU

    def test_max_parallel_agents(self):
        report = run_stress_test(quick=True)
        assert report.max_parallel_agents >= 1

    def test_cpu_benchmark_present(self):
        report = run_stress_test(quick=True)
        cpu_benches = [b for b in report.benchmarks if b.device_type == "cpu"]
        assert len(cpu_benches) >= 1

    def test_matrix_results_present(self):
        report = run_stress_test(quick=True)
        for b in report.benchmarks:
            if b.matrix_benchmarks:
                assert len(b.matrix_benchmarks) >= 1

    def test_memory_bandwidth_present(self):
        report = run_stress_test(quick=True)
        for b in report.benchmarks:
            if b.memory_bandwidth is not None:
                assert b.memory_bandwidth.bandwidth_gb_s > 0
