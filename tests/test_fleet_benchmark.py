"""Tests for FleetBenchmark — performance benchmarking suite.

Reference: fleet/fleet_benchmark.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from fleet.fleet_benchmark import BenchmarkResult, BenchmarkSuite, FleetBenchmark


class TestBenchmarkResult:
    def test_fields(self) -> None:
        result = BenchmarkResult(
            name="test",
            iterations=100,
            total_ms=1000.0,
            min_ms=5.0,
            max_ms=20.0,
            mean_ms=10.0,
            median_ms=9.5,
            p95_ms=18.0,
            p99_ms=19.5,
            stdev_ms=2.0,
            throughput_ops_per_sec=100.0,
            memory_peak_mb=1.0,
            memory_current_mb=0.5,
        )
        assert result.name == "test"
        assert result.iterations == 100
        assert result.ops_per_sec == 100.0

    def test_to_dict(self) -> None:
        result = BenchmarkResult(
            name="test",
            iterations=100,
            total_ms=1000.0,
            min_ms=5.0,
            max_ms=20.0,
            mean_ms=10.0,
            median_ms=9.5,
            p95_ms=18.0,
            p99_ms=19.5,
            stdev_ms=2.0,
            throughput_ops_per_sec=100.0,
            memory_peak_mb=1.0,
            memory_current_mb=0.5,
            metadata={"key": "val"},
        )
        d = result.to_dict()
        assert d["name"] == "test"
        assert d["mean_ms"] == 10.0
        assert d["key"] == "val"


class TestBenchmarkSuite:
    def test_empty(self) -> None:
        suite = BenchmarkSuite(suite_name="empty")
        assert suite.suite_name == "empty"
        assert len(suite.results) == 0

    def test_summary(self) -> None:
        suite = BenchmarkSuite(suite_name="test")
        suite.results.append(
            BenchmarkResult(
                name="fast",
                iterations=10,
                total_ms=10.0,
                min_ms=0.5,
                max_ms=2.0,
                mean_ms=1.0,
                median_ms=1.0,
                p95_ms=1.5,
                p99_ms=1.9,
                stdev_ms=0.3,
                throughput_ops_per_sec=1000.0,
                memory_peak_mb=0.1,
                memory_current_mb=0.05,
            )
        )
        suite.results.append(
            BenchmarkResult(
                name="slow",
                iterations=10,
                total_ms=100.0,
                min_ms=5.0,
                max_ms=20.0,
                mean_ms=10.0,
                median_ms=10.0,
                p95_ms=18.0,
                p99_ms=19.5,
                stdev_ms=2.0,
                throughput_ops_per_sec=100.0,
                memory_peak_mb=1.0,
                memory_current_mb=0.5,
            )
        )
        summary = suite.summary()
        assert summary["benchmark_count"] == 2
        assert summary["fastest"] == "fast"
        assert summary["slowest"] == "slow"

    def test_to_dict(self) -> None:
        suite = BenchmarkSuite(suite_name="test")
        suite.results.append(
            BenchmarkResult(
                name="fast",
                iterations=10,
                total_ms=10.0,
                min_ms=0.5,
                max_ms=2.0,
                mean_ms=1.0,
                median_ms=1.0,
                p95_ms=1.5,
                p99_ms=1.9,
                stdev_ms=0.3,
                throughput_ops_per_sec=1000.0,
                memory_peak_mb=0.1,
                memory_current_mb=0.05,
            )
        )
        d = suite.to_dict()
        assert d["suite_name"] == "test"
        assert len(d["results"]) == 1


class TestFleetBenchmark:
    def test_init(self) -> None:
        bm = FleetBenchmark()
        assert bm.workspace.exists()

    def test_benchmark_simple(self) -> None:
        bm = FleetBenchmark()
        counter = [0]

        def fn():
            counter[0] += 1
            time.sleep(0.001)  # 1ms sleep

        result = bm.benchmark("simple_counter", fn, iterations=10, warmup=2)
        assert result.name == "simple_counter"
        assert result.iterations == 10
        assert counter[0] >= 12  # 10 + 2 warmup
        assert result.mean_ms > 0
        assert result.median_ms > 0
        assert result.p95_ms >= result.median_ms
        assert result.p99_ms >= result.p95_ms
        assert result.throughput_ops_per_sec > 0
        assert result.memory_peak_mb >= 0

    def test_benchmark_beat(self) -> None:
        bm = FleetBenchmark()
        result = bm.benchmark_beat(iterations=5)
        assert result.name == "fleet_beat"
        assert result.iterations == 5
        assert result.mean_ms > 0
        assert result.metadata["component"] == "FleetOrchestrator"

    def test_benchmark_health_check(self) -> None:
        bm = FleetBenchmark()
        result = bm.benchmark_health_check(iterations=5)
        assert result.name == "health_check"
        assert result.iterations == 5
        assert result.metadata["operation"] == "check_fleet_health"

    def test_benchmark_harbor_report(self) -> None:
        bm = FleetBenchmark()
        result = bm.benchmark_harbor_report(iterations=5)
        assert result.name == "harbor_report"
        assert result.iterations == 5
        assert result.metadata["component"] == "Harbor"

    def test_benchmark_module_lookup(self) -> None:
        bm = FleetBenchmark()
        result = bm.benchmark_module_lookup(iterations=10)
        assert result.name == "module_lookup"
        assert result.iterations == 10

    def test_benchmark_module_stats(self) -> None:
        bm = FleetBenchmark()
        result = bm.benchmark_module_stats(iterations=10)
        assert result.name == "module_stats"
        assert result.iterations == 10

    def test_run_full_suite(self) -> None:
        bm = FleetBenchmark()
        suite = bm.run_full_suite()
        assert suite.suite_name == "fleet_full_suite"
        assert len(suite.results) == 5
        assert suite.total_duration_ms > 0
        for r in suite.results:
            assert r.iterations > 0
            assert r.mean_ms > 0

    def test_generate_markdown_report(self, tmp_path) -> None:
        bm = FleetBenchmark()
        suite = bm.run_full_suite()
        output_path = str(tmp_path / "benchmark.md")
        content = bm.generate_markdown_report(suite, output_path)
        assert "# Fleet Benchmark Report" in content
        assert "fleet_beat" in content
        assert "health_check" in content
        assert Path(output_path).exists()

    def test_generate_json_report(self, tmp_path) -> None:
        bm = FleetBenchmark()
        suite = bm.run_full_suite()
        output_path = str(tmp_path / "benchmark.json")
        content = bm.generate_json_report(suite, output_path)
        data = json.loads(content)
        assert data["suite_name"] == "fleet_full_suite"
        assert len(data["results"]) == 5
        assert Path(output_path).exists()

    def test_compare_to_baseline(self, tmp_path) -> None:
        bm = FleetBenchmark()
        suite = bm.run_full_suite()

        # Create baseline
        baseline_path = str(tmp_path / "baseline.json")
        bm.generate_json_report(suite, baseline_path)

        # Compare to itself (should be no regression)
        comparison = bm.compare_to_baseline(suite, baseline_path)
        assert comparison["suite_name"] == "fleet_full_suite"
        assert len(comparison["comparisons"]) > 0
        assert len(comparison["regressions"]) == 0  # Same baseline, no regression
        assert len(comparison["improvements"]) == 0  # Same baseline, no improvement

    def test_compare_with_regression(self, tmp_path) -> None:
        bm = FleetBenchmark()
        suite = bm.run_full_suite()

        # Create baseline with artificially fast results
        baseline = suite.to_dict()
        for r in baseline["results"]:
            r["mean_ms"] = r["mean_ms"] * 0.1  # 10x faster baseline
        baseline_path = str(tmp_path / "baseline_fast.json")
        Path(baseline_path).write_text(json.dumps(baseline))

        # Compare — current should be slower (regression)
        comparison = bm.compare_to_baseline(suite, baseline_path)
        assert len(comparison["regressions"]) > 0
        for reg in comparison["regressions"]:
            assert reg["change_pct"] > 20

    def test_suite_summary(self) -> None:
        bm = FleetBenchmark()
        suite = bm.run_full_suite()
        summary = suite.summary()
        assert summary["benchmark_count"] == 5
        assert "fastest" in summary
        assert "slowest" in summary

    def test_benchmark_metadata(self) -> None:
        bm = FleetBenchmark()

        def fn():
            pass

        result = bm.benchmark(
            "meta_test",
            fn,
            iterations=5,
            warmup=0,
            metadata={"custom": "value", "number": 42},
        )
        d = result.to_dict()
        assert d["custom"] == "value"
        assert d["number"] == 42

    def test_benchmark_statistics(self) -> None:
        bm = FleetBenchmark()

        def fn():
            time.sleep(0.001)

        result = bm.benchmark("stats_test", fn, iterations=50, warmup=5)
        assert result.min_ms <= result.mean_ms <= result.max_ms
        assert result.median_ms <= result.p95_ms <= result.p99_ms
        assert result.stdev_ms >= 0

    def test_lazy_init_orchestrator(self) -> None:
        bm = FleetBenchmark()
        assert bm._orchestrator is None
        bm._ensure_orchestrator()
        assert bm._orchestrator is not None

    def test_lazy_init_harbor(self) -> None:
        bm = FleetBenchmark()
        assert bm._harbor is None
        bm._ensure_harbor()
        assert bm._harbor is not None
