"""FleetBenchmark — performance benchmarking suite for fleet modules.

Measures execution time, throughput, and memory usage for all fleet modules.
Generates benchmark reports with percentile statistics and profiling data.

Reference
---------
- Inspired by Google's Benchmark library and pytest-benchmark patterns
- Uses time.perf_counter() for high-resolution timing
- Measures p50, p95, p99 latencies across N iterations
"""

from __future__ import annotations

__all__ = [
    "FleetBenchmark",
    "BenchmarkResult",
    "BenchmarkSuite",
]

import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fleet.fleet_orchestrator import FleetOrchestrator
from fleet.harbor import Harbor


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    iterations: int
    total_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    stdev_ms: float
    throughput_ops_per_sec: float
    memory_peak_mb: float
    memory_current_mb: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ops_per_sec(self) -> float:
        """Operations per second (alias for throughput)."""
        return self.throughput_ops_per_sec

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_ms": self.total_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "stdev_ms": self.stdev_ms,
            "throughput_ops_per_sec": self.throughput_ops_per_sec,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_current_mb": self.memory_current_mb,
            **self.metadata,
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    suite_name: str
    results: list[BenchmarkResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "timestamp": self.timestamp,
            "total_duration_ms": self.total_duration_ms,
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> dict[str, Any]:
        """Quick summary of the suite."""
        return {
            "suite_name": self.suite_name,
            "benchmark_count": len(self.results),
            "total_duration_ms": self.total_duration_ms,
            "fastest": min((r.mean_ms, r.name) for r in self.results)[1] if self.results else None,
            "slowest": max((r.mean_ms, r.name) for r in self.results)[1] if self.results else None,
        }


class FleetBenchmark:
    """Fleet benchmarking suite.

    Parameters
    ----------
    workspace : str
        Path to sunset-ecosystem workspace.
    """

    def __init__(self, workspace: str = ".") -> None:
        self.workspace = Path(workspace)
        self._orchestrator: FleetOrchestrator | None = None
        self._harbor: Harbor | None = None

    def _ensure_orchestrator(self) -> None:
        if self._orchestrator is None:
            self._orchestrator = FleetOrchestrator(workspace=str(self.workspace))
            self._orchestrator.initialize_fleet()

    def _ensure_harbor(self) -> None:
        if self._harbor is None:
            self._harbor = Harbor(str(self.workspace))
            self._harbor.bootstrap_fleet()

    def benchmark(
        self,
        name: str,
        fn: Callable[[], Any],
        iterations: int = 100,
        warmup: int = 10,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Run a single benchmark.

        Parameters
        ----------
        name : str
            Benchmark name.
        fn : Callable
            Function to benchmark.
        iterations : int
            Number of iterations.
        warmup : int
            Warmup iterations (not counted).
        metadata : dict | None
            Additional metadata.

        Returns
        -------
        BenchmarkResult
            Benchmark results.
        """
        # Warmup
        for _ in range(warmup):
            fn()

        # Start memory tracking
        tracemalloc.start()

        times: list[float] = []
        start_total = time.perf_counter()

        for _ in range(iterations):
            start = time.perf_counter()
            fn()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        total_ms = (time.perf_counter() - start_total) * 1000

        # Memory
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Statistics
        sorted_times = sorted(times)
        p95_idx = int(iterations * 0.95)
        p99_idx = int(iterations * 0.99)

        return BenchmarkResult(
            name=name,
            iterations=iterations,
            total_ms=total_ms,
            min_ms=min(times),
            max_ms=max(times),
            mean_ms=statistics.mean(times),
            median_ms=statistics.median(times),
            p95_ms=sorted_times[min(p95_idx, iterations - 1)],
            p99_ms=sorted_times[min(p99_idx, iterations - 1)],
            stdev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            throughput_ops_per_sec=(iterations / total_ms) * 1000 if total_ms > 0 else 0.0,
            memory_peak_mb=peak_mem / (1024 * 1024),
            memory_current_mb=current_mem / (1024 * 1024),
            metadata=metadata or {},
        )

    def benchmark_beat(self, iterations: int = 50) -> BenchmarkResult:
        """Benchmark fleet beat."""
        self._ensure_orchestrator()

        return self.benchmark(
            name="fleet_beat",
            fn=lambda: self._orchestrator.beat(),
            iterations=iterations,
            warmup=5,
            metadata={"component": "FleetOrchestrator", "operation": "beat"},
        )

    def benchmark_health_check(self, iterations: int = 50) -> BenchmarkResult:
        """Benchmark fleet health check."""
        self._ensure_orchestrator()

        return self.benchmark(
            name="health_check",
            fn=lambda: self._orchestrator.check_fleet_health(),
            iterations=iterations,
            warmup=5,
            metadata={"component": "FleetOrchestrator", "operation": "check_fleet_health"},
        )

    def benchmark_harbor_report(self, iterations: int = 50) -> BenchmarkResult:
        """Benchmark harbor report generation."""
        self._ensure_harbor()

        return self.benchmark(
            name="harbor_report",
            fn=lambda: self._harbor.generate_fleet_report(),
            iterations=iterations,
            warmup=5,
            metadata={"component": "Harbor", "operation": "generate_fleet_report"},
        )

    def benchmark_module_lookup(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark module lookup by name."""
        self._ensure_harbor()

        return self.benchmark(
            name="module_lookup",
            fn=lambda: self._harbor.modules.get("Gate"),
            iterations=iterations,
            warmup=10,
            metadata={"component": "Harbor", "operation": "modules.get"},
        )

    def benchmark_module_stats(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark harbor stats retrieval."""
        self._ensure_harbor()

        return self.benchmark(
            name="module_stats",
            fn=lambda: self._harbor.get_stats(),
            iterations=iterations,
            warmup=10,
            metadata={"component": "Harbor", "operation": "get_stats"},
        )

    def run_full_suite(self) -> BenchmarkSuite:
        """Run the full benchmark suite.

        Returns
        -------
        BenchmarkSuite
            Complete suite of benchmark results.
        """
        suite_start = time.perf_counter()
        suite = BenchmarkSuite(suite_name="fleet_full_suite")

        # Core operations
        suite.results.append(self.benchmark_beat(iterations=20))
        suite.results.append(self.benchmark_health_check(iterations=20))
        suite.results.append(self.benchmark_harbor_report(iterations=20))
        suite.results.append(self.benchmark_module_lookup(iterations=50))
        suite.results.append(self.benchmark_module_stats(iterations=50))

        suite.total_duration_ms = (time.perf_counter() - suite_start) * 1000
        return suite

    def generate_markdown_report(self, suite: BenchmarkSuite, output_path: str) -> str:
        """Generate a markdown benchmark report.

        Parameters
        ----------
        suite : BenchmarkSuite
            Benchmark suite to report.
        output_path : str
            Output file path.

        Returns
        -------
        str
            Markdown content.
        """
        lines = [
            "# Fleet Benchmark Report",
            "",
            f"**Suite:** {suite.suite_name}",
            f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(suite.timestamp))}",
            f"**Total Duration:** {suite.total_duration_ms:.1f} ms",
            f"**Benchmarks:** {len(suite.results)}",
            "",
            "## Summary",
            "",
            "| Benchmark | Iterations | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | Ops/sec | Peak Mem (MB) |",
            "|-----------|------------|-----------|-------------|----------|----------|---------|---------------|",
        ]

        for r in suite.results:
            lines.append(
                f"| {r.name} | {r.iterations} | {r.mean_ms:.3f} | {r.median_ms:.3f} | "
                f"{r.p95_ms:.3f} | {r.p99_ms:.3f} | {r.throughput_ops_per_sec:.1f} | {r.memory_peak_mb:.2f} |"
            )

        lines.extend([
            "",
            "## Detailed Results",
            "",
        ])

        for r in suite.results:
            lines.extend([
                f"### {r.name}",
                "",
                f"- **Iterations:** {r.iterations}",
                f"- **Total:** {r.total_ms:.2f} ms",
                f"- **Min:** {r.min_ms:.3f} ms",
                f"- **Max:** {r.max_ms:.3f} ms",
                f"- **Mean:** {r.mean_ms:.3f} ms",
                f"- **Median:** {r.median_ms:.3f} ms",
                f"- **p95:** {r.p95_ms:.3f} ms",
                f"- **p99:** {r.p99_ms:.3f} ms",
                f"- **StdDev:** {r.stdev_ms:.3f} ms",
                f"- **Throughput:** {r.throughput_ops_per_sec:.1f} ops/sec",
                f"- **Peak Memory:** {r.memory_peak_mb:.2f} MB",
                f"- **Current Memory:** {r.memory_current_mb:.2f} MB",
                "",
            ])

        content = "\n".join(lines)
        Path(output_path).write_text(content)
        return content

    def generate_json_report(self, suite: BenchmarkSuite, output_path: str) -> str:
        """Generate a JSON benchmark report.

        Parameters
        ----------
        suite : BenchmarkSuite
            Benchmark suite to report.
        output_path : str
            Output file path.

        Returns
        -------
        str
            JSON content.
        """
        import json

        content = json.dumps(suite.to_dict(), indent=2)
        Path(output_path).write_text(content)
        return content

    def compare_to_baseline(
        self, suite: BenchmarkSuite, baseline_path: str
    ) -> dict[str, Any]:
        """Compare suite to a baseline benchmark.

        Parameters
        ----------
        suite : BenchmarkSuite
            Current benchmark suite.
        baseline_path : str
            Path to baseline JSON file.

        Returns
        -------
        dict
            Comparison results with regression flags.
        """
        import json

        baseline = json.loads(Path(baseline_path).read_text())
        baseline_results = {r["name"]: r for r in baseline.get("results", [])}

        comparison = {
            "suite_name": suite.suite_name,
            "timestamp": suite.timestamp,
            "comparisons": [],
            "regressions": [],
            "improvements": [],
        }

        for result in suite.results:
            baseline_result = baseline_results.get(result.name)
            if not baseline_result:
                continue

            baseline_mean = baseline_result["mean_ms"]
            current_mean = result.mean_ms
            change_pct = ((current_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0

            comp = {
                "name": result.name,
                "baseline_mean_ms": baseline_mean,
                "current_mean_ms": current_mean,
                "change_pct": change_pct,
                "regression": change_pct > 20,  # >20% slower = regression
                "improvement": change_pct < -20,  # >20% faster = improvement
            }
            comparison["comparisons"].append(comp)

            if comp["regression"]:
                comparison["regressions"].append(comp)
            elif comp["improvement"]:
                comparison["improvements"].append(comp)

        return comparison
