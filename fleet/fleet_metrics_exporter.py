"""FleetMetricsExporter — metrics export to Prometheus and InfluxDB formats.

Exports fleet metrics in Prometheus exposition format and InfluxDB line protocol.
Supports HTTP endpoints for scraping and file-based export for batch ingestion.

Reference
---------
- Prometheus exposition format: https://prometheus.io/docs/instrumenting/exposition_formats/
- InfluxDB line protocol: https://docs.influxdata.com/influxdb/latest/reference/syntax/line-protocol/

Usage
-----
    exporter = FleetMetricsExporter()
    prometheus_text = exporter.to_prometheus()
    influxdb_text = exporter.to_influxdb()
    exporter.serve_http(port=9090)
"""

from __future__ import annotations

__all__ = [
    "FleetMetricsExporter",
    "PrometheusMetric",
    "InfluxDBPoint",
]

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleet.fleet_benchmark import FleetBenchmark
from fleet.fleet_metrics_collector import FleetMetricsCollector
from fleet.fleet_orchestrator import FleetOrchestrator


@dataclass
class PrometheusMetric:
    """A Prometheus metric."""

    name: str
    value: float
    metric_type: str = "gauge"  # gauge, counter, histogram, summary
    help_text: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    timestamp_ms: int = 0

    def to_prometheus(self) -> str:
        """Render as Prometheus exposition format."""
        lines = []
        if self.help_text:
            lines.append(f"# HELP {self.name} {self.help_text}")
        lines.append(f"# TYPE {self.name} {self.metric_type}")

        label_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        if label_str:
            lines.append(f"{self.name}{{{label_str}}} {self.value}")
        else:
            lines.append(f"{self.name} {self.value}")
        return "\n".join(lines) + "\n"


@dataclass
class InfluxDBPoint:
    """An InfluxDB line protocol point."""

    measurement: str
    fields: dict[str, float]
    tags: dict[str, str] = field(default_factory=dict)
    timestamp_ns: int = 0

    def to_line(self) -> str:
        """Render as InfluxDB line protocol."""
        tag_str = ",".join(f"{k}={v}" for k, v in self.tags.items())
        field_str = ",".join(f"{k}={v}" for k, v in self.fields.items())

        if tag_str:
            line = f"{self.measurement},{tag_str} {field_str}"
        else:
            line = f"{self.measurement} {field_str}"

        if self.timestamp_ns > 0:
            line += f" {self.timestamp_ns}"
        return line + "\n"


class FleetMetricsExporter:
    """Fleet metrics exporter.

    Parameters
    ----------
    workspace : str
        Path to sunset-ecosystem workspace.
    """

    def __init__(self, workspace: str = ".") -> None:
        self.workspace = Path(workspace)
        self._orchestrator: FleetOrchestrator | None = None
        self._metrics: FleetMetricsCollector | None = None
        self._benchmark: FleetBenchmark | None = None

    def _ensure_orchestrator(self) -> None:
        if self._orchestrator is None:
            self._orchestrator = FleetOrchestrator(workspace=str(self.workspace))
            self._orchestrator.initialize_fleet()

    def _ensure_metrics(self) -> None:
        if self._metrics is None:
            self._metrics = FleetMetricsCollector(workspace=str(self.workspace))

    def _ensure_benchmark(self) -> None:
        if self._benchmark is None:
            self._benchmark = FleetBenchmark(workspace=str(self.workspace))

    def collect_fleet_metrics(self) -> dict[str, Any]:
        """Collect all fleet metrics.

        Returns
        -------
        dict
            Combined metrics from orchestrator, metrics collector, and benchmark.
        """
        self._ensure_orchestrator()
        self._ensure_metrics()

        health = self._orchestrator.check_fleet_health()
        snapshot = self._metrics.record_beat_metrics()

        return {
            "modules_total": health.get("total_modules", 0),
            "modules_healthy": health.get("healthy", 0),
            "modules_degraded": health.get("degraded", 0),
            "modules_critical": health.get("critical", 0),
            "test_coverage": health.get("test_coverage", 0.0),
            "tests_passed": snapshot.tests_passed,
            "tests_failed": snapshot.tests_failed,
            "total_tests": snapshot.total_tests,
            "health_score": snapshot.health_score,
            "cycle_number": snapshot.cycle_number,
            "timestamp": int(time.time()),
        }

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus exposition format.

        Returns
        -------
        str
            Prometheus text format.
        """
        metrics = self.collect_fleet_metrics()
        timestamp_ms = metrics["timestamp"] * 1000

        prom_metrics = [
            PrometheusMetric(
                name="fleet_modules_total",
                value=metrics["modules_total"],
                metric_type="gauge",
                help_text="Total number of fleet modules",
                timestamp_ms=timestamp_ms,
            ),
            PrometheusMetric(
                name="fleet_modules_healthy",
                value=metrics["modules_healthy"],
                metric_type="gauge",
                help_text="Number of healthy modules",
                timestamp_ms=timestamp_ms,
            ),
            PrometheusMetric(
                name="fleet_modules_degraded",
                value=metrics["modules_degraded"],
                metric_type="gauge",
                help_text="Number of degraded modules",
                timestamp_ms=timestamp_ms,
            ),
            PrometheusMetric(
                name="fleet_modules_critical",
                value=metrics["modules_critical"],
                metric_type="gauge",
                help_text="Number of critical modules",
                timestamp_ms=timestamp_ms,
            ),
            PrometheusMetric(
                name="fleet_test_coverage",
                value=metrics["test_coverage"],
                metric_type="gauge",
                help_text="Test coverage ratio",
                timestamp_ms=timestamp_ms,
            ),
            PrometheusMetric(
                name="fleet_tests_passed",
                value=metrics["tests_passed"],
                metric_type="gauge",
                help_text="Number of tests passed",
                timestamp_ms=timestamp_ms,
            ),
            PrometheusMetric(
                name="fleet_tests_failed",
                value=metrics["tests_failed"],
                metric_type="gauge",
                help_text="Number of tests failed",
                timestamp_ms=timestamp_ms,
            ),
            PrometheusMetric(
                name="fleet_health_score",
                value=metrics["health_score"],
                metric_type="gauge",
                help_text="Fleet health score (0-1)",
                timestamp_ms=timestamp_ms,
            ),
            PrometheusMetric(
                name="fleet_cycle_number",
                value=metrics["cycle_number"],
                metric_type="counter",
                help_text="Current fleet cycle number",
                timestamp_ms=timestamp_ms,
            ),
        ]

        return "".join(m.to_prometheus() for m in prom_metrics)

    def to_influxdb(self) -> str:
        """Export metrics in InfluxDB line protocol.

        Returns
        -------
        str
            InfluxDB line protocol text.
        """
        metrics = self.collect_fleet_metrics()
        timestamp_ns = metrics["timestamp"] * 1_000_000_000

        points = [
            InfluxDBPoint(
                measurement="fleet_modules",
                fields={
                    "total": metrics["modules_total"],
                    "healthy": metrics["modules_healthy"],
                    "degraded": metrics["modules_degraded"],
                    "critical": metrics["modules_critical"],
                },
                tags={"workspace": str(self.workspace.name)},
                timestamp_ns=timestamp_ns,
            ),
            InfluxDBPoint(
                measurement="fleet_tests",
                fields={
                    "passed": metrics["tests_passed"],
                    "failed": metrics["tests_failed"],
                    "total": metrics["total_tests"],
                    "coverage": metrics["test_coverage"],
                },
                tags={"workspace": str(self.workspace.name)},
                timestamp_ns=timestamp_ns,
            ),
            InfluxDBPoint(
                measurement="fleet_health",
                fields={
                    "score": metrics["health_score"],
                    "cycle": metrics["cycle_number"],
                },
                tags={"workspace": str(self.workspace.name)},
                timestamp_ns=timestamp_ns,
            ),
        ]

        return "".join(p.to_line() for p in points)

    def to_json(self) -> str:
        """Export metrics as JSON.

        Returns
        -------
        str
            JSON string.
        """
        import json

        metrics = self.collect_fleet_metrics()
        return json.dumps(metrics, indent=2)

    def write_prometheus_file(self, path: str) -> str:
        """Write Prometheus metrics to file.

        Parameters
        ----------
        path : str
            Output file path.

        Returns
        -------
        str
            File content.
        """
        content = self.to_prometheus()
        Path(path).write_text(content)
        return content

    def write_influxdb_file(self, path: str) -> str:
        """Write InfluxDB metrics to file.

        Parameters
        ----------
        path : str
            Output file path.

        Returns
        -------
        str
            File content.
        """
        content = self.to_influxdb()
        Path(path).write_text(content)
        return content

    def serve_http(self, port: int = 9090, host: str = "0.0.0.0") -> None:
        """Start HTTP server for Prometheus scraping.

        Parameters
        ----------
        port : int
            Server port.
        host : str
            Server host.
        """
        import http.server
        import socketserver

        class MetricsHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/metrics":
                    exporter = FleetMetricsExporter()
                    content = exporter.to_prometheus().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                pass

        with socketserver.TCPServer((host, port), MetricsHandler) as httpd:
            print(f"Metrics server at http://{host}:{port}/metrics")
            httpd.serve_forever()

    def benchmark_to_prometheus(self) -> str:
        """Export benchmark results as Prometheus metrics.

        Returns
        -------
        str
            Prometheus text format.
        """
        self._ensure_benchmark()
        suite = self._benchmark.run_full_suite()
        timestamp_ms = int(time.time()) * 1000

        metrics = []
        for result in suite.results:
            safe_name = result.name.replace("-", "_").replace(" ", "_")
            metrics.append(PrometheusMetric(
                name=f"fleet_benchmark_{safe_name}_mean_ms",
                value=result.mean_ms,
                metric_type="gauge",
                help_text=f"Mean latency for {result.name}",
                labels={"benchmark": result.name},
                timestamp_ms=timestamp_ms,
            ))
            metrics.append(PrometheusMetric(
                name=f"fleet_benchmark_{safe_name}_ops_per_sec",
                value=result.throughput_ops_per_sec,
                metric_type="gauge",
                help_text=f"Throughput for {result.name}",
                labels={"benchmark": result.name},
                timestamp_ms=timestamp_ms,
            ))
            metrics.append(PrometheusMetric(
                name=f"fleet_benchmark_{safe_name}_memory_peak_mb",
                value=result.memory_peak_mb,
                metric_type="gauge",
                help_text=f"Peak memory for {result.name}",
                labels={"benchmark": result.name},
                timestamp_ms=timestamp_ms,
            ))

        return "".join(m.to_prometheus() for m in metrics)
