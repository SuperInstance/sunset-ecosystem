"""FleetCLI — unified command-line interface for fleet management.

Provides subcommands for running fleet beats, generating reports, checking health,
managing modules, running tests, and viewing dashboards.

Usage
-----
    python -m fleet.fleet_cli --help
    python -m fleet.fleet_cli beat
    python -m fleet.fleet_cli health
    python -m fleet.fleet_cli report --all
    python -m fleet.fleet_cli test --module swarm.mesh_wal
    python -m fleet.fleet_cli dashboard --serve
"""

from __future__ import annotations

__all__ = [
    "FleetCLI",
    "CLIResult",
]

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fleet.fleet_dashboard import FleetDashboard
from fleet.fleet_doc import FleetDoc
from fleet.fleet_metrics_collector import FleetMetricsCollector
from fleet.fleet_orchestrator import FleetOrchestrator
from fleet.fleet_reporter import FleetReporter
from fleet.harbor import Harbor
from fleet.ternary_types import TernaryValue


@dataclass
class CLIResult:
    """Result of a CLI command execution."""

    command: str
    success: bool
    message: str
    data: dict[str, Any] | None = None
    duration_ms: float = 0.0


class FleetCLI:
    """Fleet command-line interface.

    Parameters
    ----------
    workspace : str
        Path to sunset-ecosystem workspace.
    """

    def __init__(self, workspace: str = ".") -> None:
        self.workspace = Path(workspace)
        self._orchestrator: FleetOrchestrator | None = None
        self._harbor: Harbor | None = None
        self._dashboard: FleetDashboard | None = None
        self._reporter: FleetReporter | None = None
        self._metrics: FleetMetricsCollector | None = None

    def _ensure_orchestrator(self) -> None:
        if self._orchestrator is None:
            self._orchestrator = FleetOrchestrator(workspace=str(self.workspace))
            self._orchestrator.initialize_fleet()

    def _ensure_harbor(self) -> None:
        if self._harbor is None:
            self._harbor = Harbor(str(self.workspace))
            self._harbor.bootstrap_fleet()

    def _ensure_dashboard(self) -> None:
        if self._dashboard is None:
            self._dashboard = FleetDashboard(str(self.workspace))

    def _ensure_reporter(self) -> None:
        if self._reporter is None:
            self._reporter = FleetReporter(str(self.workspace))

    def _ensure_metrics(self) -> None:
        if self._metrics is None:
            self._metrics = FleetMetricsCollector(str(self.workspace))

    # ── Commands ──────────────────────────────────────────────

    def beat(self, count: int = 1) -> CLIResult:
        """Run fleet beat(s).

        Parameters
        ----------
        count : int
            Number of beats to run.

        Returns
        -------
        CLIResult
            Command result.
        """
        start = time.time()
        self._ensure_orchestrator()

        for i in range(count):
            self._orchestrator.beat()

        duration = (time.time() - start) * 1000
        return CLIResult(
            command="beat",
            success=True,
            message=f"Ran {count} beat(s). Cycle: {self._orchestrator.cycle_number}",
            data={"cycle": self._orchestrator.cycle_number, "beats": count},
            duration_ms=duration,
        )

    def health(self) -> CLIResult:
        """Check fleet health.

        Returns
        -------
        CLIResult
            Command result.
        """
        start = time.time()
        self._ensure_orchestrator()
        health = self._orchestrator.check_fleet_health()

        # Compute overall_status from available data
        critical = health.get("critical", 0)
        degraded = health.get("degraded", 0)
        total = health.get("total_modules", 0)
        if critical > 0:
            overall_status = "CRITICAL"
        elif degraded > total * 0.2:
            overall_status = "DEGRADED"
        else:
            overall_status = "HEALTHY"

        duration = (time.time() - start) * 1000
        return CLIResult(
            command="health",
            success=overall_status == "HEALTHY",
            message=overall_status,
            data={
                "status": overall_status,
                "healthy": health.get("healthy", 0),
                "warning": health.get("degraded", 0),
                "critical": health.get("critical", 0),
                "total": health.get("total_modules", 0),
                "issues": health.get("recommendations", []),
            },
            duration_ms=duration,
        )

    def report(
        self, report_type: str | None = None, all_reports: bool = False
    ) -> CLIResult:
        """Generate fleet reports.

        Parameters
        ----------
        report_type : str | None
            Type of report to generate. If None and all_reports is False, generates executive summary.
        all_reports : bool
            Generate all report types.

        Returns
        -------
        CLIResult
            Command result.
        """
        start = time.time()
        self._ensure_reporter()

        if all_reports:
            results = self._reporter.generate_all_reports()
            success = all(r.success for r in results)
            message = f"Generated {len(results)} reports"
        elif report_type:
            generators = {
                "dashboard": self._reporter.generate_dashboard,
                "api": self._reporter.generate_api_docs,
                "architecture": self._reporter.generate_architecture,
                "integration": self._reporter.generate_integration_guide,
                "trend": self._reporter.generate_trend_report,
                "summary": self._reporter.generate_executive_summary,
            }
            generator = generators.get(report_type)
            if not generator:
                return CLIResult(
                    command="report",
                    success=False,
                    message=f"Unknown report type: {report_type}",
                    data={"available": list(generators.keys())},
                )
            result = generator()
            success = result.success
            message = f"Generated {report_type} report"
        else:
            result = self._reporter.generate_executive_summary()
            success = result.success
            message = "Generated executive summary"

        duration = (time.time() - start) * 1000
        return CLIResult(
            command="report",
            success=success,
            message=message,
            duration_ms=duration,
        )

    def test(self, module: str | None = None, all_modules: bool = False) -> CLIResult:
        """Run fleet tests.

        Parameters
        ----------
        module : str | None
            Module name to test (e.g., 'swarm.mesh_wal').
        all_modules : bool
            Run all tests.

        Returns
        -------
        CLIResult
            Command result.
        """
        start = time.time()

        if module:
            # Convert module name to test file path
            parts = module.split(".")
            if parts[0] in ["fleet", "swarm", "nerve", "nexus"]:
                test_file = f"tests/test_{parts[-1]}.py"
            else:
                test_file = f"tests/test_{module.replace('.', '_')}.py"

            cmd = ["python3", "-m", "pytest", test_file, "-v", "--tb=short"]
        elif all_modules:
            cmd = ["python3", "-m", "pytest", "tests/", "-x", "--tb=short"]
        else:
            # Run a small subset (5 test files) to avoid timeout
            test_files = [
                "tests/test_hnsw_mesh_table.py",
                "tests/test_tiered_mesh_storage.py",
                "tests/test_fleet_memory.py",
                "tests/test_cognitive_cache.py",
                "tests/test_fleet_api.py",
            ]
            cmd = ["python3", "-m", "pytest"] + test_files + ["-v", "--tb=short"]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(self.workspace)
        )
        success = result.returncode == 0

        duration = (time.time() - start) * 1000
        return CLIResult(
            command="test",
            success=success,
            message="Tests passed" if success else "Tests failed",
            data={
                "stdout": result.stdout[-2000:]
                if len(result.stdout) > 2000
                else result.stdout,
                "stderr": result.stderr[-2000:]
                if len(result.stderr) > 2000
                else result.stderr,
                "returncode": result.returncode,
            },
            duration_ms=duration,
        )

    def dashboard(self, serve: bool = False, output: str | None = None) -> CLIResult:
        """Generate or serve fleet dashboard.

        Parameters
        ----------
        serve : bool
            Start a simple HTTP server for the dashboard.
        output : str | None
            Output file path for markdown dashboard.

        Returns
        -------
        CLIResult
            Command result.
        """
        start = time.time()
        self._ensure_dashboard()

        if serve:
            # Generate reports first
            self._ensure_reporter()
            self._reporter.generate_all_reports()

            # Start simple HTTP server
            import http.server
            import socketserver

            report_dir = str(self._reporter.output_dir)
            handler = http.server.SimpleHTTPRequestHandler

            with socketserver.TCPServer(("", 8080), handler) as httpd:
                print(f"Dashboard serving at http://localhost:8080/{report_dir}/")
                print("Press Ctrl+C to stop")
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    pass

            duration = (time.time() - start) * 1000
            return CLIResult(
                command="dashboard",
                success=True,
                message="Dashboard server stopped",
                duration_ms=duration,
            )
        else:
            output_path = output or str(self.workspace / "docs" / "FLEET_DASHBOARD.md")
            self._dashboard.generate_markdown_report(output_path)

            duration = (time.time() - start) * 1000
            return CLIResult(
                command="dashboard",
                success=True,
                message=f"Dashboard written to {output_path}",
                data={"path": output_path},
                duration_ms=duration,
            )

    def modules(
        self, list_modules: bool = False, show_stats: bool = False
    ) -> CLIResult:
        """List or analyze fleet modules.

        Parameters
        ----------
        list_modules : bool
            List all modules.
        show_stats : bool
            Show module statistics.

        Returns
        -------
        CLIResult
            Command result.
        """
        start = time.time()
        self._ensure_harbor()

        module_data = []
        for mod in self._harbor.modules.values():
            module_data.append(
                {
                    "name": mod.name,
                    "status": mod.status,
                    "tests": f"{mod.test_passed}/{mod.test_count}",
                    "coverage": f"{mod.test_coverage * 100:.0f}%",
                    "health": TernaryValue.to_emoji(mod.health_ternary),
                }
            )

        duration = (time.time() - start) * 1000
        return CLIResult(
            command="modules",
            success=True,
            message=f"{len(module_data)} modules found",
            data={"modules": module_data},
            duration_ms=duration,
        )

    def metrics(self, collect: bool = False, show_trends: bool = False) -> CLIResult:
        """Collect or show fleet metrics.

        Parameters
        ----------
        collect : bool
            Collect metrics from a beat.
        show_trends : bool
            Show trend analysis.

        Returns
        -------
        CLIResult
            Command result.
        """
        start = time.time()
        self._ensure_metrics()

        if collect:
            snapshot = self._metrics.record_beat_metrics()
            message = f"Metrics collected for cycle {snapshot.cycle_number}"
            data = {
                "cycle": snapshot.cycle_number,
                "health": f"{snapshot.health_score * 100:.0f}%",
                "tests": f"{snapshot.tests_passed}/{snapshot.total_tests}",
            }
        elif show_trends:
            # Need at least 2 snapshots for trends
            for _ in range(3):
                self._metrics.record_beat_metrics()
            trends = self._metrics.analyze_trends()
            trend_data = {}
            for name, trend in trends.items():
                trend_data[name] = {
                    "direction": trend.direction,
                    "change": f"{trend.change_pct:+.1f}%",
                    "confidence": f"{trend.confidence * 100:.0f}%",
                }
            message = f"{len(trends)} trends analyzed"
            data = {"trends": trend_data}
        else:
            snapshot = self._metrics.get_latest_snapshot()
            if snapshot:
                message = f"Latest metrics: cycle {snapshot.cycle_number}"
                data = {
                    "cycle": snapshot.cycle_number,
                    "health": f"{snapshot.health_score * 100:.0f}%",
                    "tests": f"{snapshot.tests_passed}/{snapshot.total_tests}",
                }
            else:
                message = "No metrics recorded yet"
                data = {}

        duration = (time.time() - start) * 1000
        return CLIResult(
            command="metrics",
            success=True,
            message=message,
            data=data,
            duration_ms=duration,
        )

    def status(self) -> CLIResult:
        """Get overall fleet status.

        Returns
        -------
        CLIResult
            Command result.
        """
        start = time.time()
        self._ensure_orchestrator()
        self._ensure_harbor()

        health = self._orchestrator.check_fleet_health()
        harbor_stats = self._harbor.get_stats()

        # Compute overall_status from available data
        critical = health.get("critical", 0)
        degraded = health.get("degraded", 0)
        total = health.get("total_modules", 0)
        if critical > 0:
            overall_status = "CRITICAL"
        elif degraded > total * 0.2:
            overall_status = "DEGRADED"
        else:
            overall_status = "HEALTHY"

        duration = (time.time() - start) * 1000
        return CLIResult(
            command="status",
            success=overall_status == "HEALTHY",
            message=f"Fleet status: {overall_status}",
            data={
                "status": overall_status,
                "modules": harbor_stats.get("modules", 0),
                "tests": harbor_stats.get("tests", 0),
                "healthy": health.get("healthy", 0),
                "warning": health.get("degraded", 0),
                "critical": health.get("critical", 0),
            },
            duration_ms=duration,
        )

    # ── Argument Parsing ──────────────────────────────────────

    def run_args(self, args: list[str] | None = None) -> CLIResult:
        """Parse command-line arguments and execute.

        Parameters
        ----------
        args : list[str] | None
            Command-line arguments. Defaults to sys.argv[1:].

        Returns
        -------
        CLIResult
            Command result.
        """
        parser = argparse.ArgumentParser(
            prog="fleet",
            description="Sunset Ecosystem Fleet CLI",
        )
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # beat
        beat_parser = subparsers.add_parser("beat", help="Run fleet beat(s)")
        beat_parser.add_argument(
            "--count", "-c", type=int, default=1, help="Number of beats"
        )

        # health
        subparsers.add_parser("health", help="Check fleet health")

        # report
        report_parser = subparsers.add_parser("report", help="Generate reports")
        report_parser.add_argument(
            "--type",
            "-t",
            choices=[
                "dashboard",
                "api",
                "architecture",
                "integration",
                "trend",
                "summary",
            ],
            help="Report type",
        )
        report_parser.add_argument(
            "--all", "-a", action="store_true", help="Generate all reports"
        )

        # test
        test_parser = subparsers.add_parser("test", help="Run tests")
        test_parser.add_argument("--module", "-m", help="Module to test")
        test_parser.add_argument(
            "--all", "-a", action="store_true", help="Run all tests"
        )

        # dashboard
        dash_parser = subparsers.add_parser("dashboard", help="Generate dashboard")
        dash_parser.add_argument(
            "--serve", "-s", action="store_true", help="Serve dashboard"
        )
        dash_parser.add_argument("--output", "-o", help="Output file")

        # modules
        mod_parser = subparsers.add_parser("modules", help="List modules")
        mod_parser.add_argument("--stats", action="store_true", help="Show statistics")

        # metrics
        met_parser = subparsers.add_parser("metrics", help="Show/collect metrics")
        met_parser.add_argument(
            "--collect", "-c", action="store_true", help="Collect metrics"
        )
        met_parser.add_argument("--trends", action="store_true", help="Show trends")

        # status
        subparsers.add_parser("status", help="Fleet status")

        parsed = parser.parse_args(args)

        if not parsed.command:
            parser.print_help()
            return CLIResult(command="help", success=True, message="Printed help")

        commands = {
            "beat": lambda: self.beat(count=parsed.count),
            "health": lambda: self.health(),
            "report": lambda: self.report(
                report_type=parsed.type, all_reports=parsed.all
            ),
            "test": lambda: self.test(module=parsed.module, all_modules=parsed.all),
            "dashboard": lambda: self.dashboard(
                serve=parsed.serve, output=parsed.output
            ),
            "modules": lambda: self.modules(list_modules=True, show_stats=parsed.stats),
            "metrics": lambda: self.metrics(
                collect=parsed.collect, show_trends=parsed.trends
            ),
            "status": lambda: self.status(),
        }

        cmd = commands.get(parsed.command)
        if cmd:
            return cmd()

        return CLIResult(
            command=parsed.command, success=False, message="Unknown command"
        )

    # ── Console Output ────────────────────────────────────────

    @staticmethod
    def print_result(result: CLIResult) -> None:
        """Print a CLI result to console."""
        status = "✅" if result.success else "❌"
        print(f"{status} {result.command:12} {result.message}")
        if result.data:
            print(json.dumps(result.data, indent=2))


def main() -> None:
    """CLI entry point."""
    cli = FleetCLI()
    result = cli.run_args()
    FleetCLI.print_result(result)
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
