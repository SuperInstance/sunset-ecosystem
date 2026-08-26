"""FleetReporter — automated report generator for the fleet ecosystem.

Orchestrates FleetDashboard, FleetDoc, and FleetMetricsCollector to
automatically generate and publish comprehensive fleet reports on a
schedule or on demand.

Usage
-----
    from fleet.fleet_reporter import FleetReporter

    reporter = FleetReporter()
    reporter.generate_all_reports("docs/")
    reporter.publish_to_git("docs/")
"""

from __future__ import annotations

__all__ = [
    "FleetReporter",
    "ReportJob",
    "ReportResult",
]

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleet.fleet_dashboard import FleetDashboard
from fleet.fleet_doc import FleetDoc
from fleet.fleet_metrics_collector import FleetMetricsCollector


@dataclass
class ReportJob:
    """A scheduled report generation job."""

    name: str
    report_type: str
    output_path: str
    schedule_minutes: int = 0  # 0 = on-demand only
    last_run: float = 0.0
    run_count: int = 0


@dataclass
class ReportResult:
    """Result of a report generation."""

    job_name: str
    success: bool
    output_path: str
    duration_ms: float
    error: str | None = None


class FleetReporter:
    """Fleet automated report generator.

    Parameters
    ----------
    workspace : str
        Path to sunset-ecosystem workspace.
    output_dir : str
        Directory where reports are published.
    """

    REPORT_TYPES = {
        "dashboard": "Fleet status dashboard",
        "api_docs": "API documentation",
        "architecture": "Architecture diagram",
        "integration_guide": "Integration guide",
        "trend_report": "Metrics trend report",
        "executive_summary": "Executive summary",
    }

    def __init__(
        self,
        workspace: str = ".",
        output_dir: str = "docs/reports",
    ) -> None:
        self.workspace = Path(workspace)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._dashboard: FleetDashboard | None = None
        self._doc: FleetDoc | None = None
        self._metrics: FleetMetricsCollector | None = None
        self._jobs: list[ReportJob] = []
        self._results: list[ReportResult] = []

    def _init_subsystems(self) -> None:
        """Initialize subsystems lazily."""
        if self._dashboard is None:
            self._dashboard = FleetDashboard(str(self.workspace))
        if self._doc is None:
            self._doc = FleetDoc(str(self.workspace))
        if self._metrics is None:
            self._metrics = FleetMetricsCollector(str(self.workspace))

    # ── Report Generation ─────────────────────────────────────

    def generate_dashboard(self, path: str | Path | None = None) -> ReportResult:
        """Generate fleet dashboard report.

        Parameters
        ----------
        path : str | Path | None
            Output path. Defaults to output_dir / DASHBOARD.md.

        Returns
        -------
        ReportResult
            Generation result.
        """
        self._init_subsystems()
        if not self._dashboard:
            return ReportResult("dashboard", False, "", 0.0, "Dashboard not available")

        start = time.time()
        output = Path(path) if path else self.output_dir / "DASHBOARD.md"
        try:
            self._dashboard.generate_markdown_report(output)
            duration = (time.time() - start) * 1000
            return ReportResult("dashboard", True, str(output), duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ReportResult("dashboard", False, str(output), duration, str(e))

    def generate_api_docs(self, path: str | Path | None = None) -> ReportResult:
        """Generate API documentation.

        Parameters
        ----------
        path : str | Path | None
            Output path. Defaults to output_dir / API_INDEX.md.

        Returns
        -------
        ReportResult
            Generation result.
        """
        self._init_subsystems()
        if not self._doc:
            return ReportResult(
                "api_docs", False, "", 0.0, "Doc generator not available"
            )

        start = time.time()
        output = Path(path) if path else self.output_dir / "API_INDEX.md"
        try:
            self._doc.generate_api_docs(output)
            duration = (time.time() - start) * 1000
            return ReportResult("api_docs", True, str(output), duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ReportResult("api_docs", False, str(output), duration, str(e))

    def generate_architecture(self, path: str | Path | None = None) -> ReportResult:
        """Generate architecture diagram.

        Parameters
        ----------
        path : str | Path | None
            Output path. Defaults to output_dir / ARCHITECTURE.md.

        Returns
        -------
        ReportResult
            Generation result.
        """
        self._init_subsystems()
        if not self._doc:
            return ReportResult(
                "architecture", False, "", 0.0, "Doc generator not available"
            )

        start = time.time()
        output = Path(path) if path else self.output_dir / "ARCHITECTURE.md"
        try:
            self._doc.generate_architecture_diagram(output)
            duration = (time.time() - start) * 1000
            return ReportResult("architecture", True, str(output), duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ReportResult("architecture", False, str(output), duration, str(e))

    def generate_integration_guide(
        self, path: str | Path | None = None
    ) -> ReportResult:
        """Generate integration guide.

        Parameters
        ----------
        path : str | Path | None
            Output path. Defaults to output_dir / INTEGRATION_GUIDE.md.

        Returns
        -------
        ReportResult
            Generation result.
        """
        self._init_subsystems()
        if not self._doc:
            return ReportResult(
                "integration_guide", False, "", 0.0, "Doc generator not available"
            )

        start = time.time()
        output = Path(path) if path else self.output_dir / "INTEGRATION_GUIDE.md"
        try:
            self._doc.generate_integration_guide(output)
            duration = (time.time() - start) * 1000
            return ReportResult("integration_guide", True, str(output), duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ReportResult(
                "integration_guide", False, str(output), duration, str(e)
            )

    def generate_trend_report(self, path: str | Path | None = None) -> ReportResult:
        """Generate metrics trend report.

        Parameters
        ----------
        path : str | Path | None
            Output path. Defaults to output_dir / TREND_REPORT.md.

        Returns
        -------
        ReportResult
            Generation result.
        """
        self._init_subsystems()
        if not self._metrics:
            return ReportResult(
                "trend_report", False, "", 0.0, "Metrics collector not available"
            )

        start = time.time()
        output = Path(path) if path else self.output_dir / "TREND_REPORT.md"
        try:
            # Ensure we have metrics data
            self._metrics.record_beat_metrics()
            self._metrics.generate_trend_report(output)
            duration = (time.time() - start) * 1000
            return ReportResult("trend_report", True, str(output), duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ReportResult("trend_report", False, str(output), duration, str(e))

    def generate_executive_summary(
        self, path: str | Path | None = None
    ) -> ReportResult:
        """Generate executive summary combining all reports.

        Parameters
        ----------
        path : str | Path | None
            Output path. Defaults to output_dir / EXECUTIVE_SUMMARY.md.

        Returns
        -------
        ReportResult
            Generation result.
        """
        self._init_subsystems()
        start = time.time()
        output = Path(path) if path else self.output_dir / "EXECUTIVE_SUMMARY.md"

        try:
            lines: list[str] = []
            lines.append("# 🌅 Sunset Ecosystem Executive Summary")
            lines.append("")
            lines.append(
                f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}*"
            )
            lines.append("")

            # Dashboard metrics
            if self._dashboard:
                metrics = self._dashboard.get_fleet_metrics()
                lines.append("## Fleet Status")
                lines.append("")
                lines.append(f"| Metric | Value |")
                lines.append(f"|--------|-------|")
                lines.append(f"| Modules | {metrics.total_modules} |")
                lines.append(
                    f"| Healthy | {metrics.healthy_modules} ({metrics.health_score * 100:.0f}%) |"
                )
                lines.append(f"| Tests | {metrics.total_tests} |")
                lines.append(f"| Coverage | {metrics.test_coverage_pct * 100:.0f}% |")
                lines.append(
                    f"| Integrations | {metrics.tested_integrations}/{metrics.integration_count} |"
                )
                lines.append("")

            # Trend summary
            if self._metrics:
                self._metrics.record_beat_metrics()
                trends = self._metrics.analyze_trends()
                if trends:
                    lines.append("## Trends")
                    lines.append("")
                    for name, trend in trends.items():
                        emoji = (
                            "📈"
                            if trend.direction == "improving"
                            else "📉"
                            if trend.direction == "degrading"
                            else "➡️"
                        )
                        lines.append(
                            f"- {emoji} **{name}**: {trend.direction} ({trend.change_pct:+.1f}%)"
                        )
                    lines.append("")

            # Alerts
            if self._metrics:
                alerts = self._metrics.check_alerts()
                lines.append("## Alerts")
                lines.append("")
                if alerts:
                    for alert in alerts:
                        emoji = "🔴" if alert["level"] == "critical" else "🟡"
                        lines.append(
                            f"- {emoji} **{alert['level'].upper()}**: {alert['message']}"
                        )
                else:
                    lines.append("✅ No alerts at this time.")
                lines.append("")

            # Module highlights
            if self._doc:
                try:
                    all_docs = self._doc.parse_all_modules()
                    if all_docs:
                        lines.append("## Module Highlights")
                        lines.append("")
                        for name, mod_doc in list(all_docs.items())[:5]:
                            health = "🟢" if mod_doc.test_count > 0 else "🟡"
                            lines.append(
                                f"- {health} **{name}**: {mod_doc.test_count} tests, {len(mod_doc.classes)} classes, {len(mod_doc.functions)} functions"
                            )
                        lines.append("")
                except Exception:
                    pass

            lines.append("## Report Index")
            lines.append("")
            for report_type, description in self.REPORT_TYPES.items():
                filename = f"{report_type.upper().replace('_', '_')}.md"
                lines.append(f"- [{description}]({filename})")
            lines.append("")

            content = "\n".join(lines)
            output.write_text(content)
            duration = (time.time() - start) * 1000
            return ReportResult("executive_summary", True, str(output), duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ReportResult(
                "executive_summary", False, str(output), duration, str(e)
            )

    # ── Batch Generation ──────────────────────────────────────

    def generate_all_reports(
        self, output_dir: str | Path | None = None
    ) -> list[ReportResult]:
        """Generate all report types.

        Parameters
        ----------
        output_dir : str | Path | None
            Output directory. Defaults to self.output_dir.

        Returns
        -------
        list[ReportResult]
            Results for each report.
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        results: list[ReportResult] = []
        results.append(self.generate_dashboard())
        results.append(self.generate_api_docs())
        results.append(self.generate_architecture())
        results.append(self.generate_integration_guide())
        results.append(self.generate_trend_report())
        results.append(self.generate_executive_summary())

        self._results.extend(results)
        return results

    # ── Scheduling ────────────────────────────────────────────

    def schedule_report(
        self, name: str, report_type: str, schedule_minutes: int
    ) -> ReportJob:
        """Schedule a report to be generated periodically.

        Parameters
        ----------
        name : str
            Job name.
        report_type : str
            Type of report to generate.
        schedule_minutes : int
            Generation interval in minutes.

        Returns
        -------
        ReportJob
            The scheduled job.
        """
        job = ReportJob(
            name=name,
            report_type=report_type,
            output_path=str(self.output_dir / f"{name}.md"),
            schedule_minutes=schedule_minutes,
        )
        self._jobs.append(job)
        return job

    def check_scheduled_jobs(self) -> list[ReportResult]:
        """Check and run any scheduled jobs that are due.

        Returns
        -------
        list[ReportResult]
            Results for jobs that were run.
        """
        now = time.time()
        results: list[ReportResult] = []

        for job in self._jobs:
            if job.schedule_minutes <= 0:
                continue

            elapsed = (now - job.last_run) / 60
            if elapsed >= job.schedule_minutes:
                result = self._run_job(job)
                results.append(result)
                job.last_run = now
                job.run_count += 1

        self._results.extend(results)
        return results

    def _run_job(self, job: ReportJob) -> ReportResult:
        """Run a single report job."""
        generators = {
            "dashboard": self.generate_dashboard,
            "api_docs": self.generate_api_docs,
            "architecture": self.generate_architecture,
            "integration_guide": self.generate_integration_guide,
            "trend_report": self.generate_trend_report,
            "executive_summary": self.generate_executive_summary,
        }

        generator = generators.get(job.report_type)
        if generator:
            return generator(job.output_path)

        return ReportResult(
            job.name,
            False,
            job.output_path,
            0.0,
            f"Unknown report type: {job.report_type}",
        )

    # ── Git Publishing ────────────────────────────────────────

    def publish_to_git(self, output_dir: str | Path | None = None) -> dict[str, Any]:
        """Publish reports to git.

        Parameters
        ----------
        output_dir : str | Path | None
            Directory to publish. Defaults to self.output_dir.

        Returns
        -------
        dict[str, Any]
            Git operation result.
        """
        publish_dir = Path(output_dir) if output_dir else self.output_dir

        try:
            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                cwd=str(self.workspace),
            )
            if result.returncode != 0:
                return {"status": "error", "message": "Not in a git repository"}

            # Add all reports
            subprocess.run(
                ["git", "add", str(publish_dir)],
                capture_output=True,
                cwd=str(self.workspace),
            )

            # Commit
            commit_result = subprocess.run(
                ["git", "commit", "-m", "Auto-generated fleet reports"],
                capture_output=True,
                text=True,
                cwd=str(self.workspace),
            )

            return {
                "status": "published",
                "commit_exit": commit_result.returncode,
                "commit_output": commit_result.stdout,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ── Statistics ──────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get reporter statistics."""
        total = len(self._results)
        successful = sum(1 for r in self._results if r.success)
        total_duration = sum(r.duration_ms for r in self._results)

        return {
            "total_reports": total,
            "successful": successful,
            "failed": total - successful,
            "total_duration_ms": total_duration,
            "average_duration_ms": total_duration / total if total > 0 else 0.0,
            "scheduled_jobs": len(self._jobs),
            "report_types": list(self.REPORT_TYPES.keys()),
        }

    # ── Console Output ───────────────────────────────────────

    def print_summary(self) -> None:
        """Print a console summary of report status."""
        stats = self.get_stats()
        print("═" * 50)
        print(" 📰 FLEET REPORTER")
        print("═" * 50)
        print(f"  Reports:      {stats['successful']:3d}/{stats['total_reports']:3d}")
        print(f"  Failed:       {stats['failed']:3d}")
        print(f"  Avg Duration: {stats['average_duration_ms']:6.1f}ms")
        print(f"  Scheduled:    {stats['scheduled_jobs']:3d}")
        print("═" * 50)

        if self._results:
            print("  Recent reports:")
            for result in self._results[-5:]:
                status = "✅" if result.success else "❌"
                print(f"    {status} {result.job_name:20} {result.duration_ms:6.1f}ms")
            print("═" * 50)
