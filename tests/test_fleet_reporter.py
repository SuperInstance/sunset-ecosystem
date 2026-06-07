"""Tests for FleetReporter — automated report generator.

Reference: fleet/fleet_reporter.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fleet.fleet_reporter import FleetReporter, ReportJob, ReportResult


class TestReportJob:
    def test_fields(self) -> None:
        job = ReportJob(name="test", report_type="dashboard", output_path="docs/test.md")
        assert job.name == "test"
        assert job.schedule_minutes == 0
        assert job.run_count == 0


class TestReportResult:
    def test_fields(self) -> None:
        result = ReportResult(
            job_name="test",
            success=True,
            output_path="docs/test.md",
            duration_ms=100.0,
        )
        assert result.success is True
        assert result.error is None

    def test_error(self) -> None:
        result = ReportResult(
            job_name="test",
            success=False,
            output_path="",
            duration_ms=0.0,
            error="Something failed",
        )
        assert result.error == "Something failed"


class TestFleetReporter:
    def test_init(self) -> None:
        reporter = FleetReporter()
        assert reporter.output_dir.exists()
        assert reporter.REPORT_TYPES is not None

    def test_init_custom_paths(self) -> None:
        reporter = FleetReporter(workspace=".", output_dir="custom_reports")
        assert reporter.workspace == Path(".")
        assert reporter.output_dir == Path("custom_reports")

    def test_generate_dashboard(self, tmp_path: Path) -> None:
        reporter = FleetReporter(output_dir=str(tmp_path / "reports"))
        result = reporter.generate_dashboard(tmp_path / "reports" / "DASHBOARD.md")
        assert result.success is True
        assert result.job_name == "dashboard"
        assert Path(result.output_path).exists()

    def test_generate_api_docs(self, tmp_path: Path) -> None:
        reporter = FleetReporter(output_dir=str(tmp_path / "reports"))
        result = reporter.generate_api_docs(tmp_path / "reports" / "API.md")
        assert result.success is True
        assert result.job_name == "api_docs"

    def test_generate_architecture(self, tmp_path: Path) -> None:
        reporter = FleetReporter(output_dir=str(tmp_path / "reports"))
        result = reporter.generate_architecture(tmp_path / "reports" / "ARCH.md")
        assert result.success is True

    def test_generate_integration_guide(self, tmp_path: Path) -> None:
        reporter = FleetReporter(output_dir=str(tmp_path / "reports"))
        result = reporter.generate_integration_guide(tmp_path / "reports" / "INTEGRATION.md")
        assert result.success is True

    def test_generate_trend_report(self, tmp_path: Path) -> None:
        reporter = FleetReporter(output_dir=str(tmp_path / "reports"))
        result = reporter.generate_trend_report(tmp_path / "reports" / "TREND.md")
        assert result.success is True

    def test_generate_executive_summary(self, tmp_path: Path) -> None:
        reporter = FleetReporter(output_dir=str(tmp_path / "reports"))
        result = reporter.generate_executive_summary(tmp_path / "reports" / "EXEC.md")
        assert result.success is True
        content = Path(result.output_path).read_text()
        assert "Sunset Ecosystem Executive Summary" in content

    def test_executive_summary_content(self, tmp_path: Path) -> None:
        reporter = FleetReporter(output_dir=str(tmp_path / "reports"))
        result = reporter.generate_executive_summary(tmp_path / "reports" / "EXEC.md")
        content = Path(result.output_path).read_text()
        assert "Fleet Status" in content
        assert "Report Index" in content

    def test_generate_all_reports(self, tmp_path: Path) -> None:
        reporter = FleetReporter(output_dir=str(tmp_path / "reports"))
        results = reporter.generate_all_reports()
        assert len(results) == 6
        assert all(r.success for r in results)

    def test_schedule_report(self, tmp_path: Path) -> None:
        reporter = FleetReporter(output_dir=str(tmp_path / "reports"))
        job = reporter.schedule_report("daily", "dashboard", schedule_minutes=60)
        assert job.name == "daily"
        assert job.report_type == "dashboard"
        assert job.schedule_minutes == 60

    def test_check_scheduled_jobs_empty(self, tmp_path: Path) -> None:
        reporter = FleetReporter(output_dir=str(tmp_path / "reports"))
        results = reporter.check_scheduled_jobs()
        assert results == []

    def test_check_scheduled_jobs_due(self, tmp_path: Path) -> None:
        reporter = FleetReporter(output_dir=str(tmp_path / "reports"))
        reporter.schedule_report("test", "dashboard", schedule_minutes=0)
        # schedule_minutes=0 means never auto-runs via check
        results = reporter.check_scheduled_jobs()
        assert results == []

    def test_run_job_dashboard(self, tmp_path: Path) -> None:
        reporter = FleetReporter(output_dir=str(tmp_path / "reports"))
        job = ReportJob(
            name="dash",
            report_type="dashboard",
            output_path=str(tmp_path / "reports" / "dash.md"),
        )
        result = reporter._run_job(job)
        assert result.success is True
        assert result.job_name == "dashboard"

    def test_run_job_unknown(self, tmp_path: Path) -> None:
        reporter = FleetReporter(output_dir=str(tmp_path / "reports"))
        job = ReportJob(
            name="unknown",
            report_type="nonexistent",
            output_path="",
        )
        result = reporter._run_job(job)
        assert result.success is False
        assert "Unknown report type" in (result.error or "")

    def test_get_stats_empty(self) -> None:
        reporter = FleetReporter()
        stats = reporter.get_stats()
        assert stats["total_reports"] == 0
        assert stats["successful"] == 0
        assert stats["average_duration_ms"] == 0.0

    def test_get_stats_after_generation(self, tmp_path: Path) -> None:
        reporter = FleetReporter(output_dir=str(tmp_path / "reports"))
        reporter.generate_all_reports()
        stats = reporter.get_stats()
        assert stats["total_reports"] == 6
        assert stats["successful"] == 6
        assert stats["average_duration_ms"] > 0

    def test_get_stats_scheduled_jobs(self, tmp_path: Path) -> None:
        reporter = FleetReporter(output_dir=str(tmp_path / "reports"))
        reporter.schedule_report("a", "dashboard", 60)
        reporter.schedule_report("b", "api_docs", 30)
        stats = reporter.get_stats()
        assert stats["scheduled_jobs"] == 2

    def test_print_summary(self, capsys) -> None:
        reporter = FleetReporter()
        reporter.generate_all_reports()
        reporter.print_summary()
        captured = capsys.readouterr()
        assert "FLEET REPORTER" in captured.out
        assert "Reports:" in captured.out

    def test_print_summary_empty(self, capsys) -> None:
        reporter = FleetReporter()
        reporter.print_summary()
        captured = capsys.readouterr()
        assert "FLEET REPORTER" in captured.out
        assert "Reports:" in captured.out

    def test_report_types_constant(self) -> None:
        assert "dashboard" in FleetReporter.REPORT_TYPES
        assert "api_docs" in FleetReporter.REPORT_TYPES
        assert "executive_summary" in FleetReporter.REPORT_TYPES

    def test_multiple_report_generations(self, tmp_path: Path) -> None:
        reporter = FleetReporter(output_dir=str(tmp_path / "reports"))
        reporter.generate_all_reports()
        reporter.generate_all_reports()
        stats = reporter.get_stats()
        assert stats["total_reports"] == 12
        assert stats["successful"] == 12

    def test_custom_output_dir(self, tmp_path: Path) -> None:
        custom_dir = tmp_path / "custom"
        reporter = FleetReporter(output_dir=str(custom_dir))
        result = reporter.generate_dashboard()
        assert custom_dir.exists()
        assert Path(result.output_path).exists()
