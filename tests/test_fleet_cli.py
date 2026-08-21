"""Tests for FleetCLI — unified command-line interface.

Reference: fleet/fleet_cli.py
"""

from __future__ import annotations

import pytest

from fleet.fleet_cli import CLIResult, FleetCLI


class TestCLIResult:
    def test_fields(self) -> None:
        result = CLIResult(
            command="test",
            success=True,
            message="ok",
            data={"key": "value"},
            duration_ms=100.0,
        )
        assert result.command == "test"
        assert result.success is True
        assert result.duration_ms == 100.0


class TestFleetCLI:
    def test_init(self) -> None:
        cli = FleetCLI()
        assert cli.workspace.exists()

    def test_beat(self) -> None:
        cli = FleetCLI()
        result = cli.beat(count=1)
        assert result.success is True
        assert result.command == "beat"
        assert result.data is not None
        assert result.data["beats"] == 1

    def test_beat_multiple(self) -> None:
        cli = FleetCLI()
        result = cli.beat(count=3)
        assert result.success is True
        assert result.data["beats"] == 3

    def test_health(self) -> None:
        cli = FleetCLI()
        result = cli.health()
        assert result.success is True
        assert result.command == "health"
        assert result.data is not None
        assert "status" in result.data
        assert "healthy" in result.data

    def test_health_structure(self) -> None:
        cli = FleetCLI()
        result = cli.health()
        data = result.data
        assert isinstance(data["healthy"], int)
        assert isinstance(data["warning"], int)
        assert isinstance(data["critical"], int)
        assert isinstance(data["total"], int)
        assert isinstance(data["issues"], list)

    def test_modules(self) -> None:
        cli = FleetCLI()
        result = cli.modules()
        assert result.success is True
        assert result.command == "modules"
        assert result.data is not None
        assert "modules" in result.data
        assert len(result.data["modules"]) == 20

    def test_modules_stats(self) -> None:
        cli = FleetCLI()
        result = cli.modules(show_stats=True)
        assert result.success is True

    def test_status(self) -> None:
        cli = FleetCLI()
        result = cli.status()
        assert result.success is True
        assert result.command == "status"
        assert result.data is not None
        assert "status" in result.data
        assert "modules" in result.data
        assert "tests" in result.data

    def test_status_structure(self) -> None:
        cli = FleetCLI()
        result = cli.status()
        data = result.data
        assert isinstance(data["modules"], int)
        assert isinstance(data["tests"], int)
        assert isinstance(data["healthy"], int)
        assert isinstance(data["warning"], int)
        assert isinstance(data["critical"], int)

    def test_run_args_beat(self) -> None:
        cli = FleetCLI()
        result = cli.run_args(["beat", "--count", "2"])
        assert result.success is True
        assert result.data["beats"] == 2

    def test_run_args_health(self) -> None:
        cli = FleetCLI()
        result = cli.run_args(["health"])
        assert result.command == "health"
        assert result.success is True

    def test_run_args_modules(self) -> None:
        cli = FleetCLI()
        result = cli.run_args(["modules"])
        assert result.success is True
        assert "modules" in result.data

    def test_run_args_status(self) -> None:
        cli = FleetCLI()
        result = cli.run_args(["status"])
        assert result.success is True

    def test_run_args_help(self) -> None:
        cli = FleetCLI()
        result = cli.run_args([])
        assert result.command == "help"
        assert result.success is True

    def test_print_result_success(self, capsys) -> None:
        result = CLIResult(
            command="test", success=True, message="ok", data={"key": "val"}
        )
        FleetCLI.print_result(result)
        captured = capsys.readouterr()
        assert "✅" in captured.out
        assert "test" in captured.out
        assert "ok" in captured.out

    def test_print_result_failure(self, capsys) -> None:
        result = CLIResult(command="test", success=False, message="failed")
        FleetCLI.print_result(result)
        captured = capsys.readouterr()
        assert "❌" in captured.out
        assert "failed" in captured.out

    def test_print_result_no_data(self, capsys) -> None:
        result = CLIResult(command="test", success=True, message="ok", data=None)
        FleetCLI.print_result(result)
        captured = capsys.readouterr()
        assert "✅" in captured.out

    def test_orchestrator_lazy_init(self) -> None:
        cli = FleetCLI()
        assert cli._orchestrator is None
        cli._ensure_orchestrator()
        assert cli._orchestrator is not None

    def test_harbor_lazy_init(self) -> None:
        cli = FleetCLI()
        assert cli._harbor is None
        cli._ensure_harbor()
        assert cli._harbor is not None

    def test_dashboard_lazy_init(self) -> None:
        cli = FleetCLI()
        assert cli._dashboard is None
        cli._ensure_dashboard()
        assert cli._dashboard is not None

    def test_reporter_lazy_init(self) -> None:
        cli = FleetCLI()
        assert cli._reporter is None
        cli._ensure_reporter()
        assert cli._reporter is not None

    def test_metrics_lazy_init(self) -> None:
        cli = FleetCLI()
        assert cli._metrics is None
        cli._ensure_metrics()
        assert cli._metrics is not None

    def test_beat_duration(self) -> None:
        cli = FleetCLI()
        result = cli.beat()
        assert result.duration_ms > 0

    def test_health_duration(self) -> None:
        cli = FleetCLI()
        result = cli.health()
        assert result.duration_ms > 0
