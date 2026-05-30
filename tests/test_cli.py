"""Tests for fleet CLI — argument parsing and command routing.

Covers parser setup, cmd_status, cmd_test, cmd_breed, cmd_report,
and main() entrypoint routing.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from fleet.cli import main


# ---------------------------------------------------------------------------
# Parser / Routing
# ---------------------------------------------------------------------------

class TestCLIParser:
    def test_no_command_prints_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code != 0
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "usage:" in output.lower()

    def test_status_command(self, capsys):
        with patch("fleet.cli.FleetHealthChecker") as mock_checker:
            mock_checker.return_value.check_all.return_value = []
            mock_checker.return_value.report.return_value = "OK"
            with patch.object(sys, "argv", ["sunset", "status"]):
                main()
            captured = capsys.readouterr()
            assert "OK" in captured.out or captured.out == ""

    def test_test_command(self, capsys):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            with patch.object(sys, "argv", ["sunset", "test"]):
                with pytest.raises(SystemExit) as exc:
                    main()
                assert exc.value.code == 0

    def test_breed_command(self, capsys):
        with patch("swarm.breeder_daemon_v2.BreederDaemonV2") as mock_breeder:
            mock_breeder.return_value.cycle.return_value = []
            with patch("fleet.cli.get_config") as mock_cfg:
                mock_cfg.return_value.breeding_pool_size = 10
                mock_cfg.return_value.generation_limit = 2
                mock_cfg.return_value.mutation_rate = 0.1
                mock_cfg.return_value.crossover_rate = 0.7
                mock_cfg.return_value.elitism = 0.05
                mock_cfg.return_value.latent_dim = 8
                mock_cfg.return_value.flux_weight_bounds = (-5.0, 5.0)
                mock_cfg.return_value.flux_max_l2_norm = 100.0
                mock_cfg.return_value.flux_max_variance = 10.0
                mock_cfg.return_value.flux_max_chaos = 1.0
                mock_cfg.return_value.flux_thermal_budget_gate = 0.8
                with patch.object(sys, "argv", ["sunset", "breed", "--pool", "5", "--generations", "2"]):
                    main()

    def test_report_breeding(self, capsys):
        with patch("fleet.deck.breeding_report") as mock_report:
            mock_report.return_value = "# Breeding Report\n"
            with patch.object(sys, "argv", ["sunset", "report", "--type", "breeding"]):
                main()
            captured = capsys.readouterr()
            assert "# Breeding Report" in captured.out

    def test_report_status(self, capsys):
        with patch("fleet.deck.fleet_status") as mock_report:
            mock_report.return_value = "# Status Report\n"
            with patch.object(sys, "argv", ["sunset", "report", "--type", "status"]):
                main()
            captured = capsys.readouterr()
            assert "# Status Report" in captured.out

    def test_report_flux(self, capsys):
        with patch("fleet.deck.flux_gate_decision") as mock_report:
            mock_report.return_value = "# FLUX Report\n"
            with patch.object(sys, "argv", ["sunset", "report", "--type", "flux", "--proofs", "123"]):
                main()
            captured = capsys.readouterr()
            assert "# FLUX Report" in captured.out

    def test_report_output_file(self, capsys, tmp_path):
        out = tmp_path / "report.md"
        with patch("fleet.deck.breeding_report") as mock_report:
            mock_report.return_value = "REPORT"
            with patch.object(sys, "argv", ["sunset", "report", "--type", "breeding", "--output", str(out)]):
                main()
            assert out.read_text() == "REPORT"

    def test_report_unknown_type(self):
        with patch.object(sys, "argv", ["sunset", "report", "--type", "unknown"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 2  # argparse exits with 2 for invalid choice

    def test_format_json(self, capsys):
        with patch("fleet.cli.FleetHealthChecker") as mock_checker:
            mock_checker.return_value.check_all.return_value = []
            mock_checker.return_value.report.return_value = "{}"
            with patch.object(sys, "argv", ["sunset", "--format", "json", "status"]):
                main()
            captured = capsys.readouterr()
            assert "{}" in captured.out

    def test_status_fail(self):
        with patch("fleet.cli.FleetHealthChecker") as mock_checker:
            result = MagicMock()
            result.ok = False
            mock_checker.return_value.check_all.return_value = [result]
            mock_checker.return_value.report.return_value = "DOWN"
            with patch.object(sys, "argv", ["sunset", "status", "--fail"]):
                with pytest.raises(SystemExit) as exc:
                    main()
                assert exc.value.code == 1
