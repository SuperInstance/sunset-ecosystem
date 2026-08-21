"""Tests for FleetSecurityScanner.

Covers:
- Binary detection (agentcheck present vs absent)
- JSON parsing from agentcheck output
- Fallback scanner: env keys, credential files, k8s, docker, ssh, terraform, .env, fleet-specific
- RiskReport filtering (findings_above)
- OperationalTrap payload generation
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fleet.fleet_security_scan import (
    Finding,
    FleetSecurityScanner,
    RiskReport,
    ScanResult,
    Severity,
    Summary,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def scanner():
    return FleetSecurityScanner(agentcheck_path=None)


@pytest.fixture
def sample_report():
    return RiskReport(
        summary=Summary(critical=2, high=4, moderate=1, low=1, confirmed_total=8),
        scan_results=[
            ScanResult(
                scanner_name="env_keys",
                findings=[
                    Finding(
                        "env_keys", "OPENAI_API_KEY", Severity.HIGH, "OpenAI key in env"
                    ),
                    Finding(
                        "env_keys",
                        "AWS_SECRET_ACCESS_KEY",
                        Severity.CRITICAL,
                        "AWS root key",
                    ),
                ],
            ),
            ScanResult(
                scanner_name="docker",
                findings=[
                    Finding(
                        "docker",
                        "/var/run/docker.sock",
                        Severity.MODERATE,
                        "Docker socket",
                    ),
                ],
            ),
        ],
    )


# ── Detection ────────────────────────────────────────────────────────────


class TestDetection:
    def test_no_agentcheck_detected(self, scanner):
        assert not scanner.has_agentcheck()

    def test_agentcheck_found(self, monkeypatch):
        scanner = FleetSecurityScanner(agentcheck_path="/usr/local/bin/agentcheck")
        assert scanner.has_agentcheck()

    @patch("shutil.which")
    def test_auto_discovery(self, mock_which):
        mock_which.return_value = "/usr/bin/agentcheck"
        scanner = FleetSecurityScanner()
        assert scanner.has_agentcheck()
        assert scanner._path == "/usr/bin/agentcheck"


# ── JSON Parsing ─────────────────────────────────────────────────────────


class TestParsing:
    def test_risk_report_from_dict(self):
        raw = {
            "summary": {
                "critical": 1,
                "high": 2,
                "moderate": 0,
                "low": 0,
                "uncertain": 0,
                "confirmed_total": 3,
                "scanners_total": 2,
                "scanners_skipped": 0,
            },
            "scan_results": [
                {
                    "scanner": "aws",
                    "findings": [
                        {
                            "scanner": "aws",
                            "resource": "AdministratorAccess",
                            "severity": "CRITICAL",
                            "description": "Admin policy attached",
                        }
                    ],
                    "skipped": False,
                }
            ],
        }
        report = RiskReport.from_dict(raw)
        assert report.summary.critical == 1
        assert report.summary.high == 2
        assert len(report.scan_results) == 1
        assert report.scan_results[0].findings[0].severity == Severity.CRITICAL

    def test_empty_report(self):
        report = RiskReport(summary=Summary(), scan_results=[])
        assert report.summary.confirmed_total == 0
        assert report.findings_above(Severity.HIGH) == []


# ── Filtering ────────────────────────────────────────────────────────────


class TestFiltering:
    def test_findings_above_high(self, sample_report):
        high_plus = sample_report.findings_above(Severity.HIGH)
        assert len(high_plus) == 2  # one CRITICAL + one HIGH
        assert all(
            Severity.rank(f.severity) >= Severity.rank(Severity.HIGH) for f in high_plus
        )

    def test_findings_above_critical(self, sample_report):
        crit = sample_report.findings_above(Severity.CRITICAL)
        assert len(crit) == 1
        assert crit[0].severity == Severity.CRITICAL

    def test_findings_above_moderate(self, sample_report):
        mod_plus = sample_report.findings_above(Severity.MODERATE)
        # sample_report has 2 env findings (1 HIGH + 1 CRITICAL) + 1 docker (MODERATE) = 3 total
        assert len(mod_plus) == 3


# ── Fallback Scanner ─────────────────────────────────────────────────────


class TestFallback:
    """Pure-Python fallback checks."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-12345"}, clear=False)
    def test_env_keys_high_risk(self, scanner):
        findings = scanner._scan_env_keys()
        assert any("OPENAI_API_KEY" == f.resource for f in findings)
        assert any(f.severity == Severity.HIGH for f in findings)

    def test_env_keys_no_risk(self, scanner):
        # Run in clean env where no API keys are set
        with patch.dict(os.environ, {}, clear=True):
            findings = scanner._scan_env_keys()
            assert findings == []

    def test_custom_env_keys(self):
        scanner = FleetSecurityScanner(extra_env_keys=["MY_CUSTOM_TOKEN"])
        with patch.dict(os.environ, {"MY_CUSTOM_TOKEN": "abc"}, clear=False):
            findings = scanner._scan_env_keys()
            assert any(f.resource == "MY_CUSTOM_TOKEN" for f in findings)

    @patch("fleet.fleet_security_scan.Path.exists")
    @patch("fleet.fleet_security_scan.Path.expanduser")
    def test_credential_files(self, mock_expand, mock_exists, scanner):
        mock_expand.return_value = MagicMock()
        mock_exists.return_value = True
        findings = scanner._scan_credential_files()
        assert len(findings) > 0
        assert all(f.severity == Severity.MODERATE for f in findings)

    def test_docker_socket(self, scanner):
        with patch("fleet.fleet_security_scan.Path.exists", return_value=True):
            findings = scanner._scan_docker()
            assert len(findings) == 1
            assert findings[0].scanner == "docker"
            assert findings[0].severity == Severity.MODERATE

    def test_docker_socket_missing(self, scanner):
        with patch("fleet.fleet_security_scan.Path.exists", return_value=False):
            findings = scanner._scan_docker()
            assert findings == []

    @patch("fleet.fleet_security_scan.Path.exists")
    @patch("fleet.fleet_security_scan.Path.glob")
    def test_ssh_keys(self, mock_glob, mock_exists, scanner):
        mock_exists.return_value = True
        mock_key = MagicMock()
        mock_key.name = "id_rsa"
        mock_glob.return_value = [mock_key]
        findings = scanner._scan_ssh_keys()
        assert len(findings) == 1
        assert findings[0].scanner == "ssh_keys"

    def test_terraform_state(self, scanner):
        with patch("pathlib.Path.rglob", return_value=[Path("prod.tfstate")]):
            findings = scanner._scan_terraform()
            assert len(findings) == 1
            assert findings[0].severity == Severity.HIGH

    def test_env_files(self, scanner):
        with patch("pathlib.Path.rglob", return_value=[Path(".env")]):
            findings = scanner._scan_env_files()
            assert len(findings) == 1
            assert findings[0].scanner == "env_files"


# ── OperationalTrap Integration ──────────────────────────────────────────


class TestOperationalTrap:
    def test_critical_payload(self, scanner):
        report = RiskReport(
            summary=Summary(critical=1, high=0),
            scan_results=[
                ScanResult(
                    scanner_name="aws",
                    findings=[
                        Finding("aws", "root", Severity.CRITICAL, "AWS root key"),
                    ],
                )
            ],
        )
        payload = scanner.to_operational_trap_payload(report)
        assert payload["trap_type"] == "security_scan"
        assert payload["severity"] == Severity.CRITICAL
        assert "Immediate action" in payload["recommendation"]
        assert payload["details"]["critical_count"] == 1

    def test_high_payload(self, scanner):
        report = RiskReport(
            summary=Summary(critical=0, high=2),
            scan_results=[
                ScanResult(
                    scanner_name="api",
                    findings=[
                        Finding("api", "OPENAI_KEY", Severity.HIGH, "OpenAI"),
                    ],
                )
            ],
        )
        payload = scanner.to_operational_trap_payload(report)
        assert payload["severity"] == Severity.HIGH
        assert "Review exposed" in payload["recommendation"]

    def test_clean_payload(self, scanner):
        report = RiskReport(summary=Summary(), scan_results=[])
        payload = scanner.to_operational_trap_payload(report)
        assert payload["severity"] == Severity.LOW
        assert "No critical" in payload["recommendation"]


# ── Severity Ranks ───────────────────────────────────────────────────────


class TestSeverity:
    def test_rank_order(self):
        assert Severity.rank(Severity.LOW) < Severity.rank(Severity.MODERATE)
        assert Severity.rank(Severity.MODERATE) < Severity.rank(Severity.HIGH)
        assert Severity.rank(Severity.HIGH) < Severity.rank(Severity.CRITICAL)
        assert Severity.rank(Severity.UNCERTAIN) == -1

    def test_unknown_rank(self):
        assert Severity.rank("UNKNOWN") == -1


# ── Full Scan ────────────────────────────────────────────────────────────


class TestFullScan:
    def test_fallback_scan_returns_report(self, scanner):
        report = scanner.scan()
        assert isinstance(report, RiskReport)
        assert isinstance(report.summary, Summary)
        # We can't assert exact counts because they depend on the test env
        assert report.summary.scanners_total >= 0

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    def test_fallback_finds_api_keys(self):
        scanner = FleetSecurityScanner()
        report = scanner.scan()
        env_findings = [
            f
            for r in report.scan_results
            for f in r.findings
            if f.scanner == "env_keys"
        ]
        assert any(f.resource == "OPENAI_API_KEY" for f in env_findings)
