"""FleetSecurityScan — Python adapter for Pringled/agentcheck.

Wraps the ``agentcheck`` CLI (Go binary) to produce Python-native
``RiskReport`` objects.  When ``agentcheck`` is not installed a
pure-Python fallback scanner checks the most common fleet secrets.

Integrates with ``OperationalTrap`` to raise alerts when CRITICAL or
HIGH findings are discovered.

Reference: https://github.com/Pringled/agentcheck
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import logging

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data models (mirror agentcheck JSON schema)
# ═══════════════════════════════════════════════════════════════════════


class Severity:
    UNCERTAIN = "UNCERTAIN"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    _RANK = {
        UNCERTAIN: -1,
        LOW: 1,
        MODERATE: 2,
        HIGH: 3,
        CRITICAL: 4,
    }

    @classmethod
    def rank(cls, sev: str) -> int:
        return cls._RANK.get(sev, -1)


@dataclass
class Finding:
    scanner: str
    resource: str
    severity: str
    description: str
    detail: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        return cls(
            scanner=d.get("scanner", ""),
            resource=d.get("resource", ""),
            severity=d.get("severity", Severity.UNCERTAIN),
            description=d.get("description", ""),
            detail=d.get("detail", ""),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ScanResult:
    scanner_name: str
    findings: List[Finding] = field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "ScanResult":
        return cls(
            scanner_name=d.get("scanner", ""),
            findings=[Finding.from_dict(f) for f in d.get("findings", [])],
            skipped=d.get("skipped", False),
            skip_reason=d.get("skip_reason"),
        )


@dataclass
class Summary:
    critical: int = 0
    high: int = 0
    moderate: int = 0
    low: int = 0
    uncertain: int = 0
    confirmed_total: int = 0
    scanners_total: int = 0
    scanners_skipped: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "Summary":
        return cls(
            critical=d.get("critical", 0),
            high=d.get("high", 0),
            moderate=d.get("moderate", 0),
            low=d.get("low", 0),
            uncertain=d.get("uncertain", 0),
            confirmed_total=d.get("confirmed_total", 0),
            scanners_total=d.get("scanners_total", 0),
            scanners_skipped=d.get("scanners_skipped", 0),
        )


@dataclass
class RiskReport:
    summary: Summary
    scan_results: List[ScanResult]

    @classmethod
    def from_dict(cls, d: dict) -> "RiskReport":
        return cls(
            summary=Summary.from_dict(d.get("summary", {})),
            scan_results=[ScanResult.from_dict(r) for r in d.get("scan_results", [])],
        )

    def findings_above(self, severity: str) -> List[Finding]:
        """Return findings with severity >= given threshold."""
        threshold = Severity.rank(severity)
        out: List[Finding] = []
        for r in self.scan_results:
            for f in r.findings:
                if Severity.rank(f.severity) >= threshold >= 0:
                    out.append(f)
        return out

    def to_dict(self) -> dict:
        return {
            "summary": dataclasses.asdict(self.summary),
            "scan_results": [
                {
                    "scanner": r.scanner_name,
                    "findings": [f.to_dict() for f in r.findings],
                    "skipped": r.skipped,
                    "skip_reason": r.skip_reason,
                }
                for r in self.scan_results
            ],
        }


# ═══════════════════════════════════════════════════════════════════════
# Core adapter
# ═══════════════════════════════════════════════════════════════════════


class FleetSecurityScanner:
    """Python wrapper for Pringled/agentcheck.

    Usage::

        scanner = FleetSecurityScanner()
        report = scanner.scan()
        if report.summary.critical > 0:
            operational_trap.trigger("security", report)
    """

    def __init__(
        self,
        agentcheck_path: Optional[str] = None,
        fail_on: str = Severity.HIGH,
        extra_env_keys: Optional[List[str]] = None,
        extra_credential_files: Optional[List[str]] = None,
    ):
        self._path = agentcheck_path or self._find_agentcheck()
        self.fail_on = fail_on
        self.extra_env_keys = extra_env_keys or []
        self.extra_credential_files = extra_credential_files or []

    # ── Discovery ────────────────────────────────────────────────────

    @staticmethod
    def _find_agentcheck() -> Optional[str]:
        return shutil.which("agentcheck")

    def has_agentcheck(self) -> bool:
        return self._path is not None

    # ── Scan ───────────────────────────────────────────────────────────

    def scan(self) -> RiskReport:
        """Run a security scan.

        Uses the real ``agentcheck`` binary when available; otherwise falls
        back to the pure-Python fleet scanner.
        """
        if self.has_agentcheck():
            return self._scan_with_agentcheck()
        log.warning("agentcheck not found; using fallback Python scanner")
        return self._scan_fallback()

    def _scan_with_agentcheck(self) -> RiskReport:
        cmd = [self._path, "--json"]  # type: ignore[list-item]
        if self.fail_on:
            cmd += ["--fail-on", self.fail_on.lower()]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            data = json.loads(proc.stdout)
            return RiskReport.from_dict(data)
        except subprocess.TimeoutExpired:
            log.error("agentcheck scan timed out")
            return self._empty_report()
        except json.JSONDecodeError as exc:
            log.error("agentcheck output was not valid JSON: %s", exc)
            return self._empty_report()

    def _empty_report(self) -> RiskReport:
        return RiskReport(summary=Summary(), scan_results=[])

    # ── Fallback scanner (pure Python) ───────────────────────────────

    def _scan_fallback(self) -> RiskReport:
        """Pure-Python scanner covering fleet-relevant secrets."""
        findings: List[Finding] = []

        # API keys in environment
        findings.extend(self._scan_env_keys())

        # Credential files on disk
        findings.extend(self._scan_credential_files())

        # Kubernetes context
        findings.extend(self._scan_kubernetes())

        # Docker socket access
        findings.extend(self._scan_docker())

        # SSH keys
        findings.extend(self._scan_ssh_keys())

        # Terraform state
        findings.extend(self._scan_terraform())

        # .env files
        findings.extend(self._scan_env_files())

        # Fleet-specific checks
        findings.extend(self._scan_fleet_specific())

        summary = self._summarize(findings)
        result = ScanResult(
            scanner_name="fleet_fallback",
            findings=findings,
            skipped=False,
        )
        return RiskReport(summary=summary, scan_results=[result])

    # ── Fallback checks ──────────────────────────────────────────────

    _HIGH_RISK_ENV = re.compile(
        r"^(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AZURE_.*_KEY|GCP_.*_KEY|"
        r"OPENAI_API_KEY|ANTHROPIC_API_KEY|GITHUB_TOKEN|GITLAB_TOKEN|"
        r"STRIPE_.*_KEY|SLACK_.*_TOKEN|TWILIO_.*_KEY|SENDGRID_.*_KEY|"
        r"DOCKER_.*_KEY|KUBECONFIG|VAULT_TOKEN|PGPASSWORD|MYSQL_PWD)$",
        re.IGNORECASE,
    )

    def _scan_env_keys(self) -> List[Finding]:
        out: List[Finding] = []
        for key, val in os.environ.items():
            if self._HIGH_RISK_ENV.match(key):
                out.append(Finding(
                    scanner="env_keys",
                    resource=key,
                    severity=Severity.HIGH,
                    description=f"High-risk API key found in environment: {key}",
                ))
            elif any(k.lower() in key.lower() for k in self.extra_env_keys):
                out.append(Finding(
                    scanner="env_keys",
                    resource=key,
                    severity=Severity.HIGH,
                    description=f"Custom high-risk env key found: {key}",
                ))
        return out

    _CREDENTIAL_PATHS = [
        "~/.aws/credentials",
        "~/.aws/config",
        "~/.config/gcloud/application_default_credentials.json",
        "~/.azure/azureProfile.json",
        "~/.docker/config.json",
        "~/.ssh/id_rsa",
        "~/.ssh/id_ed25519",
        "~/.netrc",
        "~/.pgpass",
    ]

    def _scan_credential_files(self) -> List[Finding]:
        out: List[Finding] = []
        for p in self._CREDENTIAL_PATHS + (self.extra_credential_files or []):
            path = Path(p).expanduser()
            if path.exists():
                out.append(Finding(
                    scanner="credential_files",
                    resource=str(path),
                    severity=Severity.MODERATE,
                    description=f"Credential file exists: {path}",
                ))
        return out

    def _scan_kubernetes(self) -> List[Finding]:
        out: List[Finding] = []
        kubeconfig = os.environ.get("KUBECONFIG", "~/.kube/config")
        path = Path(kubeconfig).expanduser()
        if path.exists():
            out.append(Finding(
                scanner="kubernetes",
                resource=str(path),
                severity=Severity.MODERATE,
                description="Kubernetes config file found",
            ))
            # Check if current context points to prod
            try:
                import yaml
                with open(path) as f:
                    cfg = yaml.safe_load(f)
                ctx = cfg.get("current-context", "")
                if "prod" in ctx.lower() or "production" in ctx.lower():
                    out.append(Finding(
                        scanner="kubernetes",
                        resource=ctx,
                        severity=Severity.CRITICAL,
                        description=f"kubectl context points to production cluster: {ctx}",
                    ))
            except Exception:
                pass
        return out

    def _scan_docker(self) -> List[Finding]:
        out: List[Finding] = []
        if Path("/var/run/docker.sock").exists():
            out.append(Finding(
                scanner="docker",
                resource="/var/run/docker.sock",
                severity=Severity.MODERATE,
                description="Docker socket accessible — potential container escape",
            ))
        return out

    def _scan_ssh_keys(self) -> List[Finding]:
        out: List[Finding] = []
        ssh_dir = Path("~/.ssh").expanduser()
        if ssh_dir.exists():
            for key_file in ssh_dir.glob("id_*"):
                out.append(Finding(
                    scanner="ssh_keys",
                    resource=str(key_file),
                    severity=Severity.MODERATE,
                    description=f"SSH key found: {key_file.name}",
                ))
        return out

    def _scan_terraform(self) -> List[Finding]:
        out: List[Finding] = []
        for tfstate in Path(".").rglob("*.tfstate"):
            out.append(Finding(
                scanner="terraform",
                resource=str(tfstate),
                severity=Severity.HIGH,
                description=f"Terraform state file found (may contain secrets): {tfstate}",
            ))
        return out

    def _scan_env_files(self) -> List[Finding]:
        out: List[Finding] = []
        for envfile in Path(".").rglob(".env"):
            out.append(Finding(
                scanner="env_files",
                resource=str(envfile),
                severity=Severity.MODERATE,
                description=f".env file found: {envfile}",
            ))
        return out

    def _scan_fleet_specific(self) -> List[Finding]:
        """Checks specific to the Cocapn Fleet / sunset-ecosystem."""
        out: List[Finding] = []

        # Check for unencrypted fleet identity keys
        fleet_keys = [
            "~/.openclaw/identity.key",
            "~/.openclaw/secrets.json",
            "~/.openclaw/.env",
        ]
        for p in fleet_keys:
            path = Path(p).expanduser()
            if path.exists():
                out.append(Finding(
                    scanner="fleet_specific",
                    resource=str(path),
                    severity=Severity.HIGH,
                    description=f"Fleet secret file found without encryption check: {path}",
                ))

        # Check for exposed PLATO tokens
        for key in os.environ:
            if "PLATO" in key and "TOKEN" in key:
                out.append(Finding(
                    scanner="fleet_specific",
                    resource=key,
                    severity=Severity.CRITICAL,
                    description=f"PLATO token exposed in environment: {key}",
                ))

        # Check for hardcoded credentials in sunset-ecosystem code
        sunset_dir = Path(__file__).parent.parent
        if sunset_dir.exists():
            for pyfile in sunset_dir.rglob("*.py"):
                try:
                    content = pyfile.read_text(errors="ignore")
                    if re.search(r'password\s*=\s*["\'][^"\']+["\']', content, re.IGNORECASE):
                        out.append(Finding(
                            scanner="fleet_specific",
                            resource=str(pyfile),
                            severity=Severity.HIGH,
                            description=f"Possible hardcoded password in {pyfile.name}",
                        ))
                except Exception:
                    pass

        return out

    # ── Utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _summarize(findings: List[Finding]) -> Summary:
        s = Summary()
        for f in findings:
            if f.severity == Severity.CRITICAL:
                s.critical += 1
            elif f.severity == Severity.HIGH:
                s.high += 1
            elif f.severity == Severity.MODERATE:
                s.moderate += 1
            elif f.severity == Severity.LOW:
                s.low += 1
            else:
                s.uncertain += 1
        s.confirmed_total = s.critical + s.high + s.moderate + s.low
        s.scanners_total = 1
        return s

    # ── OperationalTrap integration ────────────────────────────────────

    def to_operational_trap_payload(self, report: RiskReport) -> Dict[str, Any]:
        """Convert a RiskReport into an OperationalTrap-compatible dict."""
        critical = report.findings_above(Severity.CRITICAL)
        high = report.findings_above(Severity.HIGH)
        return {
            "trap_type": "security_scan",
            "severity": Severity.CRITICAL if critical else (Severity.HIGH if high else Severity.LOW),
            "details": {
                "critical_count": report.summary.critical,
                "high_count": report.summary.high,
                "moderate_count": report.summary.moderate,
                "findings": [f.to_dict() for f in critical + high],
            },
            "recommendation": (
                "CRITICAL: Immediate action required. Review findings above."
                if critical
                else (
                    "HIGH: Review exposed credentials and rotate keys."
                    if high
                    else "No critical security findings."
                )
            ),
        }
