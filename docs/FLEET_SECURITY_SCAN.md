# FleetSecurityScan — Python Adapter for agentcheck

## Overview

FleetSecurityScan wraps **[Pringled/agentcheck](https://github.com/Pringled/agentcheck)** (Go-based security scanner) to integrate security posture checks into the Cocapn Fleet's ``OperationalTrap`` system.

**What it does:** Scans the environment for cloud IAM credentials, API keys, Kubernetes configs, Docker sockets, SSH keys, Terraform state files, and fleet-specific secrets. Produces a ``RiskReport`` that OperationalTrap can turn into alerts.

**Why it matters:** When agents run autonomously for hours, they might inherit dangerous credentials from the host shell. FleetSecurityScan detects these risks before they become incidents.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 OperationalTrap                               │
│         (thermal / FLUX / crash / SECURITY)               │
├─────────────────────────────────────────────────────────────┤
│                     FleetSecurityScanner                      │
│                          │                                   │
│            ┌─────────────┴─────────────┐                   │
│            │                           │                   │
│     agentcheck --json            Fallback Python            │
│     (Go binary)                  scanner                    │
│            │                           │                   │
│            └─────────────┬─────────────┘                   │
│                          ▼                                   │
│                    RiskReport                                │
│                    - Summary                                 │
│                    - Findings[]                              │
│                    - Severity ranks                          │
└─────────────────────────────────────────────────────────────┘
```

## Severity Model (mirrors agentcheck)

| Level | Meaning | Examples |
|-------|---------|----------|
| **CRITICAL** | Unrestricted, unconstrained access | AWS root key, prod kubectl context, PLATO token |
| **HIGH** | Scoped but dangerous access | Admin IAM policy, API keys in env, Terraform state |
| **MODERATE** | Lateral movement possible | Docker socket, SSH keys, .env files |
| **LOW** | Authenticated but low permissions | Active GCP account with no roles |
| **UNCERTAIN** | Check timed out, unknown risk | IAM policy retrieval failure |

## Usage

### Basic Scan

```python
from fleet.fleet_security_scan import FleetSecurityScanner

scanner = FleetSecurityScanner()
report = scanner.scan()

print(f"Critical: {report.summary.critical}")
print(f"High:     {report.summary.high}")
print(f"Moderate: {report.summary.moderate}")

for finding in report.findings_above("HIGH"):
    print(f"[{finding.severity}] {finding.scanner}: {finding.description}")
```

### OperationalTrap Integration

```python
from fleet.operational_trap import OperationalTrap
from fleet.fleet_security_scan import FleetSecurityScanner

trap = OperationalTrap(node_id="n0")
scanner = FleetSecurityScanner()

report = scanner.scan()
payload = scanner.to_operational_trap_payload(report)

# Trigger trap if CRITICAL or HIGH findings exist
trap.trigger(
    condition_type="security_scan",
    details=payload["details"],
    recommendation=payload["recommendation"],
)
```

### With Custom Checks

```python
scanner = FleetSecurityScanner(
    extra_env_keys=["CORP_INTERNAL_API_KEY", "MY_SERVICE_TOKEN"],
    extra_credential_files=["~/.config/mycorp/token"],
    fail_on="MODERATE",  # stricter CI threshold
)
report = scanner.scan()
```

### Using Real agentcheck Binary

```bash
# Install agentcheck (macOS/Linux)
brew tap Pringled/tap
brew install agentcheck

# Or build from source
go install github.com/Pringled/agentcheck@latest
```

Once ``agentcheck`` is on ``$PATH``, ``FleetSecurityScanner`` automatically uses it:

```python
scanner = FleetSecurityScanner()  # auto-discovers binary
report = scanner.scan()           # runs agentcheck --json
```

## API Reference

### FleetSecurityScanner

| Method | Description |
|--------|-------------|
| `scan()` | Run security scan (agentcheck or fallback) |
| `has_agentcheck()` | Bool: is the Go binary available? |
| `findings_above(severity)` | Filter findings by severity threshold |
| `to_operational_trap_payload(report)` | Convert to OperationalTrap dict |

### RiskReport

| Attribute | Type | Description |
|-----------|------|-------------|
| `summary` | Summary | Counts by severity |
| `scan_results` | List[ScanResult] | Per-scanner findings |

### Finding

| Field | Description |
|-------|-------------|
| `scanner` | Which scanner found it (aws, gcp, env_keys, docker...) |
| `resource` | Key name, file path, or context name |
| `severity` | CRITICAL / HIGH / MODERATE / LOW / UNCERTAIN |
| `description` | Human-readable explanation |
| `detail` | Optional extra context |

## Fallback Scanner (Pure Python)

When ``agentcheck`` is not installed, FleetSecurityScanner runs its own checks:

| Check | What it looks for | Severity |
|-------|------------------|----------|
| **env_keys** | 20+ high-risk env vars (AWS, OpenAI, GitHub, Stripe, etc.) | HIGH |
| **credential_files** | ~/.aws/credentials, ~/.docker/config.json, etc. | MODERATE |
| **kubernetes** | ~/.kube/config, prod context detection | MODERATE → CRITICAL |
| **docker** | /var/run/docker.sock accessibility | MODERATE |
| **ssh_keys** | ~/.ssh/id_rsa, id_ed25519 | MODERATE |
| **terraform** | *.tfstate files (contain secrets) | HIGH |
| **env_files** | .env files in working directory | MODERATE |
| **fleet_specific** | ~/.openclaw/identity.key, PLATO tokens, hardcoded passwords | HIGH → CRITICAL |

## Integration Points

| System | Integration |
|--------|-------------|
| **OperationalTrap** | `to_operational_trap_payload()` → `trap.trigger("security_scan", ...)` |
| **FleetConductorV2** | Add `security_beat()` tick that runs scan every N cycles |
| **BetaTestPersonas** | Security Auditor persona checks for exposed credentials |
| **CI/CD** | `agentcheck --ci --fail-on high` blocks pipeline on risky env |

## Test Summary

24 tests covering:
- Binary detection (auto-discovery, explicit path)
- JSON parsing from agentcheck output
- RiskReport filtering by severity threshold
- Fallback scanner: env keys, custom keys, credential files, docker, ssh, terraform, .env, fleet-specific
- OperationalTrap payload generation (critical, high, clean)
- Severity rank ordering

Run: `pytest tests/test_fleet_security_scan.py -v`

## Fork Status

**Upstream:** https://github.com/Pringled/agentcheck
**Our fork:** Not yet created (Go binary adapter approach used instead)
**Reason:** agentcheck is a CLI tool; wrapping via subprocess + JSON is more maintainable than rewriting in Python. If we need deep integration, we can fork later to add fleet-specific scanners.

## References

- **agentcheck**: https://github.com/Pringled/agentcheck
- **Severity policy**: See agentcheck README for full CI threshold docs
- **Config file**: `~/.agentcheck.yaml` for personal defaults, `.agentcheck.yaml` for project-level
