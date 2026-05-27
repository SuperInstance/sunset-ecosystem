# Production Grade Audit — sunset-ecosystem

**Date:** 2026-05-27
**Current state:** 2065 tests passing, CI exists but has `|| true`

## CRITICAL (fix first)

| # | Gap | Where | Impact |
|---|-----|-------|--------|
| 1 | CI passes even if tests fail | `.github/workflows/ci.yml:7` | `pytest || true` = green on failure |
| 2 | No coverage threshold | No coverage in CI | Silent regressions possible |
| 3 | Alpha status | `pyproject.toml:21` | PyPI won't take seriously |
| 4 | No type checking | No mypy in CI | Runtime errors in production |

## HIGH

| # | Gap | Where | Impact |
|---|-----|-------|--------|
| 5 | Hardcoded fleet IP | `nexus/federation.py:31` | Can't deploy to other hosts |
| 6 | No linting/formatting | No ruff/black in CI | Code quality drifts |
| 7 | No pre-commit hooks | Missing `.pre-commit-config.yaml` | Bad commits get through |
| 8 | No Docker | Missing `Dockerfile` | Can't containerize deploy |
| 9 | Benchmarks not gated | `tests/benchmarks/` exists but CI ignores | Perf regressions undetected |
| 10 | No security scan | No `bandit` or `pip-audit` | CVEs in deps undetected |

## MEDIUM

| # | Gap | Where | Impact |
|---|-----|-------|--------|
| 11 | Missing CHANGELOG | No `CHANGELOG.md` | Users can't track releases |
| 12 | Missing CONTRIBUTING | No `CONTRIBUTING.md` | External contributors lost |
| 13 | No health check endpoint | No `/health` in any server | Load balancers can't verify |
| 14 | No release workflow | No `.github/workflows/release.yml` | Manual PyPI pushes |

## Production Grade Definition

A repo is production grade when:
1. **CI blocks bad code** — tests, coverage, lint, type check all pass
2. **Deployable anywhere** — Docker + env-configurable, not hardcoded IPs
3. **Observable** — health checks, metrics, structured logging
4. **Secure** — dependency scanning, no secrets in code
5. **Documented** — setup, deployment, monitoring, troubleshooting
6. **Releasable** — versioned, changelog, automated releases
