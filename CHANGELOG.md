# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Production-grade CI: coverage threshold (75%), mypy, ruff, caching
- Security scanning workflow (bandit, pip-audit, trufflehog)
- Automated release workflow (PyPI + GitHub releases on tag push)
- Pre-commit hooks (ruff, mypy, trailing-whitespace, private-key detection)
- Dockerfile with health check and non-root user
- Health endpoint (`/health`) on A2A server
- Environment-configurable nexus IP (`NEXUS_IP`, `NEXUS_PORT`)
- 2065 passing tests across 20+ modules

### Changed
- `Development Status` bumped from Alpha to Beta in `pyproject.toml`
- CI no longer ignores test failures (`|| true` removed)

### Fixed
- `bisect.insort` for `generation_index` ordering in WAL query
- Missing `_parse_iso8601()` in `wal_index.py`
- `genealogy()` double-counting in `test_breeder_daemon_v2.py`
- Broken `test_wal_query_index.py` rewritten for actual `WALIndex` API
- Duplicate `attach_flux_gating()` and duplicate import block in `breeder_daemon_v2.py`

## [0.1.0] - 2024-05-24

### Added
- Trinity architecture (ethos × pathos × logos)
- Agent lifecycle FSM (EGG → COMPETE → SURVIVE → BREED → SUNSET → ARCHIVE)
- BreederDaemonV2 with thermal scheduling and diversity-aware parent selection
- FLUX constraint gating (Path A Python + Path B VM)
- FleetConductorV2 central orchestrator
- SenseDecideAct unifying framework
- Cross-node mesh vector tables (CRDT)
- Distributed metronome bridge with PID drift correction
- A2A agent identity with task negotiation
- Signed WAL with integrity verification
- SSE streaming dashboard for breeding progress
- Beta-test persona framework (7 simulated visitors)
- Hebbian mesh layer for diversity-aware routing
- Operational trap registry (thermal, FLUX, crash detection)
- Gateway pacing circuit breaker
- Opcode capability index
- Dispatch router with Two-Minute Test
- 19 modules, 484+ tests

[unreleased]: https://github.com/SuperInstance/sunset-ecosystem/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SuperInstance/sunset-ecosystem/releases/tag/v0.1.0
