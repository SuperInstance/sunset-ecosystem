# Contributing to sunset-ecosystem

Thank you for your interest in the Cocapn Fleet. This document covers how to set up the project, run tests, and submit changes.

## Setup

```bash
git clone https://github.com/SuperInstance/sunset-ecosystem.git
cd sunset-ecosystem
pip install -e ".[dev]"
```

## Running Tests

```bash
# Full suite (takes ~5 minutes)
python -m pytest tests/ -q

# With coverage
python -m pytest tests/ --cov=sunset --cov-report=term-missing

# Single file
python -m pytest tests/test_breeder_daemon_v2.py -v
```

## Lint & Type Check

```bash
ruff check sunset/ swarm/ nexus/ logos/
mypy sunset/ --ignore-missing-imports
```

## Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Submitting Changes

1. **Branch**: Create a feature branch (`git checkout -b feature/name`)
2. **Tests**: All tests must pass. New features need new tests.
3. **Coverage**: Maintain or improve coverage (current threshold: 75%)
4. **Commit**: Use conventional commits (`feat:`, `fix:`, `docs:`, `test:`)
5. **PR**: Open against `main` with a clear description

## Code Style

- **Ruff** for linting and formatting
- **MyPy** for type checking (ignore missing imports for external deps)
- Docstrings for public APIs
- Type hints on function signatures

## Architecture Notes

- **Trinity**: Every agent is scored by ethos × pathos × logos. Zero in any dimension = sunset.
- **Lifecycle**: EGG → COMPETE → (SURVIVE → BREED) or (SUNSET → ARCHIVE)
- **Modules**: `ethos/`, `pathos/`, `logos/`, `swarm/`, `nexus/`, `nerve/`, `sunset/`
- **Tests**: Mirror the source structure under `tests/`

## Security

- Never commit secrets, API keys, or private keys
- Run `bandit -r .` before submitting
- The CI runs `trufflehog` to catch leaked credentials

## Questions?

Open an issue or reach out in `#cocapn-build` on Matrix.
