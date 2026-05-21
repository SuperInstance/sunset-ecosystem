"""Repo Health Metrics — SPEC-REPO-METRIC §2 implementation.

Computes the five-component health score:
  Freshness 30 | Test Coverage 25 | Documentation 15 | Dependency Health 15 | Issue Hygiene 15
"""
from __future__ import annotations

__all__ = [
    "RepoHealthMetrics",
    "HealthScore",
    "run_health_check",
]

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class HealthScore:
    """Five-component health score with total and traffic-light."""

    freshness: float      # 30 points max
    test_coverage: float  # 25 points max
    documentation: float  # 15 points max
    dependency_health: float  # 15 points max
    issue_hygiene: float  # 15 points max

    @property
    def total(self) -> float:
        return round(
            self.freshness
            + self.test_coverage
            + self.documentation
            + self.dependency_health
            + self.issue_hygiene,
            1,
        )

    @property
    def traffic_light(self) -> str:
        """🟢 Green >=80, 🟡 Yellow 50-79, 🔴 Red <50."""
        t = self.total
        if t >= 80:
            return "green"
        if t >= 50:
            return "yellow"
        return "red"

    def to_dict(self) -> dict:
        return {
            "freshness": self.freshness,
            "test_coverage": self.test_coverage,
            "documentation": self.documentation,
            "dependency_health": self.dependency_health,
            "issue_hygiene": self.issue_hygiene,
            "total": self.total,
            "traffic_light": self.traffic_light,
        }


class RepoHealthMetrics:
    """Compute health scores for a local git repo."""

    # Score weights (per SPEC-REPO-METRIC §2)
    WEIGHTS = {
        "freshness": 30.0,
        "test_coverage": 25.0,
        "documentation": 15.0,
        "dependency_health": 15.0,
        "issue_hygiene": 15.0,
    }

    def __init__(self, repo_root: str | Path) -> None:
        self.root = Path(repo_root).resolve()

    # ── 1. Freshness (30 pts) ────────────────────────────────────

    def _freshness(self) -> float:
        """Days since last commit → score (linear decay, 0 at 90 days)."""
        try:
            days = self._days_since_last_commit()
        except Exception:
            days = 90  # unknown = worst case
        # Linear: 30 pts at 0 days, 0 pts at 90 days
        return max(0.0, 30.0 * (1.0 - days / 90.0))

    def _days_since_last_commit(self) -> float:
        import time

        cmd = ["git", "log", "-1", "--format=%ct"]
        ts = subprocess.check_output(cmd, cwd=self.root, text=True).strip()
        commit_time = int(ts)
        now = time.time()
        return (now - commit_time) / 86400.0

    # ── 2. Test Coverage (25 pts) ─────────────────────────────────

    def _test_coverage(self) -> float:
        """Estimate coverage from presence + size of test suite."""
        test_dir = self.root / "tests"
        if not test_dir.exists():
            return 0.0

        # Count test files
        test_files = list(test_dir.rglob("test_*.py")) + list(test_dir.rglob("*_test.py"))
        if not test_files:
            return 0.0

        # Count lines of test code
        test_lines = sum(len(f.read_text().splitlines()) for f in test_files)

        # Count source lines
        src_lines = 0
        for pkg in self._discover_packages():
            for py_file in pkg.rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue
                src_lines += len(py_file.read_text().splitlines())

        if src_lines == 0:
            return 0.0

        # Ratio heuristic: 1:1 test:src ≈ 25 pts, 1:4 ≈ 6 pts
        ratio = test_lines / src_lines
        # Asymptotic: 25 * (1 - exp(-4*ratio))
        import math

        score = 25.0 * (1.0 - math.exp(-4.0 * ratio))
        return round(score, 1)

    # ── 3. Documentation (15 pts) ─────────────────────────────────

    def _documentation(self) -> float:
        """Check README, docs/, and inline docstring coverage."""
        score = 0.0

        # README present
        readme = self.root / "README.md"
        if readme.exists():
            score += 5.0

        # docs/ directory present
        docs_dir = self.root / "docs"
        if docs_dir.exists() and any(docs_dir.iterdir()):
            score += 5.0

        # Docstring coverage heuristic
        docstring_score = self._docstring_coverage()
        score += docstring_score  # 0-5 pts

        return min(score, 15.0)

    def _docstring_coverage(self) -> float:
        """Rough estimate: count functions with docstrings / total functions."""
        import ast

        total_funcs = 0
        docstring_funcs = 0
        for pkg in self._discover_packages():
            for py_file in pkg.rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue
                try:
                    tree = ast.parse(py_file.read_text())
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_funcs += 1
                        if ast.get_docstring(node):
                            docstring_funcs += 1

        if total_funcs == 0:
            return 0.0
        ratio = docstring_funcs / total_funcs
        return 5.0 * ratio

    # ── 4. Dependency Health (15 pts) ─────────────────────────────

    def _dependency_health(self) -> float:
        """Check for requirements/pyproject, no known CVEs."""
        score = 0.0

        # pyproject.toml or requirements.txt present
        if (self.root / "pyproject.toml").exists():
            score += 5.0
        elif (self.root / "requirements.txt").exists():
            score += 3.0

        # requirements-dev.txt present
        if (self.root / "requirements-dev.txt").exists():
            score += 2.0

        # Check for lock file (more reproducible)
        if any((self.root / f).exists() for f in ("poetry.lock", "Pipfile.lock", "Cargo.lock")):
            score += 3.0

        # Check for security scan results
        if (self.root / "security-audit.txt").exists() or (self.root / ".github" / "dependabot.yml").exists():
            score += 5.0

        return min(score, 15.0)

    # ── 5. Issue Hygiene (15 pts) ─────────────────────────────────

    def _issue_hygiene(self) -> float:
        """Read local issue cache or fallback to 10."""
        cache = self.root / ".triage_cache" / "issue_hygiene.json"
        if cache.exists():
            data = json.loads(cache.read_text())
            return data.get("score", 10.0)
        # Without GitHub API access, return a neutral mid-score
        return 10.0

    # ── Helpers ───────────────────────────────────────────────────

    def _discover_packages(self) -> list[Path]:
        """Find top-level Python package directories."""
        pkgs = []
        for item in self.root.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                init = item / "__init__.py"
                if init.exists():
                    pkgs.append(item)
        return pkgs

    def run(self) -> HealthScore:
        """Compute full health score."""
        return HealthScore(
            freshness=self._freshness(),
            test_coverage=self._test_coverage(),
            documentation=self._documentation(),
            dependency_health=self._dependency_health(),
            issue_hygiene=self._issue_hygiene(),
        )


def run_health_check(repo_root: str | Path) -> HealthScore:
    """One-liner entrypoint."""
    return RepoHealthMetrics(repo_root).run()
