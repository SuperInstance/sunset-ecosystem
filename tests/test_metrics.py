"""Tests for RepoHealthMetrics — SPEC-REPO-METRIC §2 implementation.

Covers HealthScore, RepoHealthMetrics scoring, and run_health_check entrypoint.
"""

import subprocess
from pathlib import Path

import pytest

from triage.metrics import HealthScore, RepoHealthMetrics, run_health_check


# ---------------------------------------------------------------------------
# HealthScore
# ---------------------------------------------------------------------------


class TestHealthScore:
    def test_total(self):
        h = HealthScore(
            freshness=30.0,
            test_coverage=25.0,
            documentation=15.0,
            dependency_health=15.0,
            issue_hygiene=15.0,
        )
        assert h.total == 100.0

    def test_traffic_light_green(self):
        h = HealthScore(
            freshness=30,
            test_coverage=25,
            documentation=15,
            dependency_health=15,
            issue_hygiene=15,
        )
        assert h.traffic_light == "green"

    def test_traffic_light_yellow(self):
        h = HealthScore(
            freshness=10,
            test_coverage=10,
            documentation=10,
            dependency_health=10,
            issue_hygiene=10,
        )
        assert h.total == 50.0
        assert h.traffic_light == "yellow"

    def test_traffic_light_red(self):
        h = HealthScore(
            freshness=5,
            test_coverage=5,
            documentation=5,
            dependency_health=5,
            issue_hygiene=5,
        )
        assert h.total == 25.0
        assert h.traffic_light == "red"

    def test_to_dict(self):
        h = HealthScore(
            freshness=10,
            test_coverage=10,
            documentation=10,
            dependency_health=10,
            issue_hygiene=10,
        )
        d = h.to_dict()
        assert d["total"] == 50.0
        assert d["traffic_light"] == "yellow"


# ---------------------------------------------------------------------------
# RepoHealthMetrics
# ---------------------------------------------------------------------------


class TestRepoHealthMetricsInit:
    def test_init(self, tmp_path):
        m = RepoHealthMetrics(tmp_path)
        assert m.root == tmp_path.resolve()


class TestRepoHealthMetricsFreshness:
    def test_freshness_recent(self, tmp_path):
        # Initialize git repo with a recent commit
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        (tmp_path / "f").write_text("x")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True
        )
        m = RepoHealthMetrics(tmp_path)
        score = m._freshness()
        assert score > 25.0  # very recent

    def test_freshness_no_git(self, tmp_path):
        m = RepoHealthMetrics(tmp_path)
        score = m._freshness()
        assert score == 0.0  # 90 days assumed


class TestRepoHealthMetricsCoverage:
    def test_no_tests(self, tmp_path):
        m = RepoHealthMetrics(tmp_path)
        assert m._test_coverage() == 0.0

    def test_with_tests(self, tmp_path):
        # Create a minimal package and tests
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("def foo(): pass\n")
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_foo.py").write_text("def test_foo(): pass\n")
        m = RepoHealthMetrics(tmp_path)
        score = m._test_coverage()
        assert score > 0.0
        assert score <= 25.0


class TestRepoHealthMetricsDocumentation:
    def test_nothing(self, tmp_path):
        m = RepoHealthMetrics(tmp_path)
        score = m._documentation()
        assert score == 0.0

    def test_readme_only(self, tmp_path):
        (tmp_path / "README.md").write_text("# Hello\n")
        m = RepoHealthMetrics(tmp_path)
        score = m._documentation()
        assert score == 5.0

    def test_readme_and_docs(self, tmp_path):
        (tmp_path / "README.md").write_text("# Hello\n")
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "guide.md").write_text("Guide\n")
        m = RepoHealthMetrics(tmp_path)
        score = m._documentation()
        assert score >= 10.0


class TestRepoHealthMetricsDependencies:
    def test_nothing(self, tmp_path):
        m = RepoHealthMetrics(tmp_path)
        score = m._dependency_health()
        assert score == 0.0

    def test_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        m = RepoHealthMetrics(tmp_path)
        score = m._dependency_health()
        assert score == 5.0

    def test_requirements(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("pytest\n")
        m = RepoHealthMetrics(tmp_path)
        score = m._dependency_health()
        assert score == 3.0

    def test_lock_file(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        (tmp_path / "poetry.lock").write_text("[[package]]\n")
        m = RepoHealthMetrics(tmp_path)
        score = m._dependency_health()
        assert score == 8.0


class TestRepoHealthMetricsIssueHygiene:
    def test_no_cache(self, tmp_path):
        m = RepoHealthMetrics(tmp_path)
        score = m._issue_hygiene()
        assert score == 10.0

    def test_with_cache(self, tmp_path):
        cache = tmp_path / ".triage_cache"
        cache.mkdir()
        (cache / "issue_hygiene.json").write_text('{"score": 12.5}')
        m = RepoHealthMetrics(tmp_path)
        score = m._issue_hygiene()
        assert score == 12.5


class TestRepoHealthMetricsRun:
    def test_run(self, tmp_path):
        # Set up a repo with some files
        (tmp_path / "README.md").write_text("# Test\n")
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("""\"\"\"Package.\"\"\"\n\n""")
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_x.py").write_text("def test_x(): pass\n")
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True
        )
        m = RepoHealthMetrics(tmp_path)
        score = m.run()
        assert isinstance(score, HealthScore)
        assert score.total > 0.0
        assert score.total <= 100.0


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


class TestRunHealthCheck:
    def test_entrypoint(self, tmp_path):
        score = run_health_check(tmp_path)
        assert isinstance(score, HealthScore)
