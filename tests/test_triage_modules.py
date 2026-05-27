"""Tests for triage modules: drift_detect, duplicate_detect, metrics."""

from __future__ import annotations

import os
import tempfile

import pytest

from triage.drift_detect import DriftDetector, DriftReport, detect_drift
from triage.duplicate_detect import DuplicateDetector, DuplicatePair, find_duplicates
from triage.metrics import HealthScore, RepoHealthMetrics


class TestDriftReport:
    def test_to_dict(self):
        report = DriftReport(repo="test-repo", severity="high", stale_dependencies=["pkg1"])
        d = report.to_dict()
        assert d["repo"] == "test-repo"
        assert d["severity"] == "high"
        assert "pkg1" in d["stale_dependencies"]

    def test_defaults(self):
        report = DriftReport(repo="test")
        assert report.severity == "low"
        assert report.dead_code == []


class TestDriftDetector:
    def test_doc_drift_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create README referencing missing file
            readme = os.path.join(tmpdir, "README.md")
            with open(readme, "w") as f:
                f.write("See `missing_file.py` for details.\n")
            detector = DriftDetector(tmpdir)
            drift = detector._doc_drift()
            assert any("missing_file.py" in d for d in drift)

    def test_doc_drift_valid_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            readme = os.path.join(tmpdir, "README.md")
            with open(readme, "w") as f:
                f.write("See `existing.py` for details.\n")
            with open(os.path.join(tmpdir, "existing.py"), "w") as f:
                f.write("# exists")
            detector = DriftDetector(tmpdir)
            drift = detector._doc_drift()
            assert len(drift) == 0

    def test_doc_drift_no_readme(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = DriftDetector(tmpdir)
            drift = detector._doc_drift()
            assert drift == []

    def test_dead_code_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Python file that is never imported
            with open(os.path.join(tmpdir, "orphan.py"), "w") as f:
                f.write("def hello(): pass\n")
            # Create a main file that doesn't import it
            with open(os.path.join(tmpdir, "main.py"), "w") as f:
                f.write("print('hello')\n")
            detector = DriftDetector(tmpdir)
            dead = detector._dead_code()
            assert any("orphan.py" in d for d in dead)

    def test_run_returns_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # init git repo
            os.system(f"cd {tmpdir} && git init && git config user.email t@t.com && git config user.name t && echo '# test' > README.md && git add . && git commit -m init")
            detector = DriftDetector(tmpdir)
            report = detector.run()
            assert isinstance(report, DriftReport)
            assert report.repo == os.path.basename(tmpdir)

    def test_detect_drift_function(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.system(f"cd {tmpdir} && git init && git config user.email t@t.com && git config user.name t && echo '# test' > README.md && git add . && git commit -m init")
            report = detect_drift(tmpdir)
            assert isinstance(report, DriftReport)


class TestDuplicateDetector:
    def _make_issues(self):
        return [
            {"number": 1, "title": "Bug in deployment pipeline", "body": "The deployment fails on staging"},
            {"number": 2, "title": "Deploy pipeline broken", "body": "The deployment fails on staging environment"},
            {"number": 3, "title": "Feature request: dark mode", "body": "Add dark mode to the UI"},
        ]

    def test_finds_duplicates(self):
        detector = DuplicateDetector(threshold=0.3, min_shared_terms=2)
        pairs = detector.detect(self._make_issues())
        # Issues 1 and 2 should be flagged as similar
        pair_numbers = [(p.issue_a, p.issue_b) for p in pairs]
        assert (1, 2) in pair_numbers

    def test_no_duplicates(self):
        issues = [
            {"number": 1, "title": "Fix login bug", "body": "Login button doesn't work"},
            {"number": 2, "title": "Feature request: dark mode", "body": "Add dark mode to the UI"},
        ]
        detector = DuplicateDetector(threshold=0.9)
        pairs = detector.detect(issues)
        assert len(pairs) == 0

    def test_single_issue(self):
        detector = DuplicateDetector()
        pairs = detector.detect([{"number": 1, "title": "test", "body": "test"}])
        assert len(pairs) == 0

    def test_empty_list(self):
        detector = DuplicateDetector()
        pairs = detector.detect([])
        assert len(pairs) == 0

    def test_find_duplicates_function(self):
        pairs = find_duplicates(self._make_issues(), threshold=0.3, min_shared_terms=2)
        assert isinstance(pairs, list)

    def test_duplicate_pair_repr(self):
        pair = DuplicatePair(issue_a=1, issue_b=2, similarity=0.85, shared_terms=["deploy", "fail"])
        r = repr(pair)
        assert "0.85" in r


class TestHealthScore:
    def test_total(self):
        score = HealthScore(
            freshness=30.0,
            test_coverage=25.0,
            documentation=15.0,
            dependency_health=15.0,
            issue_hygiene=15.0,
        )
        assert score.total == 100.0

    def test_traffic_light_green(self):
        score = HealthScore(freshness=30, test_coverage=25, documentation=15, dependency_health=15, issue_hygiene=15)
        assert score.traffic_light == "green"

    def test_traffic_light_yellow(self):
        score = HealthScore(freshness=20, test_coverage=15, documentation=10, dependency_health=5, issue_hygiene=5)
        assert score.traffic_light == "yellow"

    def test_traffic_light_red(self):
        score = HealthScore(freshness=5, test_coverage=5, documentation=0, dependency_health=0, issue_hygiene=0)
        assert score.traffic_light == "red"

    def test_to_dict(self):
        score = HealthScore(freshness=20, test_coverage=10, documentation=5, dependency_health=5, issue_hygiene=5)
        d = score.to_dict()
        assert "total" in d
        assert "traffic_light" in d


class TestRepoHealthMetrics:
    def test_documentation_score_with_readme(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "README.md"), "w") as f:
                f.write("# Test\n")
            metrics = RepoHealthMetrics(tmpdir)
            doc_score = metrics._documentation()
            assert doc_score >= 5.0  # README gives 5 pts

    def test_documentation_score_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = RepoHealthMetrics(tmpdir)
            doc_score = metrics._documentation()
            assert doc_score == 0.0

    def test_dependency_health_pyproject(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "pyproject.toml"), "w") as f:
                f.write("[project]\nname='test'\n")
            metrics = RepoHealthMetrics(tmpdir)
            dep_score = metrics._dependency_health()
            assert dep_score >= 5.0

    def test_run_returns_health_score(self):
        # Use the actual sunset-ecosystem repo for a real test
        import os
        repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        metrics = RepoHealthMetrics(repo_path)
        score = metrics.run()
        assert isinstance(score, HealthScore)
        assert score.total > 0
