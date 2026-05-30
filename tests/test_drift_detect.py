"""Tests for DriftDetector — structural drift detection in repos.

Covers DriftReport, DriftDetector _doc_drift, _dead_code, _branch_divergence,
run() orchestrator, severity scoring, and detect_drift entrypoint.
"""

import os
import subprocess
from pathlib import Path

import pytest

from triage.drift_detect import DriftDetector, DriftReport, detect_drift


# ---------------------------------------------------------------------------
# DriftReport
# ---------------------------------------------------------------------------

class TestDriftReport:
    def test_defaults(self):
        r = DriftReport(repo="test")
        assert r.stale_dependencies == []
        assert r.test_regression is None
        assert r.doc_drift == []
        assert r.dead_code == []
        assert r.branch_divergence is None
        assert r.license_change is None
        assert r.severity == "low"

    def test_to_dict(self):
        r = DriftReport(repo="test", severity="high", test_regression="fail")
        d = r.to_dict()
        assert d["repo"] == "test"
        assert d["severity"] == "high"
        assert d["test_regression"] == "fail"


# ---------------------------------------------------------------------------
# DriftDetector init
# ---------------------------------------------------------------------------

class TestDetectorInit:
    def test_from_path(self, tmp_path):
        d = DriftDetector(tmp_path)
        assert d.repo_name == tmp_path.name
        assert d.root == tmp_path.resolve()


# ---------------------------------------------------------------------------
# _doc_drift
# ---------------------------------------------------------------------------

class TestDocDrift:
    def test_missing_readme(self, tmp_path):
        d = DriftDetector(tmp_path)
        assert d._doc_drift() == []

    def test_missing_file_reference(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("See `missing.py` for details.\n")
        d = DriftDetector(tmp_path)
        findings = d._doc_drift()
        assert len(findings) == 1
        assert "missing.py" in findings[0]

    def test_missing_link(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("[Guide](docs/guide.md)\n")
        d = DriftDetector(tmp_path)
        findings = d._doc_drift()
        assert len(findings) == 1
        assert "docs/guide.md" in findings[0]

    def test_external_url_ignored(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("[Link](https://example.com)\n")
        d = DriftDetector(tmp_path)
        assert d._doc_drift() == []

    def test_anchor_ignored(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("[Section](#heading)\n")
        d = DriftDetector(tmp_path)
        assert d._doc_drift() == []

    def test_existing_file_ok(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("See `real.py` for details.\n")
        (tmp_path / "real.py").write_text("# ok\n")
        d = DriftDetector(tmp_path)
        assert d._doc_drift() == []


# ---------------------------------------------------------------------------
# _dead_code
# ---------------------------------------------------------------------------

class TestDeadCode:
    def test_no_py_files(self, tmp_path):
        d = DriftDetector(tmp_path)
        assert d._dead_code() == []

    def test_unimported_module(self, tmp_path):
        (tmp_path / "main.py").write_text("import os\n")
        (tmp_path / "orphan.py").write_text("print('hello')\n")
        d = DriftDetector(tmp_path)
        findings = d._dead_code()
        assert any("orphan.py" in f for f in findings)

    def test_imported_module_ok(self, tmp_path):
        (tmp_path / "util.py").write_text("def helper(): pass\n")
        (tmp_path / "main.py").write_text("import util\n")
        d = DriftDetector(tmp_path)
        findings = d._dead_code()
        assert not any("util.py" in f for f in findings)

    def test_from_import_ok(self, tmp_path):
        (tmp_path / "util.py").write_text("def helper(): pass\n")
        (tmp_path / "main.py").write_text("from util import helper\n")
        d = DriftDetector(tmp_path)
        findings = d._dead_code()
        assert not any("util.py" in f for f in findings)

    def test_skips_tests(self, tmp_path):
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_x.py").write_text("def test(): pass\n")
        d = DriftDetector(tmp_path)
        findings = d._dead_code()
        assert not any("test_x.py" in f for f in findings)

    def test_skips_init(self, tmp_path):
        (tmp_path / "__init__.py").write_text("")
        d = DriftDetector(tmp_path)
        findings = d._dead_code()
        assert not any("__init__.py" in f for f in findings)


# ---------------------------------------------------------------------------
# _branch_divergence
# ---------------------------------------------------------------------------

class TestBranchDivergence:
    def test_no_git(self, tmp_path):
        d = DriftDetector(tmp_path)
        assert d._branch_divergence() is None

    def test_no_upstream(self, tmp_path):
        # Initialize git repo with no upstream
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        (tmp_path / "f").write_text("x")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
        d = DriftDetector(tmp_path)
        assert d._branch_divergence() is None


# ---------------------------------------------------------------------------
# _license_change
# ---------------------------------------------------------------------------

class TestLicenseChange:
    def test_no_license(self, tmp_path):
        d = DriftDetector(tmp_path)
        assert d._license_change() is None

    def test_recent_license(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        (tmp_path / "LICENSE").write_text("MIT\n")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
        d = DriftDetector(tmp_path)
        result = d._license_change()
        assert result is not None
        assert "LICENSE modified" in result

    def test_no_git(self, tmp_path):
        d = DriftDetector(tmp_path)
        assert d._license_change() is None


# ---------------------------------------------------------------------------
# run / severity
# ---------------------------------------------------------------------------

class TestRun:
    def test_clean_repo(self, tmp_path):
        d = DriftDetector(tmp_path)
        report = d.run()
        assert isinstance(report, DriftReport)
        assert report.repo == tmp_path.name
        assert report.severity == "low"

    def test_critical_with_regression(self, tmp_path):
        d = DriftDetector(tmp_path)
        # Monkeypatch to simulate regression
        d._test_regression = lambda: "FAILED test_x"
        report = d.run()
        assert report.severity == "critical"
        assert report.test_regression == "FAILED test_x"

    def test_high_with_stale(self, tmp_path):
        d = DriftDetector(tmp_path)
        d._stale_dependencies = lambda: ["vuln found"]
        report = d.run()
        assert report.severity == "high"

    def test_medium_with_doc_drift(self, tmp_path):
        d = DriftDetector(tmp_path)
        d._doc_drift = lambda: ["missing file"]
        d._branch_divergence = lambda: "5 commits behind"
        report = d.run()
        assert report.severity == "medium"


# ---------------------------------------------------------------------------
# detect_drift entrypoint
# ---------------------------------------------------------------------------

class TestDetectDrift:
    def test_entrypoint(self, tmp_path):
        report = detect_drift(tmp_path)
        assert isinstance(report, DriftReport)
        assert report.repo == tmp_path.name
