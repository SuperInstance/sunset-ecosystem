"""Tests for Code Review CI Integration.

Covers CI entry point, JSON output, severity filtering, mock git diff.
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

from fleet.review_code_ci import (
    get_changed_files,
    run_ci_review,
    format_markdown_comment,
    format_json_output,
    severity_rank,
)
from fleet.review_code import ReviewPersona, ReviewFinding


# ---------------------------------------------------------------------------
# Severity rank
# ---------------------------------------------------------------------------

class TestSeverityRank:
    def test_info(self):
        assert severity_rank("info") == 0

    def test_warning(self):
        assert severity_rank("warning") == 1

    def test_critical(self):
        assert severity_rank("critical") == 2

    def test_unknown(self):
        assert severity_rank("unknown") == 0


# ---------------------------------------------------------------------------
# Git diff helper
# ---------------------------------------------------------------------------

class TestGetChangedFiles:
    def test_git_diff(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], capture_output=True)
        
        # Create a file and commit
        (tmp_path / "foo.py").write_text("x = 1")
        subprocess.run(["git", "add", "."], capture_output=True)
        subprocess.run(["git", "commit", "-m", "first"], capture_output=True)
        
        # Modify file
        (tmp_path / "foo.py").write_text("x = 2")
        subprocess.run(["git", "add", "."], capture_output=True)
        subprocess.run(["git", "commit", "-m", "second"], capture_output=True)
        
        files = get_changed_files("HEAD~1")
        assert "foo.py" in files

    def test_git_diff_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        files = get_changed_files()
        assert files == []


# ---------------------------------------------------------------------------
# CI review runner
# ---------------------------------------------------------------------------

class TestRunCIReview:
    def test_empty_files(self):
        result = run_ci_review([], fail_on_severity="critical")
        assert result["summary"]["total_files"] == 0
        assert not result["summary"]["should_fail"]

    def test_single_file_no_issues(self, tmp_path):
        code = "def foo(a):\n    return a + 1\n"
        path = tmp_path / "clean.py"
        path.write_text(code)
        result = run_ci_review([str(path)], fail_on_severity="critical")
        assert result["summary"]["total_files"] == 1
        # May have info findings, but no critical
        assert not result["summary"]["should_fail"]

    def test_single_file_with_critical(self, tmp_path):
        code = "x = eval('1')\n"
        path = tmp_path / "bad.py"
        path.write_text(code)
        result = run_ci_review([str(path)], fail_on_severity="critical")
        assert result["summary"]["critical"] > 0
        assert result["summary"]["should_fail"]

    def test_fail_on_warning(self, tmp_path):
        code = "for i in range(10):\n    for j in range(10):\n        for k in range(10):\n            pass\n"
        path = tmp_path / "nested.py"
        path.write_text(code)
        result = run_ci_review([str(path)], fail_on_severity="warning")
        assert result["summary"]["should_fail"]

    def test_fail_on_info(self, tmp_path):
        code = "def foo(a):\n    return a + 1\n"
        path = tmp_path / "simple.py"
        path.write_text(code)
        result = run_ci_review([str(path)], fail_on_severity="info")
        # Should fail if any info-level findings exist
        assert result["summary"]["total_findings"] > 0

    def test_multiple_files(self, tmp_path):
        (tmp_path / "a.py").write_text("def foo(a):\n    return a + 1\n")
        (tmp_path / "b.py").write_text("x = eval('1')\n")
        files = [str(tmp_path / "a.py"), str(tmp_path / "b.py")]
        result = run_ci_review(files, fail_on_severity="critical")
        assert result["summary"]["total_files"] == 2
        assert result["summary"]["critical"] > 0

    def test_nonexistent_file(self, tmp_path):
        result = run_ci_review([str(tmp_path / "missing.py")])
        assert result["summary"]["total_files"] == 1
        assert result["summary"]["total_findings"] == 0


# ---------------------------------------------------------------------------
# Format output
# ---------------------------------------------------------------------------

class TestFormatOutput:
    def test_format_json(self):
        result = {
            "files": {},
            "summary": {
                "total_files": 1,
                "total_findings": 2,
                "critical": 1,
                "warning": 1,
                "info": 0,
                "should_fail": True,
            },
            "findings": [],
        }
        json_str = format_json_output(result)
        parsed = json.loads(json_str)
        assert parsed["summary"]["critical"] == 1

    def test_format_markdown(self):
        result = {
            "files": {
                "test.py": {
                    "findings": [
                        {"persona": "Security", "severity": "critical", "message": "eval used", "line": 1},
                    ],
                    "count": 1,
                    "critical": 1,
                    "warning": 0,
                    "info": 0,
                }
            },
            "summary": {
                "total_files": 1,
                "total_findings": 1,
                "critical": 1,
                "warning": 0,
                "info": 0,
                "should_fail": True,
            },
            "findings": [],
        }
        md = format_markdown_comment(result)
        assert "Fleet Code Review" in md
        assert "eval used" in md
        assert "🔴" in md

    def test_format_markdown_no_findings(self):
        result = {
            "files": {},
            "summary": {
                "total_files": 0,
                "total_findings": 0,
                "critical": 0,
                "warning": 0,
                "info": 0,
                "should_fail": False,
            },
            "findings": [],
        }
        md = format_markdown_comment(result)
        assert "Fleet Code Review" in md


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

class TestCLI:
    def test_main_files_flag(self, tmp_path, monkeypatch, capsys):
        code = "def foo(a):\n    return a + 1\n"
        path = tmp_path / "clean.py"
        path.write_text(code)
        
        monkeypatch.chdir(tmp_path)
        with patch.object(sys, "argv", ["review_code_ci", "--files", str(path), "--output", "json"]):
            from fleet import review_code_ci
            with pytest.raises(SystemExit) as exc_info:
                review_code_ci.main()
            assert exc_info.value.code == 0
        
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["summary"]["total_files"] == 1

    def test_main_pr_files_no_git(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        with patch.object(sys, "argv", ["review_code_ci", "--pr-files", "--output", "json"]):
            from fleet import review_code_ci
            with pytest.raises(SystemExit) as exc_info:
                review_code_ci.main()
            assert exc_info.value.code == 0
        
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["summary"]["total_files"] == 0

    def test_main_comment_file(self, tmp_path, monkeypatch):
        code = "x = eval('1')\n"
        path = tmp_path / "bad.py"
        path.write_text(code)
        
        monkeypatch.chdir(tmp_path)
        comment_file = tmp_path / "comment.md"
        with patch.object(sys, "argv", [
            "review_code_ci",
            "--files", str(path),
            "--output", "markdown",
            "--comment-file", str(comment_file),
        ]):
            with pytest.raises(SystemExit) as exc_info:
                from fleet import review_code_ci
                review_code_ci.main()
            assert exc_info.value.code == 1  # critical findings, exit 1
        
        assert comment_file.exists()
        content = comment_file.read_text()
        assert "Fleet Code Review" in content

    def test_main_default_all_files(self, tmp_path, monkeypatch, capsys):
        code = "def foo(a):\n    return a + 1\n"
        (tmp_path / "foo.py").write_text(code)
        
        monkeypatch.chdir(tmp_path)
        with patch.object(sys, "argv", ["review_code_ci", "--output", "json"]):
            from fleet import review_code_ci
            with pytest.raises(SystemExit) as exc_info:
                review_code_ci.main()
            assert exc_info.value.code == 0
        
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["summary"]["total_files"] >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_severity_order(self):
        assert severity_rank("critical") > severity_rank("warning")
        assert severity_rank("warning") > severity_rank("info")

    def test_run_ci_review_with_syntax_error(self, tmp_path):
        code = "def foo(\n"
        path = tmp_path / "syntax_error.py"
        path.write_text(code)
        result = run_ci_review([str(path)])
        assert result["summary"]["total_files"] == 1
        # Should not crash, may report syntax error as critical
