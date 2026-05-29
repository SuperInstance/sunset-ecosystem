"""Tests for CodebaseState — codebase survey and analysis.

Covers _count_files_and_lines, _detect_patterns, _scan_debt,
survey_codebase, and the CodebaseState dataclass.
"""

import os
import tempfile
from pathlib import Path

import pytest

from logos.codebase_state import (
    CodebaseState,
    _count_files_and_lines,
    _detect_patterns,
    _scan_debt,
    survey_codebase,
)


# ---------------------------------------------------------------------------
# CodebaseState
# ---------------------------------------------------------------------------

class TestCodebaseState:
    def test_defaults(self):
        cs = CodebaseState(root="/tmp")
        assert cs.file_count == 0
        assert cs.total_lines == 0
        assert cs.language_breakdown == {}

    def test_repr(self):
        cs = CodebaseState(root="/tmp", file_count=5, total_lines=100)
        r = repr(cs)
        assert "files=5" in r
        assert "lines=100" in r


# ---------------------------------------------------------------------------
# _count_files_and_lines
# ---------------------------------------------------------------------------

class TestCountFilesAndLines:
    def test_empty_dir(self, tmp_path):
        fc, tl, lf, ll = _count_files_and_lines(tmp_path)
        assert fc == 0
        assert tl == 0

    def test_python_file(self, tmp_path):
        (tmp_path / "hello.py").write_text("print('hello')\nprint('world')\n")
        fc, tl, lf, ll = _count_files_and_lines(tmp_path)
        assert fc == 1
        assert tl == 2
        assert lf.get("Python") == 1
        assert ll.get("Python") == 2

    def test_skips_pycache(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "foo.cpython-39.pyc").write_text("x")
        (tmp_path / "real.py").write_text("1\n")
        fc, tl, lf, ll = _count_files_and_lines(tmp_path)
        assert fc == 1  # only real.py
        assert "Python" in lf

    def test_unknown_extension(self, tmp_path):
        (tmp_path / "data.xyz").write_text("abc\n")
        fc, tl, lf, ll = _count_files_and_lines(tmp_path)
        assert fc == 1
        # unknown extensions don't appear in language breakdown
        assert lf == {}


# ---------------------------------------------------------------------------
# _detect_patterns
# ---------------------------------------------------------------------------

class TestDetectPatterns:
    def test_module_dirs(self, tmp_path):
        (tmp_path / "foo").mkdir()
        (tmp_path / "foo" / "__init__.py").write_text("")
        pats = _detect_patterns(tmp_path)
        assert "foo" in pats["module_dirs"]

    def test_imports(self, tmp_path):
        (tmp_path / "a.py").write_text("import os\nfrom pathlib import Path\n")
        pats = _detect_patterns(tmp_path)
        assert "os" in pats["imported_packages"]
        assert "pathlib" in pats["imported_packages"]

    def test_entry_points(self, tmp_path):
        (tmp_path / "main.py").write_text("if __name__ == '__main__': pass\n")
        pats = _detect_patterns(tmp_path)
        assert "main.py" in pats["entry_points"]


# ---------------------------------------------------------------------------
# _scan_debt
# ---------------------------------------------------------------------------

class TestScanDebt:
    def test_todo(self, tmp_path):
        (tmp_path / "a.py").write_text("# TODO: fix this\n")
        debt = _scan_debt(tmp_path)
        assert len(debt["TODO"]) == 1
        assert "fix this" in debt["TODO"][0]

    def test_fixme(self, tmp_path):
        (tmp_path / "a.py").write_text("# FIXME: broken\n")
        debt = _scan_debt(tmp_path)
        assert len(debt["FIXME"]) == 1

    def test_hack(self, tmp_path):
        (tmp_path / "a.py").write_text("# HACK: workaround\n")
        debt = _scan_debt(tmp_path)
        assert len(debt["HACK"]) == 1

    def test_no_debt(self, tmp_path):
        (tmp_path / "clean.py").write_text("print('ok')\n")
        debt = _scan_debt(tmp_path)
        assert all(len(v) == 0 for v in debt.values())

    def test_multiple_in_one_file(self, tmp_path):
        (tmp_path / "a.py").write_text("# TODO: a\n# FIXME: b\n# HACK: c\n")
        debt = _scan_debt(tmp_path)
        assert len(debt["TODO"]) == 1
        assert len(debt["FIXME"]) == 1
        assert len(debt["HACK"]) == 1

    def test_makefile_included(self, tmp_path):
        (tmp_path / "Makefile").write_text("build:\n\t# TODO: finish\n")
        debt = _scan_debt(tmp_path)
        assert len(debt["TODO"]) == 1


# ---------------------------------------------------------------------------
# survey_codebase
# ---------------------------------------------------------------------------

class TestSurveyCodebase:
    def test_survey_current_repo(self):
        # survey the sunset-ecosystem repo itself
        state = survey_codebase()
        assert state.file_count > 0
        assert state.total_lines > 0
        assert "Python" in state.language_breakdown
        assert state.test_count > 0

    def test_nonexistent_dir(self):
        state = survey_codebase("/nonexistent/path/12345")
        assert len(state.errors) == 1

    def test_empty_dir(self, tmp_path):
        state = survey_codebase(str(tmp_path))
        assert state.file_count == 0
        assert state.total_lines == 0
