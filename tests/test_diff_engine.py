"""Tests for diff_engine.py — Text diff comparison.

Run: python3 -m pytest tests/test_diff_engine.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.diff_engine import DiffEngine, DiffResult


class TestDiffEngine:
    def test_create(self):
        engine = DiffEngine()
        assert engine is not None

    def test_identical(self):
        engine = DiffEngine()
        result = engine.compare("hello world", "hello world")
        assert result.ratio() == 1.0
        assert len(result.chunks) == 1
        assert result.chunks[0].tag == "equal"

    def test_completely_different(self):
        engine = DiffEngine()
        result = engine.compare("abc", "xyz")
        assert result.ratio() == 0.0

    def test_partial(self):
        engine = DiffEngine()
        result = engine.compare("hello world", "hello fleet")
        # "hello" is equal, "world" vs "fleet" differs
        assert 0.0 < result.ratio() < 1.0

    def test_empty_strings(self):
        engine = DiffEngine()
        result = engine.compare("", "")
        assert result.ratio() == 1.0

    def test_one_empty(self):
        engine = DiffEngine()
        result = engine.compare("hello", "")
        assert result.ratio() == 0.0

    def test_compare_lines(self):
        engine = DiffEngine()
        result = engine.compare_lines("line1\nline2", "line1\nline3")
        assert 0.0 < result.ratio() < 1.0

    def test_unified_format(self):
        engine = DiffEngine()
        result = engine.compare("a b c", "a x c")
        text = result.unified_format()
        assert " a" in text
        assert "-b" in text
        assert "+x" in text

    def test_multiword_diff(self):
        engine = DiffEngine()
        result = engine.compare(
            "the quick brown fox",
            "the slow brown fox",
        )
        assert 0.0 < result.ratio() < 1.0

    def test_chunks_structure(self):
        engine = DiffEngine()
        result = engine.compare("a b", "a c")
        tags = [c.tag for c in result.chunks]
        assert "equal" in tags
        assert "delete" in tags or "insert" in tags
