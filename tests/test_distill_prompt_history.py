"""Tests for distill.prompt_history."""

from __future__ import annotations

import pytest

from distill.prompt_history import PromptHistory, PromptRecord


class TestPromptRecord:
    def test_creation_defaults(self):
        r = PromptRecord(prompt="hello", response="world")
        assert r.prompt == "hello"
        assert r.response == "world"
        assert r.seed == 42
        assert r.temperature == 0.7
        assert r.model == "unknown"
        assert r.quality_score == -1.0

    def test_repr(self):
        r = PromptRecord(prompt="test", response="resp", model="gpt-4", hint_level=3)
        r2 = repr(r)
        assert "gpt-4" in r2
        assert "hints=3" in r2


class TestPromptHistory:
    def test_add_and_count(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="a", response="b"))
        h.add(PromptRecord(prompt="c", response="d"))
        assert h.count() == 2

    def test_query_by_model(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="a", response="b", model="gpt-4"))
        h.add(PromptRecord(prompt="c", response="d", model="claude"))
        results = h.query(model="gpt-4")
        assert len(results) == 1
        assert results[0].model == "gpt-4"

    def test_query_by_hint_level(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="a", response="b", hint_level=5))
        h.add(PromptRecord(prompt="c", response="d", hint_level=0))
        results = h.query(hint_level=0)
        assert len(results) == 1
        assert results[0].hint_level == 0

    def test_query_by_application(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="a", response="b", application="chat"))
        h.add(PromptRecord(prompt="c", response="d", application="code"))
        results = h.query(application="chat")
        assert len(results) == 1

    def test_query_min_quality(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="a", response="b", quality_score=0.8))
        h.add(PromptRecord(prompt="c", response="d", quality_score=0.3))
        results = h.query(min_quality=0.5)
        assert len(results) == 1
        assert results[0].quality_score == 0.8

    def test_query_limit(self):
        h = PromptHistory()
        for i in range(10):
            h.add(PromptRecord(prompt=f"p{i}", response=f"r{i}"))
        results = h.query(limit=3)
        assert len(results) == 3

    def test_max_records_eviction(self):
        h = PromptHistory(max_records=3)
        for i in range(5):
            h.add(PromptRecord(prompt=f"p{i}", response=f"r{i}"))
        assert h.count() == 3

    def test_random_returns_record(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="a", response="b"))
        r = h.random()
        assert r is not None
        assert r.prompt == "a"

    def test_random_empty(self):
        h = PromptHistory()
        assert h.random() is None

    def test_random_by_application(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="a", response="b", application="chat"))
        h.add(PromptRecord(prompt="c", response="d", application="code"))
        r = h.random(application="chat")
        assert r is not None
        assert r.application == "chat"

    def test_average_quality(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="a", response="b", quality_score=0.8))
        h.add(PromptRecord(prompt="c", response="d", quality_score=0.6))
        assert h.average_quality() == pytest.approx(0.7)

    def test_average_quality_no_scored(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="a", response="b"))
        assert h.average_quality() == 0.0

    def test_average_quality_by_application(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="a", response="b", quality_score=0.9, application="chat"))
        h.add(PromptRecord(prompt="c", response="d", quality_score=0.5, application="code"))
        assert h.average_quality(application="chat") == pytest.approx(0.9)

    def test_repr(self):
        h = PromptHistory()
        r = repr(h)
        assert "records=0" in r
