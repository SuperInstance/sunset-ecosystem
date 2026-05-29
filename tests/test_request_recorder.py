"""Tests for request_recorder.py — HTTP request/response recording.

Run: python3 -m pytest tests/test_request_recorder.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.request_recorder import RequestRecorder


class TestRequestRecorder:
    def test_create(self):
        rec = RequestRecorder()
        assert rec.count() == 0

    def test_record_and_get(self):
        rec = RequestRecorder()
        rec.record(
            method="GET",
            url="http://example.com/api",
            status=200,
            response={"x": 1},
        )
        assert rec.count() == 1
        entry = rec.get(0)
        assert entry["method"] == "GET"
        assert entry["status"] == 200

    def test_record_with_error(self):
        rec = RequestRecorder()
        rec.record(
            method="POST",
            url="http://example.com/api",
            status=500,
            error="timeout",
        )
        entry = rec.get(0)
        assert entry["error"] == "timeout"

    def test_filter_by_status(self):
        rec = RequestRecorder()
        rec.record(method="GET", url="/a", status=200)
        rec.record(method="GET", url="/b", status=500)
        errors = rec.filter(status_min=400)
        assert len(errors) == 1
        assert errors[0]["url"] == "/b"

    def test_filter_by_method(self):
        rec = RequestRecorder()
        rec.record(method="GET", url="/a", status=200)
        rec.record(method="POST", url="/b", status=200)
        posts = rec.filter(method="POST")
        assert len(posts) == 1
        assert posts[0]["url"] == "/b"

    def test_capacity_eviction(self):
        rec = RequestRecorder(capacity=2)
        rec.record(method="GET", url="/1", status=200)
        rec.record(method="GET", url="/2", status=200)
        rec.record(method="GET", url="/3", status=200)
        assert rec.count() == 2
        assert rec.get(0)["url"] == "/2"

    def test_clear(self):
        rec = RequestRecorder()
        rec.record(method="GET", url="/a", status=200)
        rec.clear()
        assert rec.count() == 0

    def test_stats(self):
        rec = RequestRecorder()
        rec.record(method="GET", url="/a", status=200)
        rec.record(method="GET", url="/b", status=500)
        stats = rec.stats()
        assert stats["total"] == 2
        assert stats["errors"] == 1

    def test_repr(self):
        rec = RequestRecorder()
        assert "RequestRecorder" in repr(rec)
