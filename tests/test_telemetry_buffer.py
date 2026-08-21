"""Tests for telemetry_buffer.py — Telemetry buffering and batching.

Run: python3 -m pytest tests/test_telemetry_buffer.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.telemetry_buffer import TelemetryBuffer


class TestTelemetryBuffer:
    def test_create(self):
        buf = TelemetryBuffer(max_size=100, flush_interval_sec=60)
        assert buf.stats()["max_size"] == 100
        assert buf.stats()["buffer_size"] == 0

    def test_record(self):
        buf = TelemetryBuffer()
        buf.record({"metric": "cpu", "value": 45.0})
        assert buf.size() == 1

    def test_should_flush_size(self):
        buf = TelemetryBuffer(max_size=2)
        buf.record({"a": 1})
        buf.record({"a": 2})
        assert buf.should_flush() is True

    def test_should_flush_time(self):
        buf = TelemetryBuffer(flush_interval_sec=10, clock=lambda: 0)
        buf.record({"a": 1})
        buf._clock = lambda: 15
        assert buf.should_flush() is True

    def test_flush(self):
        buf = TelemetryBuffer()
        buf.record({"a": 1})
        buf.record({"a": 2})
        batch = buf.flush()
        assert len(batch) == 2
        assert buf.size() == 0

    def test_flush_empty(self):
        buf = TelemetryBuffer()
        batch = buf.flush()
        assert batch == []

    def test_is_empty(self):
        buf = TelemetryBuffer()
        assert buf.is_empty() is True
        buf.record({"a": 1})
        assert buf.is_empty() is False

    def test_time_since_flush(self):
        buf = TelemetryBuffer(clock=lambda: 0)
        buf.flush()
        buf._clock = lambda: 30
        assert buf.time_since_flush() == 30

    def test_stats(self):
        buf = TelemetryBuffer()
        buf.record({"a": 1})
        stats = buf.stats()
        assert stats["buffer_size"] == 1
        assert stats["total_flushed"] == 0

    def test_repr(self):
        buf = TelemetryBuffer()
        assert "TelemetryBuffer" in repr(buf)
