"""Tests for signal_handler.py — Event-driven signal system.

Run: python3 -m pytest tests/test_signal_handler.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.signal_handler import SignalHandler


class TestSignalHandler:
    def test_create(self):
        signals = SignalHandler()
        assert len(signals.signals()) == 0

    def test_connect_and_emit(self):
        signals = SignalHandler()
        received = []
        signals.connect("test", lambda ctx: received.append(ctx))
        signals.emit("test", "payload")
        assert received == ["payload"]

    def test_multiple_handlers(self):
        signals = SignalHandler()
        results = []
        signals.connect("multi", lambda ctx: results.append(1))
        signals.connect("multi", lambda ctx: results.append(2))
        signals.emit("multi", None)
        assert sorted(results) == [1, 2]

    def test_disconnect(self):
        signals = SignalHandler()
        handler = lambda ctx: None
        signals.connect("x", handler)
        assert signals.disconnect("x", handler) is True
        assert signals.handler_count("x") == 0
        assert signals.disconnect("x", handler) is False

    def test_disconnect_all(self):
        signals = SignalHandler()
        signals.connect("x", lambda ctx: None)
        signals.connect("x", lambda ctx: None)
        signals.disconnect_all("x")
        assert signals.handler_count("x") == 0

    def test_emit_one(self):
        signals = SignalHandler()
        received = []
        signals.connect("test", lambda ctx: received.append(ctx))
        signals.connect("test", lambda ctx: received.append(ctx + "2"))
        signals.emit_one("test", "hello")
        assert received == ["hello"]

    def test_no_handlers(self):
        signals = SignalHandler()
        signals.emit("missing", None)  # should not raise
        assert signals.emit_one("missing", None) is False

    def test_handler_error_not_fatal(self):
        signals = SignalHandler()
        results = []
        signals.connect("test", lambda ctx: (_ for _ in ()).throw(ValueError("boom")))
        signals.connect("test", lambda ctx: results.append("ok"))
        signals.emit("test", None)
        assert results == ["ok"]

    def test_has_handlers(self):
        signals = SignalHandler()
        assert signals.has_handlers("x") is False
        signals.connect("x", lambda ctx: None)
        assert signals.has_handlers("x") is True

    def test_signals_list(self):
        signals = SignalHandler()
        signals.connect("a", lambda ctx: None)
        signals.connect("b", lambda ctx: None)
        assert sorted(signals.signals()) == ["a", "b"]

    def test_repr(self):
        signals = SignalHandler()
        signals.connect("x", lambda ctx: None)
        assert "SignalHandler" in repr(signals)
