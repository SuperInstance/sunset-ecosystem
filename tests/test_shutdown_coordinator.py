"""Tests for shutdown_coordinator.py — Graceful shutdown with ordered hooks.

Run: python3 -m pytest tests/test_shutdown_coordinator.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.shutdown_coordinator import ShutdownCoordinator, ShutdownTimeout


class TestShutdownCoordinator:
    def test_create(self):
        sc = ShutdownCoordinator()
        assert sc.stats()["hooks"] == 0
        assert not sc.is_shutting_down()

    def test_register_hook(self):
        sc = ShutdownCoordinator()
        sc.register_hook("db", lambda: None, priority=1)
        assert sc.stats()["hooks"] == 1

    def test_priority_order(self):
        order = []
        sc = ShutdownCoordinator()
        sc.register_hook("second", lambda: order.append("second"), priority=2)
        sc.register_hook("first", lambda: order.append("first"), priority=1)
        sc.shutdown()
        assert order == ["first", "second"]

    def test_drain_in_flight(self):
        sc = ShutdownCoordinator(timeout=1.0)
        sc.start_work()
        sc.start_work()

        def finish_later():
            time.sleep(0.1)
            sc.finish_work()
            sc.finish_work()

        import threading
        t = threading.Thread(target=finish_later)
        t.start()
        sc.shutdown()
        t.join()
        assert sc.stats()["in_flight"] == 0

    def test_drain_timeout(self):
        sc = ShutdownCoordinator(timeout=0.1)
        sc.start_work()
        sc.shutdown()  # Should not hang, proceed after timeout
        assert sc.is_shutting_down()

    def test_custom_drain(self):
        drained = [False]
        sc = ShutdownCoordinator()
        sc.register_drain(lambda: drained[0])
        sc.shutdown()  # drain is False, but no in_flight, so returns immediately

    def test_hook_failure_continues(self):
        order = []
        sc = ShutdownCoordinator()
        sc.register_hook("fail", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        sc.register_hook("ok", lambda: order.append("ok"))
        sc.shutdown()
        assert "ok" in order

    def test_idempotent_shutdown(self):
        sc = ShutdownCoordinator()
        sc.register_hook("db", lambda: None)
        sc.shutdown()
        sc.shutdown()  # Should not re-run
        assert sc.stats()["hooks_ran"] == 1

    def test_signal_handler_install(self):
        sc = ShutdownCoordinator()
        sc.install_signal_handlers()
        assert sc.stats()["signal_handlers_installed"] is True

    def test_signal_handler_idempotent(self):
        sc = ShutdownCoordinator()
        sc.install_signal_handlers()
        sc.install_signal_handlers()  # Should not crash
        assert sc.stats()["signal_handlers_installed"] is True

    def test_repr(self):
        sc = ShutdownCoordinator()
        assert "ShutdownCoordinator" in repr(sc)
