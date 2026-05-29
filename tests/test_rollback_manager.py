"""Tests for rollback_manager.py — Deployment rollback and versioning.

Run: python3 -m pytest tests/test_rollback_manager.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.rollback_manager import RollbackManager


class TestRollbackManager:
    def test_create(self):
        rm = RollbackManager()
        assert rm.stats()["history_size"] == 0
        assert rm.current() is None

    def test_deploy(self):
        rm = RollbackManager()
        assert rm.deploy("v1.0") is True
        assert rm.current() == "v1.0"
        assert rm.stats()["history_size"] == 1

    def test_deploy_with_metadata(self):
        rm = RollbackManager()
        rm.deploy("v1.0", {"status": "ok"})
        history = rm.history()
        assert history[0]["metadata"] == {"status": "ok"}

    def test_deploy_health_check_fail(self):
        rm = RollbackManager()
        assert rm.deploy("v1.0", health_check=lambda: False) is False
        assert rm.current() is None

    def test_rollback(self):
        rm = RollbackManager()
        rm.deploy("v1.0")
        rm.deploy("v1.1")
        assert rm.rollback() == "v1.0"
        assert rm.current() == "v1.0"

    def test_rollback_multiple_steps(self):
        rm = RollbackManager()
        rm.deploy("v1.0")
        rm.deploy("v1.1")
        rm.deploy("v1.2")
        assert rm.rollback(steps=2) == "v1.0"

    def test_rollback_not_available(self):
        rm = RollbackManager()
        rm.deploy("v1.0")
        assert rm.rollback() is None

    def test_rollback_to(self):
        rm = RollbackManager()
        rm.deploy("v1.0")
        rm.deploy("v1.1")
        rm.deploy("v1.2")
        assert rm.rollback_to("v1.0") is True
        assert rm.current() == "v1.0"
        assert rm.rollback_to("missing") is False

    def test_previous(self):
        rm = RollbackManager()
        rm.deploy("v1.0")
        assert rm.previous() is None
        rm.deploy("v1.1")
        assert rm.previous() == "v1.0"

    def test_versions(self):
        rm = RollbackManager()
        rm.deploy("v1.0")
        rm.deploy("v1.1")
        assert rm.versions() == ["v1.0", "v1.1"]

    def test_history_trim(self):
        rm = RollbackManager(max_history=2)
        rm.deploy("v1.0")
        rm.deploy("v1.1")
        rm.deploy("v1.2")
        assert len(rm.history()) == 2
        assert rm.versions() == ["v1.1", "v1.2"]

    def test_stats(self):
        rm = RollbackManager()
        rm.deploy("v1.0")
        stats = rm.stats()
        assert stats["current"] == "v1.0"
        assert stats["rollback_available"] is False

    def test_repr(self):
        rm = RollbackManager()
        assert "RollbackManager" in repr(rm)
