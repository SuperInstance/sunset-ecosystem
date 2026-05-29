"""Tests for audit_logger.py — Structured audit logging with HMAC.

Run: python3 -m pytest tests/test_audit_logger.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.audit_logger import AuditLogger


class TestAuditLogger:
    def test_create(self):
        logger = AuditLogger(secret="key")
        assert logger.count() == 0

    def test_record(self):
        logger = AuditLogger(secret="key")
        logger.record("user.login", {"user": "alice"})
        assert logger.count() == 1

    def test_query_by_action(self):
        logger = AuditLogger(secret="key")
        logger.record("user.login")
        logger.record("user.logout")
        results = logger.query(action="user.login")
        assert len(results) == 1

    def test_query_by_time(self):
        import time
        logger = AuditLogger(secret="key")
        before = time.time()
        logger.record("user.login")
        after = time.time()
        results = logger.query(since=before)
        assert len(results) == 1
        results = logger.query(until=before)
        assert len(results) == 0

    def test_hmac_integrity(self):
        logger = AuditLogger(secret="key")
        logger.record("user.login")
        entry = logger.query()[0]
        assert logger.verify(entry) is True

    def test_tamper_detection(self):
        logger = AuditLogger(secret="key")
        logger.record("user.login")
        entry = logger.query()[0]
        entry["action"] = "tampered"
        assert logger.verify(entry) is False

    def test_capacity_eviction(self):
        logger = AuditLogger(secret="key", capacity=2)
        logger.record("a")
        logger.record("b")
        logger.record("c")
        assert logger.count() == 2
        assert logger.query(action="a") == []

    def test_stats(self):
        logger = AuditLogger(secret="key")
        logger.record("user.login")
        logger.record("user.login")
        logger.record("user.logout")
        stats = logger.stats()
        assert stats["total"] == 3
        assert stats["by_action"]["user.login"] == 2

    def test_verify_all(self):
        logger = AuditLogger(secret="key")
        logger.record("user.login")
        assert logger.verify_all() == []

    def test_repr(self):
        logger = AuditLogger(secret="key")
        logger.record("user.login")
        assert "AuditLogger" in repr(logger)
