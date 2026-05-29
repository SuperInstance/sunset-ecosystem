"""Tests for audit_logger.py — Immutable audit trail.

Run: python3 -m pytest tests/test_audit_logger.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.audit_logger import AuditLogger, AuditTamperError


class TestAuditLogger:
    def test_create(self):
        logger = AuditLogger(secret=b"secret")
        assert logger.count() == 0

    def test_record(self):
        logger = AuditLogger(secret=b"secret")
        event = logger.record("breed", agent="a1", score=0.95)
        assert event.seq == 1
        assert event.category == "breed"
        assert event.payload == {"agent": "a1", "score": 0.95}

    def test_sequence_increments(self):
        logger = AuditLogger(secret=b"secret")
        e1 = logger.record("a")
        e2 = logger.record("b")
        assert e2.seq == e1.seq + 1

    def test_tail(self):
        logger = AuditLogger(secret=b"secret")
        for i in range(5):
            logger.record("evt", n=i)
        tail = logger.tail(3)
        assert len(tail) == 3
        assert tail[-1].payload["n"] == 4

    def test_by_category(self):
        logger = AuditLogger(secret=b"secret")
        logger.record("breed", x=1)
        logger.record("trap", x=2)
        logger.record("breed", x=3)
        assert len(logger.by_category("breed")) == 2

    def test_since(self):
        logger = AuditLogger(secret=b"secret")
        before = time.time()
        logger.record("a")
        logger.record("b")
        after = time.time()
        assert len(logger.since(before)) == 2
        assert len(logger.since(after + 1)) == 0

    def test_verify_clean(self):
        logger = AuditLogger(secret=b"secret")
        for i in range(10):
            logger.record("evt", i=i)
        logger.verify()

    def test_verify_tampered(self):
        logger = AuditLogger(secret=b"secret")
        for i in range(5):
            logger.record("evt", i=i)
        logger._events[2].payload["i"] = 999
        with pytest.raises(AuditTamperError):
            logger.verify()

    def test_verify_range(self):
        logger = AuditLogger(secret=b"secret")
        for i in range(10):
            logger.record("evt", i=i)
        assert logger.verify_range(1, 5) is True

    def test_verify_range_tampered(self):
        logger = AuditLogger(secret=b"secret")
        for i in range(10):
            logger.record("evt", i=i)
        logger._events[5].payload["i"] = 999
        assert logger.verify_range(1, 5) is True
        assert logger.verify_range(5, 7) is False

    def test_export_import_roundtrip(self):
        logger = AuditLogger(secret=b"secret")
        for i in range(5):
            logger.record("evt", i=i)
        data = logger.export_json()
        logger2 = AuditLogger(secret=b"secret")
        logger2.import_json(data)
        assert logger2.count() == 5
        logger2.verify()

    def test_max_events_eviction(self):
        logger = AuditLogger(secret=b"secret", max_events=3)
        for i in range(5):
            logger.record("evt", i=i)
        assert logger.count() == 3
        assert logger.tail(1)[0].payload["i"] == 4

    def test_repr(self):
        logger = AuditLogger(secret=b"secret")
        assert "AuditLogger" in repr(logger)
