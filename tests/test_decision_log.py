"""Tests for logos.decision_log — architectural decision records."""

import json
import tempfile
from datetime import datetime, timezone

import pytest

from logos.decision_log import (
    DecisionLog,
    DecisionRecord,
    DecisionRecords,
    DecisionType,
)


class TestDecisionRecord:
    def test_create(self):
        rec = DecisionRecord(
            id="ADR-0001",
            title="Use WAL for fleet persistence",
            decision_type=DecisionType.ARCHITECTURE,
            decided_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            decided_by="kimi1",
            context="Fleet needs crash-safe storage",
            decision="Adopt append-only WAL",
            rationale="",
        )
        assert rec.id == "ADR-0001"
        assert rec.title == "Use WAL for fleet persistence"

    def test_to_dict(self):
        rec = DecisionRecord(
            id="ADR-0001",
            title="Use WAL",
            decision_type=DecisionType.ARCHITECTURE,
            decided_at=datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
            decided_by="kimi1",
            context="Need persistence",
            decision="Use WAL",
            rationale="WAL is append-only",
            alternatives=["SQLite", "JSON"],
            components=["swarm"],
            tags=["persistence"],
        )
        d = rec.to_dict()
        assert d["id"] == "ADR-0001"
        assert d["decision_type"] == "architecture"
        assert d["alternatives"] == ["SQLite", "JSON"]

    def test_from_dict(self):
        rec = DecisionRecord(
            id="ADR-0001",
            title="Use WAL",
            decision_type=DecisionType.ARCHITECTURE,
            decided_at=datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
            decided_by="kimi1",
            context="Need persistence",
            decision="Use WAL",
            rationale="",
        )
        d = rec.to_dict()
        rec2 = DecisionRecord.from_dict(d)
        assert rec2.id == rec.id
        assert rec2.title == rec.title
        assert rec2.decided_at == rec.decided_at


class TestDecisionLog:
    def test_create(self):
        log = DecisionLog()
        assert len(log._records) == 0

    def test_record(self):
        log = DecisionLog()
        rec = log.record(
            title="Use WAL",
            decision_type=DecisionType.ARCHITECTURE,
            decided_by="kimi1",
            context="Need persistence",
            decision="Use WAL",
            rationale="",
        )
        assert rec.id.startswith("ADR-")
        assert rec.title == "Use WAL"
        assert len(log._records) == 1

    def test_get(self):
        log = DecisionLog()
        rec = log.record(
            title="Use WAL",
            decision_type=DecisionType.ARCHITECTURE,
            decided_by="kimi1",
            context="Need persistence",
            decision="Use WAL",
            rationale="",
            record_id="ADR-0001",
        )
        assert log.get("ADR-0001") == rec
        assert log.get("ADR-9999") is None

    def test_supersede(self):
        log = DecisionLog()
        log.record(
            title="Old",
            decision_type=DecisionType.ARCHITECTURE,
            decided_by="kimi1",
            context="Old",
            decision="Old",
            rationale="",
            record_id="ADR-0001",
        )
        log.record(
            title="New",
            decision_type=DecisionType.ARCHITECTURE,
            decided_by="kimi1",
            context="New",
            decision="New",
            rationale="",
            record_id="ADR-0002",
        )
        assert log.supersede("ADR-0001", "ADR-0002") is True
        assert log.get("ADR-0001").superseded_by == "ADR-0002"
        assert log.supersede("ADR-9999", "ADR-0002") is False

    def test_set_outcome(self):
        log = DecisionLog()
        log.record(
            title="Test",
            decision_type=DecisionType.ARCHITECTURE,
            decided_by="kimi1",
            context="Test",
            decision="Test",
            rationale="",
            record_id="ADR-0001",
        )
        assert log.set_outcome("ADR-0001", "Success") is True
        assert log.get("ADR-0001").outcome == "Success"
        assert log.set_outcome("ADR-9999", "Success") is False

    def test_query_by_type(self):
        log = DecisionLog()
        log.record(
            title="Arch",
            decision_type=DecisionType.ARCHITECTURE,
            decided_by="kimi1",
            context="Test",
            decision="Test",
            rationale="",
        )
        log.record(
            title="Tech",
            decision_type=DecisionType.TECHNOLOGY,
            decided_by="kimi1",
            context="Test",
            decision="Test",
            rationale="",
        )
        results = log.query(decision_type=DecisionType.ARCHITECTURE)
        assert results.total == 1
        assert results.records[0].title == "Arch"

    def test_query_by_component(self):
        log = DecisionLog()
        log.record(
            title="Test",
            decision_type=DecisionType.ARCHITECTURE,
            decided_by="kimi1",
            context="Test",
            decision="Test",
            rationale="",
            components=["swarm"],
        )
        results = log.query(component="swarm")
        assert results.total == 1
        results2 = log.query(component="nexus")
        assert results2.total == 0

    def test_query_by_tag(self):
        log = DecisionLog()
        log.record(
            title="Test",
            decision_type=DecisionType.ARCHITECTURE,
            decided_by="kimi1",
            context="Test",
            decision="Test",
            rationale="",
            tags=["persistence"],
        )
        results = log.query(tag="persistence")
        assert results.total == 1

    def test_query_text_search(self):
        log = DecisionLog()
        log.record(
            title="WAL design",
            decision_type=DecisionType.ARCHITECTURE,
            decided_by="kimi1",
            context="We need append-only storage",
            decision="Use WAL",
            rationale="",
        )
        results = log.query(text_search="append-only")
        assert results.total == 1
        results2 = log.query(text_search="nonexistent")
        assert results2.total == 0

    def test_query_topic(self):
        log = DecisionLog()
        log.record(
            title="WAL design",
            decision_type=DecisionType.ARCHITECTURE,
            decided_by="kimi1",
            context="We need storage",
            decision="Use WAL",
            rationale="",
            tags=["storage"],
        )
        results = log.query(topic="storage")
        assert results.total == 1

    def test_query_exclude_superseded(self):
        log = DecisionLog()
        log.record(
            title="Old",
            decision_type=DecisionType.ARCHITECTURE,
            decided_by="kimi1",
            context="Old",
            decision="Old",
            rationale="",
            record_id="ADR-0001",
        )
        log.record(
            title="New",
            decision_type=DecisionType.ARCHITECTURE,
            decided_by="kimi1",
            context="New",
            decision="New",
            rationale="",
            record_id="ADR-0002",
        )
        log.supersede("ADR-0001", "ADR-0002")
        results = log.query()
        assert results.total == 1
        results_all = log.query(include_superseded=True)
        assert results_all.total == 2

    def test_all_records(self):
        log = DecisionLog()
        log.record(
            title="First",
            decision_type=DecisionType.ARCHITECTURE,
            decided_by="kimi1",
            context="Test",
            decision="Test",
            rationale="",
        )
        log.record(
            title="Second",
            decision_type=DecisionType.ARCHITECTURE,
            decided_by="kimi1",
            context="Test",
            decision="Test",
            rationale="",
        )
        results = log.all_records()
        assert results.total == 2
        # Should be sorted by decided_at descending
        assert results.records[0].title == "Second"

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/decisions.json"
            log = DecisionLog(path)
            log.record(
                title="Test",
                decision_type=DecisionType.ARCHITECTURE,
                decided_by="kimi1",
                context="Test",
                decision="Test",
                rationale="",
                record_id="ADR-0001",
            )
            # Re-create from disk
            log2 = DecisionLog(path)
            assert log2.get("ADR-0001") is not None
            assert log2.get("ADR-0001").title == "Test"

    def test_repr(self):
        log = DecisionLog()
        assert "DecisionLog" in repr(log)

    def test_record_repr(self):
        rec = DecisionRecord(
            id="ADR-0001",
            title="Test",
            decision_type=DecisionType.ARCHITECTURE,
            decided_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            decided_by="kimi1",
            context="Test",
            decision="Test",
            rationale="",
        )
        assert "ADR-0001" in repr(rec)
