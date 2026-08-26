"""Tests for DecisionJournal — FLAME format logging.

Covers Decision dataclass, DecisionJournal persistence, convenience loggers,
and JSONL querying.
"""

import json
import time
from pathlib import Path

import pytest

from logos.decision_journal import (
    Decision,
    DecisionJournal,
    log_spawn,
    log_sunset,
    log_breed,
    log_human_command,
    get_decision_history,
)


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------


class TestDecision:
    def test_defaults(self):
        d = Decision(
            timestamp=1.0,
            why="test",
            what="action",
            expected="ok",
            actual="",
            confidence=0.5,
            scope="a",
        )
        assert d.timestamp == 1.0
        assert d.metadata == {}

    def test_to_dict(self):
        d = Decision(
            timestamp=1.0,
            why="test",
            what="action",
            expected="ok",
            actual="done",
            confidence=0.75,
            scope="a",
            metadata={"k": "v"},
        )
        data = d.to_dict()
        assert data["confidence"] == 0.75
        assert data["metadata"]["k"] == "v"

    def test_from_dict(self):
        data = {
            "timestamp": 2.0,
            "why": "q",
            "what": "w",
            "expected": "e",
            "actual": "r",
            "confidence": 0.9,
            "scope": "s",
        }
        d = Decision.from_dict(data)
        assert d.why == "q"
        assert d.confidence == 0.9
        assert d.metadata == {}

    def test_from_dict_missing_optional(self):
        data = {
            "timestamp": 1.0,
            "why": "q",
            "what": "w",
            "expected": "e",
            "scope": "s",
        }
        d = Decision.from_dict(data)
        assert d.actual == ""
        assert d.confidence == 0.0


# ---------------------------------------------------------------------------
# DecisionJournal in-memory
# ---------------------------------------------------------------------------


class TestDecisionJournalMemory:
    def test_empty(self):
        dj = DecisionJournal()
        assert dj.recent(10) == []
        assert dj.all_entries() == []

    def test_record(self):
        dj = DecisionJournal()
        d = dj.record(
            why="because", what="do_it", expected="win", confidence=0.9, scope="fleet"
        )
        assert len(dj.all_entries()) == 1
        assert d.why == "because"
        assert d.confidence == 0.9

    def test_recent_order(self):
        dj = DecisionJournal()
        dj.record(why="first", what="a", expected="ok", timestamp=1.0)
        dj.record(why="second", what="b", expected="ok", timestamp=2.0)
        recent = dj.recent(1)
        assert len(recent) == 1
        assert recent[0].timestamp == 2.0

    def test_update_actual(self):
        dj = DecisionJournal()
        dj.record(why="q", what="w", expected="e", timestamp=1.0)
        assert dj.update_actual(0, "done")
        assert dj.all_entries()[0].actual == "done"

    def test_update_actual_bad_index(self):
        dj = DecisionJournal()
        assert not dj.update_actual(0, "nope")


# ---------------------------------------------------------------------------
# DecisionJournal persistence
# ---------------------------------------------------------------------------


class TestDecisionJournalPersistence:
    def test_save_and_load(self, tmp_path):
        p = tmp_path / "journal.json"
        dj = DecisionJournal(store_path=str(p))
        dj.record(why="save", what="test", expected="ok", timestamp=1.0)
        dj2 = DecisionJournal(store_path=str(p))
        assert len(dj2.all_entries()) == 1
        assert dj2.all_entries()[0].why == "save"

    def test_load_corrupt(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        dj = DecisionJournal(store_path=str(p))
        assert dj.all_entries() == []

    def test_load_missing_keys(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps([{"timestamp": 1.0}]))
        dj = DecisionJournal(store_path=str(p))
        assert dj.all_entries() == []

    def test_no_save_without_path(self):
        dj = DecisionJournal()
        dj.record(why="mem", what="only", expected="ok")
        # should not crash
        assert len(dj.all_entries()) == 1


# ---------------------------------------------------------------------------
# Convenience loggers
# ---------------------------------------------------------------------------


class TestLogSpawn:
    def test_writes_record(self, tmp_path):
        p = tmp_path / "day.jsonl"
        rec = log_spawn(
            42, parents=(1, 2), generation=3, reason="test", journal_path=str(p)
        )
        assert rec["operation"] == "spawn"
        assert rec["agent_id"] == 42
        assert rec["parents"] == [1, 2]
        assert rec["generation"] == 3
        lines = p.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["agent_id"] == 42

    def test_none_parents_filtered(self, tmp_path):
        p = tmp_path / "day.jsonl"
        rec = log_spawn(42, parents=(None, 2), journal_path=str(p))
        assert rec["parents"] == [2]

    def test_default_reason(self, tmp_path):
        p = tmp_path / "day.jsonl"
        rec = log_spawn(1, journal_path=str(p))
        assert rec["reason"] == "fleet spawn"


class TestLogSunset:
    def test_writes_record(self, tmp_path):
        p = tmp_path / "day.jsonl"
        rec = log_sunset(7, "old", generation=2, journal_path=str(p))
        assert rec["operation"] == "sunset"
        assert rec["agent_id"] == 7
        assert rec["reason"] == "old"


class TestLogBreed:
    def test_with_two_parents(self, tmp_path):
        p = tmp_path / "day.jsonl"
        rec = log_breed(1, 2, 3, generation=1, journal_path=str(p))
        assert rec["operation"] == "breed"
        assert rec["parents"] == [1, 2]
        assert rec["agent_id"] == 3

    def test_with_one_parent(self, tmp_path):
        p = tmp_path / "day.jsonl"
        rec = log_breed(1, None, 3, journal_path=str(p))
        assert rec["parents"] == [1]


class TestLogHumanCommand:
    def test_confirmed(self, tmp_path):
        p = tmp_path / "day.jsonl"
        intent = type(
            "I",
            (),
            {
                "raw_command": "deploy",
                "action": "deploy",
                "is_destructive": lambda self: False,
            },
        )()
        rec = log_human_command(
            intent, confirmed=True, scope="all", journal_path=str(p)
        )
        assert rec["operation"] == "human_command"
        assert rec["actual"] == "confirmed"
        assert rec["confidence"] == 1.0
        assert rec["metadata"]["confirmed"] is True

    def test_pending(self, tmp_path):
        p = tmp_path / "day.jsonl"
        intent = type(
            "I",
            (),
            {
                "raw_command": "stop",
                "action": "stop",
                "is_destructive": lambda self: True,
            },
        )()
        rec = log_human_command(
            intent, confirmed=False, scope="node-1", journal_path=str(p)
        )
        assert rec["actual"] == "pending"
        assert rec["confidence"] == 0.5
        assert rec["metadata"]["destructive"] is True

    def test_missing_attrs(self, tmp_path):
        p = tmp_path / "day.jsonl"
        rec = log_human_command(
            object(), confirmed=True, scope="x", journal_path=str(p)
        )
        assert rec["what"] == "unknown → x"


# ---------------------------------------------------------------------------
# get_decision_history
# ---------------------------------------------------------------------------


class TestGetDecisionHistory:
    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        assert get_decision_history(journal_path=str(p)) == []

    def test_filter_by_agent(self, tmp_path):
        p = tmp_path / "day.jsonl"
        log_spawn(1, journal_path=str(p))
        log_sunset(2, "old", journal_path=str(p))
        matches = get_decision_history(agent_id=1, journal_path=str(p))
        assert len(matches) == 1
        assert matches[0]["operation"] == "spawn"

    def test_filter_by_operation(self, tmp_path):
        p = tmp_path / "day.jsonl"
        log_spawn(1, journal_path=str(p))
        log_sunset(2, "old", journal_path=str(p))
        matches = get_decision_history(operation="sunset", journal_path=str(p))
        assert len(matches) == 1
        assert matches[0]["agent_id"] == 2

    def test_bad_lines_skipped(self, tmp_path):
        p = tmp_path / "day.jsonl"
        p.write_text("not json\n")
        assert get_decision_history(journal_path=str(p)) == []

    def test_default_path_directory(self, tmp_path, monkeypatch):
        from logos import decision_journal as dj

        monkeypatch.setattr(
            dj, "_default_journal_path", lambda: tmp_path / "2024-01-01.jsonl"
        )
        # write then read
        p = tmp_path / "2024-01-01.jsonl"
        log_spawn(99, journal_path=str(p))
        matches = get_decision_history(agent_id=99)
        assert len(matches) == 1
