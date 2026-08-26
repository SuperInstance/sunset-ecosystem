"""Tests for the logos package."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from logos import (
    AgentGeneration,
    CodebaseState,
    DecisionLog,
    DecisionRecord,
    DecisionRecords,
    GenerationHistory,
    GenerationMemory,
    TrinityConnection,
    score_trinity_connection,
    survey_codebase,
)
from logos.decision_log import DecisionType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_codebase(tmp_path):
    """Create a small fake codebase."""
    # Python package
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('"""My package."""\n')
    (pkg / "core.py").write_text(
        "# TODO: refactor this\n"
        "import os\nimport json\n\n"
        'def hello():\n    return "hello"\n\n'
        "# FIXME: handle errors\n"
        'def world():\n    return "world"\n'
    )
    (pkg / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    # Tests
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("")
    (tests_dir / "test_core.py").write_text(
        "import pytest\nfrom mypkg.core import hello\n\n"
        'def test_hello():\n    assert hello() == "hello"\n'
    )
    # Non-code file
    (tmp_path / "README.md").write_text("# My Package\n")
    return tmp_path


@pytest.fixture
def tmp_store(tmp_path):
    """Return a temp file path for JSON storage."""
    return str(tmp_path / "store.json")


# ---------------------------------------------------------------------------
# CodebaseState
# ---------------------------------------------------------------------------


class TestCodebaseState:
    def test_repr(self):
        s = CodebaseState(root="/tmp/test")
        assert "CodebaseState" in repr(s)
        assert "/tmp/test" in repr(s)

    def test_survey_nonexistent_dir(self):
        state = survey_codebase("/nonexistent/path/xyz")
        assert state.errors
        assert state.file_count == 0

    def test_survey_counts_files(self, tmp_codebase):
        state = survey_codebase(str(tmp_codebase))
        assert state.file_count >= 5
        assert state.total_lines > 0

    def test_language_breakdown(self, tmp_codebase):
        state = survey_codebase(str(tmp_codebase))
        assert "Python" in state.language_breakdown
        assert "Markdown" in state.language_breakdown

    def test_architecture_patterns(self, tmp_codebase):
        state = survey_codebase(str(tmp_codebase))
        assert "mypkg" in state.architecture_patterns.get("module_dirs", [])

    def test_technical_debt(self, tmp_codebase):
        state = survey_codebase(str(tmp_codebase))
        assert len(state.technical_debt.get("TODO", [])) >= 1
        assert len(state.technical_debt.get("FIXME", [])) >= 1

    def test_imported_packages(self, tmp_codebase):
        state = survey_codebase(str(tmp_codebase))
        pkgs = state.architecture_patterns.get("imported_packages", [])
        assert "os" in pkgs or "json" in pkgs


# ---------------------------------------------------------------------------
# DecisionLog
# ---------------------------------------------------------------------------


class TestDecisionLog:
    def test_record_and_get(self):
        log = DecisionLog()
        rec = log.record(
            title="Use Python 3.12",
            decision_type=DecisionType.TECHNOLOGY,
            decided_by="team",
            context="Need modern features",
            decision="Upgrade to Python 3.12",
            rationale="Pattern matching, perf improvements",
        )
        assert rec.id.startswith("ADR-")
        assert log.get(rec.id) is rec

    def test_repr(self):
        log = DecisionLog()
        assert "DecisionLog" in repr(log)

    def test_query_by_type(self):
        log = DecisionLog()
        log.record("t1", DecisionType.TECHNOLOGY, "a", "c", "d")
        log.record("t2", DecisionType.ARCHITECTURE, "a", "c", "d")
        results = log.query(decision_type=DecisionType.TECHNOLOGY)
        assert results.total == 1

    def test_query_by_text(self):
        log = DecisionLog()
        log.record("API design", DecisionType.API, "a", "c", "Use REST")
        log.record("Database", DecisionType.DATA_MODEL, "a", "c", "Use SQL")
        results = log.query(text_search="REST")
        assert results.total == 1
        assert "REST" in results.records[0].decision

    def test_query_by_topic(self):
        log = DecisionLog()
        log.record("API", DecisionType.API, "a", "c", "d", tags=["api", "v2"])
        results = log.query(topic="api")
        assert results.total == 1

    def test_query_by_tag(self):
        log = DecisionLog()
        log.record("t1", DecisionType.OTHER, "a", "c", "d", tags=["infra"])
        results = log.query(tag="infra")
        assert results.total == 1

    def test_supersede(self):
        log = DecisionLog()
        old = log.record("old", DecisionType.OTHER, "a", "c", "d", record_id="ADR-0001")
        new = log.record("new", DecisionType.OTHER, "a", "c", "d", record_id="ADR-0002")
        log.supersede("ADR-0001", "ADR-0002")
        assert old.superseded_by == "ADR-0002"
        # Default query excludes superseded
        results = log.all_records()
        assert results.total == 1

    def test_set_outcome(self):
        log = DecisionLog()
        rec = log.record("t", DecisionType.OTHER, "a", "c", "d", record_id="ADR-0010")
        log.set_outcome("ADR-0010", "Worked great")
        assert rec.outcome == "Worked great"

    def test_persistence(self, tmp_store):
        log1 = DecisionLog(store_path=tmp_store)
        log1.record("Persisted", DecisionType.TECHNOLOGY, "a", "c", "d")
        log2 = DecisionLog(store_path=tmp_store)
        all_recs = log2.all_records()
        assert all_recs.total == 1
        assert all_recs.records[0].title == "Persisted"

    def test_record_round_trip(self):
        rec = DecisionRecord(
            id="ADR-0099",
            title="Test",
            decision_type=DecisionType.PATTERN,
            decided_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            decided_by="bot",
            context="ctx",
            decision="dec",
            rationale="rat",
            alternatives=["alt1"],
            outcome="out",
            components=["comp"],
            tags=["tag"],
            superseded_by=None,
        )
        d = rec.to_dict()
        rec2 = DecisionRecord.from_dict(d)
        assert rec2.id == rec.id
        assert rec2.decision_type == DecisionType.PATTERN


# ---------------------------------------------------------------------------
# GenerationMemory
# ---------------------------------------------------------------------------


class TestGenerationMemory:
    def test_register_and_get(self):
        mem = GenerationMemory()
        gen = mem.register("agent-1", "Alpha", 1, purpose="Build things")
        assert gen.agent_id == "agent-1"
        assert mem.get("agent-1") is gen

    def test_repr(self):
        mem = GenerationMemory()
        assert "GenerationMemory" in repr(mem)

    def test_sunset(self):
        mem = GenerationMemory()
        mem.register("agent-1", "Alpha", 1)
        ok = mem.sunset("agent-1", "Replaced by Beta", lessons=["Keep it simple"])
        assert ok
        gen = mem.get("agent-1")
        assert gen.sunset_at is not None
        assert gen.sunset_reason == "Replaced by Beta"
        assert "Keep it simple" in gen.lessons_learned

    def test_sunset_nonexistent(self):
        mem = GenerationMemory()
        assert not mem.sunset("nope", "reason")

    def test_parent_child(self):
        mem = GenerationMemory()
        mem.register("parent", "Gen1", 1)
        mem.register("child", "Gen2", 2, parent_id="parent")
        children = mem.get_children("parent")
        assert len(children) == 1
        assert children[0].agent_id == "child"

    def test_lineage(self):
        mem = GenerationMemory()
        mem.register("g1", "Gen1", 1)
        mem.register("g2", "Gen2", 2, parent_id="g1")
        mem.register("g3", "Gen3", 3, parent_id="g2")
        lineage = mem.get_lineage("g3")
        assert len(lineage) == 2
        assert lineage[0].agent_id == "g1"

    def test_get_history(self):
        mem = GenerationMemory()
        mem.register("a1", "A1", 1)
        mem.register("a2", "A2", 2, parent_id="a1")
        mem.sunset("a1", "old", patterns=["modular design"])
        history = mem.get_history()
        assert history.total_generations == 2
        assert history.active_agents == 1
        assert "modular design" in history.surviving_patterns

    def test_persistence(self, tmp_store):
        m1 = GenerationMemory(store_path=tmp_store)
        m1.register("x", "X", 1, purpose="testing")
        m2 = GenerationMemory(store_path=tmp_store)
        assert m2.get("x") is not None
        assert m2.get("x").name == "X"

    def test_generation_round_trip(self):
        gen = AgentGeneration(
            agent_id="g1",
            name="Test",
            generation=1,
            created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            sunset_at=datetime(2025, 7, 1, tzinfo=timezone.utc),
            purpose="test",
            achievements=["did stuff"],
            sunset_reason="done",
            onboarding_docs=["doc.md"],
            children_spawned=["g2"],
            parent_id=None,
            patterns_preserved=["modular"],
            lessons_learned=["keep it simple"],
            metadata={"key": "val"},
        )
        d = gen.to_dict()
        gen2 = AgentGeneration.from_dict(d)
        assert gen2.agent_id == gen.agent_id
        assert gen2.sunset_at is not None


# ---------------------------------------------------------------------------
# TrinityConnection
# ---------------------------------------------------------------------------


class TestTrinityConnection:
    def test_repr(self):
        tc = TrinityConnection(overall=0.75)
        r = repr(tc)
        assert "TrinityConnection" in r
        assert "0.75" in r

    def test_score_with_codebase(self, tmp_codebase):
        tc = score_trinity_connection(root=str(tmp_codebase))
        assert 0.0 <= tc.overall <= 1.0
        assert 0.0 <= tc.codebase_understanding <= 1.0
        assert 0.0 <= tc.integration_quality <= 1.0
        assert 0.0 <= tc.maintainability <= 1.0
        assert tc.details
        assert tc.recommendations

    def test_score_with_precomputed_state(self, tmp_codebase):
        state = survey_codebase(str(tmp_codebase))
        tc = score_trinity_connection(codebase_state=state)
        assert tc.overall > 0.0

    def test_score_with_decision_log(self, tmp_codebase):
        log = DecisionLog()
        for i in range(5):
            log.record(f"Decision {i}", DecisionType.OTHER, "team", "ctx", "dec")
        tc = score_trinity_connection(
            root=str(tmp_codebase),
            decision_log=log,
        )
        assert tc.codebase_understanding > 0.0

    def test_score_with_generation_memory(self, tmp_codebase):
        mem = GenerationMemory()
        mem.register("a1", "Alpha", 1, purpose="build")
        mem.sunset("a1", "retired", patterns=["modular"])
        tc = score_trinity_connection(
            root=str(tmp_codebase),
            generation_memory=mem,
        )
        assert tc.codebase_understanding > 0.0

    def test_score_empty_dir(self, tmp_path):
        tc = score_trinity_connection(root=str(tmp_path))
        assert tc.overall >= 0.0
        # Should have recommendations for improvement
        assert len(tc.recommendations) > 0

    def test_all_components_integrated(self, tmp_codebase):
        """Full integration test: codebase + decisions + generations."""
        log = DecisionLog()
        log.record(
            "Use pytest", DecisionType.TECHNOLOGY, "team", "Need testing", "pytest"
        )
        mem = GenerationMemory()
        mem.register("logos-1", "Logos Gen 1", 1, purpose="Code memory")
        state = survey_codebase(str(tmp_codebase))
        tc = score_trinity_connection(
            codebase_state=state,
            decision_log=log,
            generation_memory=mem,
        )
        assert tc.overall > 0.0
        assert all(
            0.0 <= v <= 1.0
            for v in [
                tc.overall,
                tc.codebase_understanding,
                tc.integration_quality,
                tc.maintainability,
            ]
        )
