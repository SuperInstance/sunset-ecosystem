"""tests/test_agent_identity_bridge.py — Test suite for git-agent-standard bridge.

Covers:
- AbstractionPlane construction and capability checks
- Charter parsing and serialization
- AgentState parsing and serialization
- Bottle parsing, serialization, and filename generation
- TaskBoard parsing and serialization
- Skills parsing and serialization
- AgentVessel creation, save/load, bottle read/write
- Diary read/write
- Summary generation
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fleet.agent_identity_bridge import (
    AbstractionPlane,
    Charter,
    AgentState,
    Bottle,
    TaskBoard,
    Skills,
    SkillEntry,
    AgentVessel,
)


class TestAbstractionPlane:
    def test_defaults(self):
        p = AbstractionPlane()
        assert p.primary == 4
        assert p.floor == 2
        assert p.ceiling == 5

    def test_can_read(self):
        p = AbstractionPlane(reads_from=[3, 4, 5], floor=2, ceiling=5)
        assert p.can_read(4) is True
        assert p.can_read(6) is False
        assert p.can_read(1) is False

    def test_can_write(self):
        p = AbstractionPlane(writes_to=[2, 3, 4], floor=2, ceiling=5)
        assert p.can_write(3) is True
        assert p.can_write(5) is False

    def test_roundtrip(self):
        p = AbstractionPlane(primary=3, reads_from=[2, 3, 4], floor=1, ceiling=6)
        d = p.to_dict()
        p2 = AbstractionPlane.from_dict(d)
        assert p2.primary == p.primary
        assert p2.reads_from == p.reads_from
        assert p2.ceiling == p.ceiling


class TestCharter:
    def test_from_text(self):
        text = """# Charter: TestAgent

## Purpose
Build things.

## Contracts
- Maintain quality
- Ship on time

## Constraints
- Don't touch production
"""
        c = Charter.from_text(text)
        assert c.name == "TestAgent"
        assert c.purpose == "Build things."
        assert c.contracts == ["Maintain quality", "Ship on time"]
        assert c.constraints == ["Don't touch production"]

    def test_from_text_minimal(self):
        c = Charter.from_text("# Charter: Minimal")
        assert c.name == "Minimal"
        assert c.purpose == ""

    def test_to_text_roundtrip(self):
        c = Charter(name="Round", purpose="Test", contracts=["A"], constraints=["B"])
        text = c.to_text()
        c2 = Charter.from_text(text)
        assert c2.name == c.name
        assert c2.purpose == c.purpose
        assert c2.contracts == c.contracts
        assert c2.constraints == c.constraints


class TestAgentState:
    def test_from_text(self):
        text = """# State: 🟢 ACTIVE
**Last active:** 2026-06-01 12:00 UTC
**Health:** 🟢 ACTIVE
**Current task:** Building bridge
**Pending:** 3 tasks in queue
**Blockers:** None
"""
        s = AgentState.from_text(text)
        assert s.health == "🟢 ACTIVE"
        assert s.current_task == "Building bridge"
        assert s.pending == 3
        assert s.blockers == []

    def test_from_text_with_blockers(self):
        text = """# State: 🟡 DEGRADED
**Last active:** 2026-06-01
**Health:** 🟡 DEGRADED
**Current task:** Debugging
**Pending:** 5 tasks
**Blockers:** API timeout, missing token
"""
        s = AgentState.from_text(text)
        assert s.blockers == ["API timeout", "missing token"]

    def test_to_text_contains_fields(self):
        s = AgentState(health="🟢 ACTIVE", current_task="Test", pending=2)
        text = s.to_text()
        assert "Test" in text
        assert "2 tasks in queue" in text
        assert "🟢 ACTIVE" in text


class TestBottle:
    def test_construction(self):
        b = Bottle(
            from_agent="ccc",
            to_agent="oracle1",
            content="Hello",
            timestamp="2026-06-01T00:00:00+00:00",
        )
        assert b.from_agent == "ccc"
        assert b.to_agent == "oracle1"
        assert b.content == "Hello"

    def test_filename_outgoing(self):
        b = Bottle(from_agent="ccc", to_agent="oracle1", content="Hi", timestamp="")
        assert b.filename().startswith("BOTTLE-TO-oracle1-")

    def test_filename_incoming(self):
        b = Bottle(from_agent="oracle1", to_agent="", content="Hi", timestamp="")
        assert b.filename().startswith("MESSAGE-FROM-oracle1-")

    def test_to_text_roundtrip(self):
        b = Bottle(
            from_agent="ccc",
            to_agent="oracle1",
            content="Found pattern",
            timestamp="2026-06-01T00:00:00+00:00",
        )
        text = b.to_text()
        assert "Found pattern" in text
        assert "ccc" in text
        assert "oracle1" in text

    def test_from_file(self, tmp_path):
        path = tmp_path / "BOTTLE-TO-oracle1-2026-06-01.md"
        path.write_text(
            "**From:** ccc\n**To:** oracle1\n**Date:** 2026-06-01\n\n## Message\n\nHello!"
        )
        b = Bottle.from_file(path)
        assert b.from_agent == "ccc"
        assert b.to_agent == "oracle1"
        assert "Hello!" in b.content


class TestTaskBoard:
    def test_from_text(self):
        text = """# Task Board

## 🔴 Critical
- [ ] Fix bug

## 🟠 High
- [ ] Add feature

## 🟡 Medium
- [ ] Refactor

## ✅ Done
- [x] Initial setup
"""
        board = TaskBoard.from_text(text)
        assert board.critical == ["Fix bug"]
        assert board.high == ["Add feature"]
        assert board.medium == ["Refactor"]
        assert board.done == ["Initial setup"]

    def test_to_text_roundtrip(self):
        board = TaskBoard(
            critical=["A"],
            high=["B"],
            medium=["C"],
            done=["D"],
        )
        text = board.to_text()
        board2 = TaskBoard.from_text(text)
        assert board2.critical == board.critical
        assert board2.high == board.high
        assert board2.done == board.done


class TestSkills:
    def test_from_text(self):
        text = """# Skills

## Core Skills
- **Coding** — Write Python
- **Testing** — Write pytest

## Tools
- Python
- Git

## What I've Learned
- Always test first
"""
        skills = Skills.from_text(text)
        assert len(skills.core_skills) == 2
        assert skills.core_skills[0].name == "Coding"
        assert skills.tools == ["Python", "Git"]
        assert skills.learned == ["Always test first"]

    def test_to_text_roundtrip(self):
        skills = Skills(
            core_skills=[SkillEntry(name="A", description="B")],
            tools=["C"],
            learned=["D"],
        )
        text = skills.to_text()
        skills2 = Skills.from_text(text)
        assert skills2.core_skills[0].name == "A"
        assert skills2.tools == ["C"]
        assert skills2.learned == ["D"]


class TestAgentVessel:
    def test_create_and_load(self, tmp_path):
        repo = tmp_path / "test-agent"
        vessel = AgentVessel.create(repo, "TestAgent", "Build bridges")

        assert (repo / "CHARTER.md").exists()
        assert (repo / "STATE.md").exists()
        assert (repo / "TASK-BOARD.md").exists()
        assert (repo / "SKILLS.md").exists()
        assert (repo / "ABSTRACTION.md").exists()

        vessel2 = AgentVessel.from_repo(repo)
        assert vessel2.charter.name == "TestAgent"
        assert vessel2.charter.purpose == "Build bridges"
        assert vessel2.state.health == "🟢 ACTIVE"

    def test_bottle_write_and_read(self, tmp_path):
        repo = tmp_path / "test-agent"
        vessel = AgentVessel.create(repo, "Sender", "Send bottles")

        bottle = vessel.write_bottle(to="Receiver", content="Hello there")
        assert bottle.from_agent == "Sender"
        assert bottle.to_agent == "Receiver"

        # Simulate receiving by moving the bottle
        from_fleet = repo / "from-fleet"
        from_fleet.mkdir(exist_ok=True)
        (from_fleet / bottle.filename()).write_text(bottle.to_text())

        bottles = vessel.read_bottles()
        assert len(bottles) == 1
        assert bottles[0].from_agent == "Sender"
        assert "Hello there" in bottles[0].content

    def test_diary(self, tmp_path):
        repo = tmp_path / "test-agent"
        vessel = AgentVessel.create(repo, "Diarist", "Write diaries")

        path = vessel.write_diary(date="2026-06-01", content="Today I built a bridge")
        assert path.exists()

        entries = vessel.read_diary()
        assert len(entries) == 1
        assert entries[0][0] == "2026-06-01"
        assert "bridge" in entries[0][1]

    def test_summary(self, tmp_path):
        repo = tmp_path / "test-agent"
        vessel = AgentVessel.create(repo, "Summarizer", "Summarize")
        vessel.state.current_task = "Testing"
        vessel.state.pending = 5
        vessel.save()

        summary = vessel.summary()
        assert summary["name"] == "Summarizer"
        assert summary["current_task"] == "Testing"
        assert summary["pending"] == 5
        assert summary["bottles_out"] == 0
        assert summary["diary_entries"] == 0

    def test_abstraction_plane_yaml(self, tmp_path):
        repo = tmp_path / "test-agent"
        vessel = AgentVessel.create(repo, "Planar", "Test planes")
        (repo / "ABSTRACTION.md").write_text(
            "primary_plane: 3\nreads_from: [2, 3, 4]\nwrites_to: [2, 3]\nfloor: 1\nceiling: 6\n"
        )
        vessel2 = AgentVessel.from_repo(repo)
        assert vessel2.plane.primary == 3
        assert vessel2.plane.reads_from == [2, 3, 4]
        assert vessel2.plane.floor == 1

    def test_abstraction_plane_json(self, tmp_path):
        repo = tmp_path / "test-agent"
        vessel = AgentVessel.create(repo, "Planar", "Test planes")
        (repo / "ABSTRACTION.md").write_text(
            json.dumps({"primary": 5, "floor": 3, "ceiling": 7})
        )
        vessel2 = AgentVessel.from_repo(repo)
        assert vessel2.plane.primary == 5
        assert vessel2.plane.floor == 3
        assert vessel2.plane.ceiling == 7
