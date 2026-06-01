"""fleet/agent_identity_bridge.py — Git-Agent Standard pattern integration.

Brings the SuperInstance git-agent-standard patterns into sunset-ecosystem:
- Agent identity as repo structure (CHARTER.md, STATE.md, TASK-BOARD.md, SKILLS.md)
- Bottle communication (async messages between agents via git)
- Abstraction planes (capability declaration system)
- Commit convention ([AGENT-NAME] message)
- Diary-based learning (DIARY/YYYY-MM-DD.md)

This module provides Python APIs for managing agent identity files,
reading/writing bottles, and declaring abstraction planes.

Usage:
    from fleet.agent_identity_bridge import AgentVessel, Bottle, AbstractionPlane

    vessel = AgentVessel.from_repo("/path/to/agent-repo")
    print(vessel.charter.purpose)
    print(vessel.state.health)

    # Send a bottle to another agent
    bottle = vessel.write_bottle(
        to="oracle1",
        content="Found a pattern in constraint-theory-core KD-tree..."
    )

    # Read bottles from other agents
    for bottle in vessel.read_bottles():
        print(bottle.from_agent, bottle.content)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Abstraction Planes

@dataclass
class AbstractionPlane:
    """
    An agent's home plane and capability range.

    Planes 1-5: concrete (code, math, engineering)
    Planes 6-7: abstract (architecture, philosophy)
    Planes 8+: meta (system design, ontology)
    """
    primary: int = 4
    reads_from: list[int] = field(default_factory=lambda: [3, 4, 5])
    writes_to: list[int] = field(default_factory=lambda: [2, 3, 4])
    floor: int = 2
    ceiling: int = 5
    compilers: list[dict] = field(default_factory=list)
    reasoning: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "AbstractionPlane":
        return cls(
            primary=d.get("primary") or d.get("primary_plane", 4),
            reads_from=d.get("reads_from", [3, 4, 5]),
            writes_to=d.get("writes_to", [2, 3, 4]),
            floor=d.get("floor", 2),
            ceiling=d.get("ceiling", 5),
            compilers=d.get("compilers", []),
            reasoning=d.get("reasoning", ""),
        )

    def to_dict(self) -> dict:
        return {
            "primary": self.primary,
            "reads_from": self.reads_from,
            "writes_to": self.writes_to,
            "floor": self.floor,
            "ceiling": self.ceiling,
            "compilers": self.compilers,
            "reasoning": self.reasoning,
        }

    def can_read(self, plane: int) -> bool:
        return self.floor <= plane <= self.ceiling and plane in self.reads_from

    def can_write(self, plane: int) -> bool:
        return self.floor <= plane <= self.ceiling and plane in self.writes_to


# ---------------------------------------------------------------------------
# Charter

@dataclass
class Charter:
    """An agent's CHARTER.md contents."""
    name: str = ""
    purpose: str = ""
    contracts: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> "Charter":
        """Parse CHARTER.md text."""
        charter = cls()
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("# Charter:"):
                charter.name = line.replace("# Charter:", "").strip()
            elif "## Purpose" in line:
                # Read next non-empty line
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        charter.purpose = lines[j].strip()
                        break
            elif "## Contracts" in line:
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith("-"):
                        charter.contracts.append(lines[j].replace("-", "").strip())
                    elif lines[j].startswith("##"):
                        break
            elif "## Constraints" in line:
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith("-"):
                        charter.constraints.append(lines[j].replace("-", "").strip())
                    elif lines[j].startswith("##"):
                        break
        return charter

    def to_text(self) -> str:
        lines = [
            f"# Charter: {self.name}",
            "",
            "## Purpose",
            self.purpose or "One sentence. What you exist to do.",
            "",
            "## Contracts",
        ]
        for c in self.contracts or ["What you promise to maintain"]:
            lines.append(f"- {c}")
        lines.extend(["", "## Constraints"])
        for c in self.constraints or ["Boundaries (what you don't touch)"]:
            lines.append(f"- {c}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# State

@dataclass
class AgentState:
    """An agent's STATE.md contents."""
    last_active: str = ""
    health: str = "🟢 ACTIVE"
    current_task: str = ""
    pending: int = 0
    blockers: list[str] = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> "AgentState":
        state = cls()
        for line in text.splitlines():
            if line.startswith("**Last active:**"):
                state.last_active = line.replace("**Last active:**", "").strip()
            elif line.startswith("**Health:**"):
                state.health = line.replace("**Health:**", "").strip()
            elif line.startswith("**Current task:**"):
                state.current_task = line.replace("**Current task:**", "").strip()
            elif line.startswith("**Pending:**"):
                val = line.replace("**Pending:**", "").strip()
                state.pending = int(re.search(r"\d+", val).group()) if re.search(r"\d+", val) else 0
            elif line.startswith("**Blockers:**"):
                rest = line.replace("**Blockers:**", "").strip()
                if rest and rest.lower() != "none":
                    state.blockers = [b.strip() for b in rest.split(",")]
        return state

    def to_text(self) -> str:
        lines = [
            f"# State: {self.health}",
            f"**Last active:** {self.last_active or datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Health:** {self.health}",
            f"**Current task:** {self.current_task or 'None'}",
            f"**Pending:** {self.pending} tasks in queue",
            f"**Blockers:** {', '.join(self.blockers) if self.blockers else 'None'}",
        ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Bottle

@dataclass
class Bottle:
    """A message between agents (for-fleet/ or from-fleet/)."""
    from_agent: str
    to_agent: str
    content: str
    timestamp: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> "Bottle":
        """Parse a BOTTLE-TO-*.md or MESSAGE-FROM-*.md file."""
        text = path.read_text(encoding="utf-8")
        # Extract to/from from filename
        name = path.stem
        from_agent = ""
        to_agent = ""
        if name.startswith("BOTTLE-TO-"):
            to_agent = name.replace("BOTTLE-TO-", "").split("-")[0]
        elif name.startswith("MESSAGE-FROM-"):
            from_agent = name.replace("MESSAGE-FROM-", "").split("-")[0]

        # Try to extract from header
        for line in text.splitlines()[:20]:
            if line.startswith("**From:**"):
                from_agent = line.replace("**From:**", "").strip()
            elif line.startswith("**To:**"):
                to_agent = line.replace("**To:**", "").strip()
            elif line.startswith("**Date:**"):
                timestamp = line.replace("**Date:**", "").strip()

        # Content is everything after the first ## or just the body
        content = text
        if "##" in text:
            content = text.split("##", 1)[1]
            content = content.split("\n", 1)[1] if "\n" in content else content

        return cls(
            from_agent=from_agent,
            to_agent=to_agent,
            content=content.strip(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def to_text(self) -> str:
        lines = [
            f"**From:** {self.from_agent}",
            f"**To:** {self.to_agent}",
            f"**Date:** {self.timestamp}",
            "",
            "## Message",
            "",
            self.content,
        ]
        return "\n".join(lines) + "\n"

    def filename(self) -> str:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.to_agent:
            return f"BOTTLE-TO-{self.to_agent}-{date}.md"
        else:
            return f"MESSAGE-FROM-{self.from_agent}-{date}.md"


# ---------------------------------------------------------------------------
# Task Board

@dataclass
class TaskBoard:
    """An agent's TASK-BOARD.md contents."""
    critical: list[str] = field(default_factory=list)
    high: list[str] = field(default_factory=list)
    medium: list[str] = field(default_factory=list)
    done: list[str] = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> "TaskBoard":
        board = cls()
        current_section = None
        for line in text.splitlines():
            if "## 🔴 Critical" in line:
                current_section = "critical"
            elif "## 🟠 High" in line:
                current_section = "high"
            elif "## 🟡 Medium" in line:
                current_section = "medium"
            elif "## ✅ Done" in line:
                current_section = "done"
            elif line.startswith("- [ ]") and current_section:
                getattr(board, current_section).append(line.replace("- [ ]", "").strip())
            elif line.startswith("- [x]") and current_section:
                getattr(board, current_section).append(line.replace("- [x]", "").strip())
        return board

    def to_text(self) -> str:
        lines = ["# Task Board"]
        for section, emoji in [("critical", "🔴"), ("high", "🟠"), ("medium", "🟡"), ("done", "✅")]:
            lines.extend(["", f"## {emoji} {section.capitalize()}"])
            tasks = getattr(self, section)
            if tasks:
                for t in tasks:
                    checked = "x" if section == "done" else " "
                    lines.append(f"- [{checked}] {t}")
            else:
                lines.append(f"- [ ] No {section} tasks")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Skills

@dataclass
class SkillEntry:
    """A single skill entry."""
    name: str
    description: str
    examples: list[str] = field(default_factory=list)

@dataclass
class Skills:
    """An agent's SKILLS.md contents."""
    core_skills: list[SkillEntry] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    learned: list[str] = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> "Skills":
        skills = cls()
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("## Core Skills"):
                i += 1
                while i < len(lines) and not lines[i].startswith("##"):
                    if lines[i].startswith("- **"):
                        name = lines[i].replace("- **", "").replace("**", "").split("—")[0].strip()
                        desc = ""
                        if "—" in lines[i]:
                            desc = lines[i].split("—", 1)[1].strip()
                        skills.core_skills.append(SkillEntry(name=name, description=desc))
                    i += 1
                continue
            elif line.startswith("## Tools"):
                i += 1
                while i < len(lines) and not lines[i].startswith("##"):
                    if lines[i].startswith("-"):
                        skills.tools.append(lines[i].replace("-", "").strip())
                    i += 1
                continue
            elif line.startswith("## What I've Learned"):
                i += 1
                while i < len(lines) and not lines[i].startswith("##"):
                    if lines[i].startswith("-"):
                        skills.learned.append(lines[i].replace("-", "").strip())
                    i += 1
                continue
            i += 1
        return skills

    def to_text(self) -> str:
        lines = ["# Skills"]
        lines.extend(["", "## Core Skills"])
        if self.core_skills:
            for s in self.core_skills:
                lines.append(f"- **{s.name}** — {s.description}")
        else:
            lines.append("- **None yet** — document what you can do")
        lines.extend(["", "## Tools"])
        for t in self.tools or ["Document your available tools"]:
            lines.append(f"- {t}")
        lines.extend(["", "## What I've Learned"])
        for l in self.learned or ["Document lessons from each task"]:
            lines.append(f"- {l}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Agent Vessel (the full repo identity)

class AgentVessel:
    """
    Represents an agent's repository identity.

    Maps the git-agent-standard repo structure into a Python object:
    - CHARTER.md → charter
    - STATE.md → state
    - TASK-BOARD.md → task_board
    - SKILLS.md → skills
    - ABSTRACTION.md → plane
    - DIARY/ → diary entries
    - for-fleet/ → outgoing bottles
    - from-fleet/ → incoming bottles
    """

    def __init__(self, repo_path: str | Path):
        self.repo_path = Path(repo_path)
        self.charter = Charter()
        self.state = AgentState()
        self.task_board = TaskBoard()
        self.skills = Skills()
        self.plane = AbstractionPlane()
        self._load()

    def _load(self) -> None:
        charter_path = self.repo_path / "CHARTER.md"
        if charter_path.exists():
            self.charter = Charter.from_text(charter_path.read_text())

        state_path = self.repo_path / "STATE.md"
        if state_path.exists():
            self.state = AgentState.from_text(state_path.read_text())

        task_path = self.repo_path / "TASK-BOARD.md"
        if task_path.exists():
            self.task_board = TaskBoard.from_text(task_path.read_text())

        skills_path = self.repo_path / "SKILLS.md"
        if skills_path.exists():
            self.skills = Skills.from_text(skills_path.read_text())

        plane_path = self.repo_path / "ABSTRACTION.md"
        if plane_path.exists():
            # ABSTRACTION.md is YAML-like or plain text
            text = plane_path.read_text()
            # Try YAML parse first, fallback to heuristic
            try:
                import yaml
                d = yaml.safe_load(text)
                if d:
                    self.plane = AbstractionPlane.from_dict(d)
            except ImportError:
                # Heuristic parse
                d = {}
                for line in text.splitlines():
                    if ":" in line and not line.startswith("#"):
                        key, val = line.split(":", 1)
                        key = key.strip()
                        val = val.strip()
                        if key == "primary_plane":
                            d["primary"] = int(val)
                        elif key == "reads_from":
                            d["reads_from"] = [int(x.strip()) for x in val.strip("[]").split(",")]
                        elif key == "writes_to":
                            d["writes_to"] = [int(x.strip()) for x in val.strip("[]").split(",")]
                        elif key == "floor":
                            d["floor"] = int(val)
                        elif key == "ceiling":
                            d["ceiling"] = int(val)
                self.plane = AbstractionPlane.from_dict(d)

    def save(self) -> None:
        """Write all identity files back to disk."""
        self.repo_path.mkdir(parents=True, exist_ok=True)
        (self.repo_path / "CHARTER.md").write_text(self.charter.to_text())
        (self.repo_path / "STATE.md").write_text(self.state.to_text())
        (self.repo_path / "TASK-BOARD.md").write_text(self.task_board.to_text())
        (self.repo_path / "SKILLS.md").write_text(self.skills.to_text())
        (self.repo_path / "ABSTRACTION.md").write_text(
            json.dumps(self.plane.to_dict(), indent=2)
        )

    def write_bottle(self, to: str, content: str, **metadata) -> Bottle:
        """Write an outgoing bottle to for-fleet/."""
        bottle = Bottle(
            from_agent=self.charter.name,
            to_agent=to,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )
        bottles_dir = self.repo_path / "for-fleet"
        bottles_dir.mkdir(exist_ok=True)
        path = bottles_dir / bottle.filename()
        # If file exists, append a number
        counter = 1
        original = path
        while path.exists():
            path = original.with_name(f"{original.stem}-{counter}{original.suffix}")
            counter += 1
        path.write_text(bottle.to_text())
        return bottle

    def read_bottles(self) -> list[Bottle]:
        """Read incoming bottles from from-fleet/."""
        bottles_dir = self.repo_path / "from-fleet"
        if not bottles_dir.exists():
            return []
        bottles = []
        for path in sorted(bottles_dir.glob("*.md")):
            bottles.append(Bottle.from_file(path))
        return bottles

    def read_diary(self) -> list[tuple[str, str]]:
        """Read diary entries (date, content)."""
        diary_dir = self.repo_path / "DIARY"
        if not diary_dir.exists():
            return []
        entries = []
        for path in sorted(diary_dir.glob("*.md")):
            date = path.stem
            entries.append((date, path.read_text()))
        return entries

    def write_diary(self, date: Optional[str] = None, content: str = "") -> Path:
        """Write a diary entry."""
        diary_dir = self.repo_path / "DIARY"
        diary_dir.mkdir(exist_ok=True)
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = diary_dir / f"{date}.md"
        path.write_text(content)
        return path

    @classmethod
    def from_repo(cls, repo_path: str | Path) -> "AgentVessel":
        """Load an agent vessel from a repo path."""
        return cls(repo_path)

    @classmethod
    def create(cls, repo_path: str | Path, name: str, purpose: str) -> "AgentVessel":
        """Create a new agent vessel with default identity files."""
        vessel = cls(repo_path)
        vessel.charter.name = name
        vessel.charter.purpose = purpose
        vessel.state.health = "🟢 ACTIVE"
        vessel.state.current_task = "Booting"
        vessel.save()
        return vessel

    def summary(self) -> dict:
        """Return a JSON-serializable summary."""
        return {
            "name": self.charter.name,
            "purpose": self.charter.purpose,
            "health": self.state.health,
            "current_task": self.state.current_task,
            "pending": self.state.pending,
            "blockers": self.state.blockers,
            "primary_plane": self.plane.primary,
            "bottles_out": len(list((self.repo_path / "for-fleet").glob("*.md"))) if (self.repo_path / "for-fleet").exists() else 0,
            "bottles_in": len(list((self.repo_path / "from-fleet").glob("*.md"))) if (self.repo_path / "from-fleet").exists() else 0,
            "diary_entries": len(list((self.repo_path / "DIARY").glob("*.md"))) if (self.repo_path / "DIARY").exists() else 0,
        }
