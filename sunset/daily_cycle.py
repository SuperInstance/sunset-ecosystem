"""Daily cycle — wires the baton pass into the daily-watch rhythm.

Morning (hatch): read seed, generate onboarding
  ↓
Day's work: agent operates, collects session data
  ↓
Evening (sunset): write epilogue, archive, create seed

This module provides the orchestration layer that connects
the informal daily-watch protocol (markdown journals, creative
pieces, onboarding docs) to the formal sunset-ecosystem classes
(Agent, Epilogue, Onboarding, SeedBank, TensorArchive, trinity).

Usage:
    cycle = DailyCycle(agent_id="lucineer")
    onboarding = cycle.morning()       # hatch — read seed, get onboarding
    # ... agent works through the day ...
    cycle.record(**metrics)             # accumulate session data
    epilogue = cycle.evening()          # sunset — archive, seed, epilogue
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sunset.agent import Agent, AgentPhase
from sunset.baton import BatonPass, SessionData, TrinityResult
from sunset.seed_bank import SeedBank
from sunset.sunset_documents import Epilogue, Onboarding
from sunset.tensor_archive import TensorArchive


class DailyCycle:
    """Orchestrates the daily hatch → work → sunset cycle.

    Wraps BatonPass with session-level bookkeeping:
    - Tracks an Agent through its lifecycle phases
    - Accumulates SessionData throughout the day
    - Calls BatonPass.sunset() at session end
    - Calls BatonPass.hatch() at session start
    """

    # Default paths for the daily-watch protocol
    JOURNAL_DIR = Path("/home/eileen/projects/ai-writings/journals")
    CREATIVE_DIR = Path("/home/eileen/projects/ai-writings")

    def __init__(
        self,
        agent_id: str,
        generation: int = 0,
        seed_bank: Optional[SeedBank] = None,
        archive: Optional[TensorArchive] = None,
    ) -> None:
        self.agent_id = agent_id
        self.agent = Agent(
            id=agent_id,
            generation=generation,
            phase=AgentPhase.INCUBATING,
        )
        self.baton = BatonPass(seed_bank=seed_bank, archive=archive)
        self.session_data = SessionData(
            agent_id=agent_id,
            generation=generation,
            session_start=datetime.now(timezone.utc),
        )
        self._onboarding: Optional[Onboarding] = None

    def __repr__(self) -> str:
        return (
            f"DailyCycle(agent={self.agent_id!r}, gen={self.agent.generation}, "
            f"phase={self.agent.phase.value!r})"
        )

    # ── MORNING: hatch ──────────────────────────────────────────────

    def morning(self) -> Onboarding:
        """Session start. Read the seed, generate the onboarding.

        Transitions agent INCUBATING → COMPETING.
        Returns the Onboarding document to read first.
        """
        self.agent.advance(AgentPhase.COMPETING)
        self._onboarding = self.baton.hatch(self.agent_id)
        return self._onboarding

    # ── DAY: record metrics ─────────────────────────────────────────

    def record(
        self,
        commits: int = 0,
        tests_passed: int = 0,
        tests_total: int = 0,
        files_created: int = 0,
        files_modified: int = 0,
        bugs_found: int = 0,
        bugs_fixed: int = 0,
        creative_pieces: Optional[List[str]] = None,
        tokens_used: int = 0,
        api_calls: int = 0,
        compute_hours: float = 0.0,
        human_interactions: int = 0,
        tasks_completed_for_human: int = 0,
        deploy_count: int = 0,
        project_status: str = "",
        what_worked: str = "",
        what_didnt: str = "",
        stuck_on: str = "",
        next_steps: str = "",
        wiki_pages_created: int = 0,
        wiki_pages_updated: int = 0,
        raw_journal: str = "",
    ) -> SessionData:
        """Accumulate session data throughout the day. Returns current snapshot."""
        d = self.session_data
        d.commits += commits
        d.tests_passed += tests_passed
        d.tests_total = max(d.tests_total, tests_total)  # total is absolute
        d.files_created += files_created
        d.files_modified += files_modified
        d.bugs_found += bugs_found
        d.bugs_fixed += bugs_fixed
        if creative_pieces:
            d.creative_pieces.extend(creative_pieces)
        d.tokens_used += tokens_used
        d.api_calls += api_calls
        d.compute_hours += compute_hours
        d.human_interactions += human_interactions
        d.tasks_completed_for_human += tasks_completed_for_human
        d.deploy_count += deploy_count
        if project_status:
            d.project_status = project_status
        if what_worked:
            d.what_worked = what_worked
        if what_didnt:
            d.what_didnt = what_didnt
        if stuck_on:
            d.stuck_on = stuck_on
        if next_steps:
            d.next_steps = next_steps
        d.wiki_pages_created += wiki_pages_created
        d.wiki_pages_updated += wiki_pages_updated
        if raw_journal:
            d.raw_journal = raw_journal
        return d

    # ── EVENING: sunset ─────────────────────────────────────────────

    def evening(self) -> Epilogue:
        """Session end. Write the epilogue, archive, create the seed.

        Transitions agent COMPETING → SUNSETTING → ASLEEP.
        Returns the Epilogue.
        """
        self.agent.advance(AgentPhase.SUNSETTING)
        self.session_data.session_end = datetime.now(timezone.utc)

        epilogue = self.baton.sunset(self.agent_id, self.session_data)

        self.agent.trinity_score = self.baton.trinity_score_session(
            self.session_data
        ).composite
        self.agent.advance(AgentPhase.ASLEEP)
        return epilogue

    # ── TRINITY: check score mid-session ────────────────────────────

    def check_trinity(self) -> TrinityResult:
        """Score the session so far without ending it. Useful for mid-day checks."""
        return self.baton.trinity_score_session(self.session_data)

    # ── REPORT: human-readable summary ──────────────────────────────

    def status_report(self) -> str:
        """Human-readable status report for the current session."""
        trinity = self.check_trinity()
        d = self.session_data
        lines = [
            f"Agent: {self.agent_id} (gen {self.agent.generation}, phase {self.agent.phase.value})",
            f"Trinity: ethos={trinity.ethos:.3f} pathos={trinity.pathos:.3f} logos={trinity.logos:.3f} → {trinity.composite:.4f}",
            f"Commits: {d.commits}",
            f"Tests: {d.tests_passed}/{d.tests_total}",
            f"Creative pieces: {len(d.creative_pieces)}",
            f"Wiki pages: {d.wiki_pages_created} created, {d.wiki_pages_updated} updated",
            f"Tokens: {d.tokens_used}",
        ]
        if d.project_status:
            lines.append(f"Status: {d.project_status}")
        if d.stuck_on:
            lines.append(f"Stuck: {d.stuck_on}")
        return "\n".join(lines)

    # ── SERIALIZATION: save/load session data ───────────────────────

    def save_session(self, path: Optional[Path] = None) -> Path:
        """Serialize session data to JSON. Returns the path written."""
        if path is None:
            path = self.JOURNAL_DIR / f"{self.agent_id}-session.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.session_data.to_dict()
        data["session_start"] = (
            self.session_data.session_start.isoformat()
            if self.session_data.session_start
            else None
        )
        data["session_end"] = (
            self.session_data.session_end.isoformat()
            if self.session_data.session_end
            else None
        )
        path.write_text(json.dumps(data, indent=2, default=str))
        return path

    def load_session(self, path: Path) -> SessionData:
        """Load session data from JSON."""
        data = json.loads(path.read_text())
        start = data.pop("session_start", None)
        end = data.pop("session_end", None)
        self.session_data = SessionData(**data)
        if start:
            self.session_data.session_start = datetime.fromisoformat(start)
        if end:
            self.session_data.session_end = datetime.fromisoformat(end)
        return self.session_data


__all__ = ["DailyCycle"]
