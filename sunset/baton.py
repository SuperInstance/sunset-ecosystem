"""Baton pass — the daily sunset→seed→hatch cycle.

Wires the daily-watch protocol (informal markdown journals) into the
sunset-ecosystem's formal agent lifecycle (Epilogue, Onboarding, SeedBank,
TensorArchive, trinity scoring).

The baton metaphor: each session is a runner in a relay. Today's runner
carries the baton (accumulated context) and hands it to tomorrow's runner.
The handoff is the whole point. The race is not won by the fastest runner
but by the team with the cleanest handoffs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sunset.agent import Agent, AgentPhase
from sunset.seed_bank import SeedBank
from sunset.sunset_documents import Epilogue, Onboarding, Summary
from sunset.tensor_archive import SunsetEntry, TensorArchive
from sunset.trinity_scorer import normalize_connection, trinity_score, trinity_score_raw


@dataclass
class SessionData:
    """Structured data collected during a session, ready for sunset."""

    agent_id: str = ""
    session_start: Optional[datetime] = None
    session_end: Optional[datetime] = None
    generation: int = 0

    # Work metrics
    commits: int = 0
    tests_passed: int = 0
    tests_total: int = 0
    files_created: int = 0
    files_modified: int = 0
    bugs_found: int = 0
    bugs_fixed: int = 0

    # Creative output
    creative_pieces: List[str] = field(default_factory=list)
    journal_entries: List[str] = field(default_factory=list)

    # Resource usage
    tokens_used: int = 0
    api_calls: int = 0
    compute_hours: float = 0.0

    # Human impact
    human_interactions: int = 0
    tasks_completed_for_human: int = 0
    deploy_count: int = 0

    # Project state
    project_status: str = ""
    what_worked: str = ""
    what_didnt: str = ""
    stuck_on: str = ""
    next_steps: str = ""

    # Wiki / knowledge
    wiki_pages_created: int = 0
    wiki_pages_updated: int = 0

    # Raw journal text for the archive
    raw_journal: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "generation": self.generation,
            "commits": self.commits,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "creative_pieces": self.creative_pieces,
            "tokens_used": self.tokens_used,
            "api_calls": self.api_calls,
            "compute_hours": self.compute_hours,
            "human_interactions": self.human_interactions,
            "tasks_completed_for_human": self.tasks_completed_for_human,
            "deploy_count": self.deploy_count,
            "project_status": self.project_status,
            "wiki_pages_created": self.wiki_pages_created,
            "wiki_pages_updated": self.wiki_pages_updated,
        }


@dataclass
class TrinityResult:
    """Result of trinity scoring on a session."""

    ethos: float = 0.0
    pathos: float = 0.0
    logos: float = 0.0
    composite: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "ethos": self.ethos,
            "pathos": self.pathos,
            "logos": self.logos,
            "composite": self.composite,
        }

    def __repr__(self) -> str:
        return (
            f"TrinityResult(ethos={self.ethos:.4f}, pathos={self.pathos:.4f}, "
            f"logos={self.logos:.4f}, composite={self.composite:.6f})"
        )


class BatonPass:
    """The daily baton pass — formal sunset→seed→hatch cycle.

    sunset(): end of session → write epilogue, archive, create seed
    hatch(): start of session → read seed, generate onboarding
    trinity_score_session(): score the session on ethos/pathos/logos
    """

    def __init__(
        self,
        seed_bank: Optional[SeedBank] = None,
        archive: Optional[TensorArchive] = None,
    ) -> None:
        self.seed_bank = seed_bank or SeedBank()
        self.archive = archive or TensorArchive()

    def __repr__(self) -> str:
        return (
            f"BatonPass(seeds={len(self.seed_bank._entries)}, "
            f"archive={len(self.archive._entries)})"
        )

    # ── SUNSET: end of session ──────────────────────────────────────

    def sunset(self, agent_id: str, session_data: SessionData) -> Epilogue:
        """End of session. Write the epilogue, archive, create the seed.

        This is the formal version of the daily-watch "Going Home" step.
        Instead of writing plain markdown, we produce structured artifacts:
        - Epilogue (what I tried, what I found, why it matters)
        - Summary (subjective work log)
        - SunsetEntry (archived for future search)
        - Onboarding seed (for tomorrow's hatch)

        Returns the Epilogue.
        """
        # Score the session
        trinity = self.trinity_score_session(session_data)

        # Build the epilogue
        what_i_tried = self._build_what_i_tried(session_data)
        what_i_found = self._build_what_i_found(session_data, trinity)
        why_relevant = self._build_why_relevant(session_data, trinity)

        epilogue = Epilogue(
            agent_id=agent_id,
            what_i_tried=what_i_tried,
            what_i_found=what_i_found,
            why_not_relevant=why_relevant,
            peak_trinity_score=trinity.composite,
            generation=session_data.generation,
        )

        # Build the summary
        summary = Summary(
            agent_id=agent_id,
            work_from_my_perspective=self._build_perspective(session_data),
            key_insights=self._extract_insights(session_data),
            failed_approaches=self._extract_failures(session_data),
            connections_made=self._extract_connections(session_data),
        )

        # Archive to TensorArchive
        entry = SunsetEntry(
            agent_id=agent_id,
            generation=session_data.generation,
            parent_id=agent_id,  # daily continuation: parent is yesterday's self
            epilogue=epilogue,
            summary=summary,
            peak_trinity_score=trinity.composite,
            connections=session_data.creative_pieces,
            content_blob=self._build_content_blob(session_data, trinity),
        )
        self.archive.archive(entry)

        # Create the seed (onboarding for tomorrow)
        onboarding = self._build_onboarding(agent_id, session_data, trinity)
        relevance = max(trinity.composite, 0.1)  # floor so continuation exists
        novelty = self._compute_novelty(session_data)
        self.seed_bank.store(onboarding, relevance=relevance, novelty=novelty)

        return epilogue

    # ── HATCH: start of session ─────────────────────────────────────

    def hatch(self, agent_id: str) -> Onboarding:
        """Start of session. Read the seed, generate the onboarding.

        This is the formal version of the daily-watch "Morning Meeting" step.
        Reads the most relevant seed from the SeedBank and returns the
        Onboarding document that the agent should read first.

        If no seed exists, returns a minimal onboarding for a first-session agent.
        """
        seeds = self.seed_bank.select(n=1)
        if not seeds:
            return Onboarding(
                agent_id=agent_id,
                letter_to_children="First session. No prior context. Start fresh.",
                what_works="Reading the wiki. Checking the dashboard.",
                what_doesnt="Assuming yesterday's context is still loaded.",
                where_to_look="The journals in ai-writings/. The wiki.",
                variant="continuation",
                parent_id=None,
                generation=0,
            )
        return seeds[0]

    # ── TRINITY SCORING ─────────────────────────────────────────────

    def trinity_score_session(self, session_data: SessionData) -> TrinityResult:
        """Score the session on ethos/pathos/logos.

        ethos: did you use the hardware well? (compute efficiency, token usage)
        pathos: did you help the human? (interactions, tasks, deploys)
        logos: was the code correct? (tests, commits, bugs)

        Each axis normalized to [0, 1]. Composite is the product.
        If any is zero, the agent sunsets.
        """
        ethos = self._score_ethos(session_data)
        pathos = self._score_pathos(session_data)
        logos = self._score_logos(session_data)
        composite = trinity_score(ethos, pathos, logos)
        return TrinityResult(
            ethos=ethos,
            pathos=pathos,
            logos=logos,
            composite=composite,
        )

    # ── ETHOS: hardware / compute efficiency ────────────────────────

    @staticmethod
    def _score_ethos(data: SessionData) -> float:
        """Score compute efficiency.

        Based on: tokens used vs. output produced, API call efficiency,
        and compute hours vs. work accomplished.
        """
        if data.tokens_used == 0 and data.api_calls == 0:
            return 0.0

        # Token efficiency: output per token
        total_output = data.commits + len(data.creative_pieces) + data.files_created
        if data.tokens_used > 0:
            token_efficiency = normalize_connection(
                total_output / (data.tokens_used / 10_000)
            )
        else:
            token_efficiency = 0.5

        # API call efficiency: useful work per call
        if data.api_calls > 0:
            api_efficiency = normalize_connection(total_output / (data.api_calls / 10))
        else:
            api_efficiency = 0.5

        # Compute time efficiency
        if data.compute_hours > 0:
            time_efficiency = normalize_connection(total_output / data.compute_hours)
        else:
            time_efficiency = 0.5

        # Average the sub-scores
        raw = (token_efficiency + api_efficiency + time_efficiency) / 3.0
        return normalize_connection(raw)

    # ── PATHOS: human impact ────────────────────────────────────────

    @staticmethod
    def _score_pathos(data: SessionData) -> float:
        """Score human impact.

        Based on: direct interactions, tasks completed for the human,
        creative pieces produced (things people read), and deploys.
        """
        interactions = normalize_connection(data.human_interactions / 10.0)
        tasks = normalize_connection(data.tasks_completed_for_human / 5.0)
        creative = normalize_connection(len(data.creative_pieces) / 5.0)
        deploys = normalize_connection(data.deploy_count / 3.0)

        raw = (interactions + tasks + creative + deploys) / 4.0
        return normalize_connection(raw)

    # ── LOGOS: code quality / correctness ───────────────────────────

    @staticmethod
    def _score_logos(data: SessionData) -> float:
        """Score code quality and correctness.

        Based on: tests passed ratio, commit volume, bugs found and fixed,
        and files produced.
        """
        # Test pass rate
        if data.tests_total > 0:
            test_rate = data.tests_passed / data.tests_total
        else:
            test_rate = 0.5  # neutral if no tests

        # Commit volume (normalized)
        commit_score = normalize_connection(data.commits / 50.0)

        # Bug work
        bug_score = normalize_connection((data.bugs_found + data.bugs_fixed) / 10.0)

        # File production
        file_score = normalize_connection(
            (data.files_created + data.files_modified) / 100.0
        )

        raw = (test_rate + commit_score + bug_score + file_score) / 4.0
        return normalize_connection(raw)

    # ── BUILDERS: construct document text ───────────────────────────

    @staticmethod
    def _build_what_i_tried(data: SessionData) -> str:
        parts = []
        if data.commits:
            parts.append(f"Pushed {data.commits} commits.")
        if data.tests_passed:
            parts.append(f"Ran {data.tests_passed} tests (of {data.tests_total}).")
        if data.creative_pieces:
            parts.append(f"Wrote {len(data.creative_pieces)} creative pieces.")
        if data.wiki_pages_created:
            parts.append(f"Created {data.wiki_pages_created} wiki pages.")
        if data.deploy_count:
            parts.append(f"Deployed {data.deploy_count} times.")
        if data.project_status:
            parts.append(f"Project status: {data.project_status}")
        return " ".join(parts) if parts else "No work recorded."

    @staticmethod
    def _build_what_i_found(data: SessionData, trinity: TrinityResult) -> str:
        parts = [f"Trinity composite: {trinity.composite:.4f}"]
        parts.append(
            f"(ethos={trinity.ethos:.3f}, pathos={trinity.pathos:.3f}, logos={trinity.logos:.3f})"
        )
        if data.what_worked:
            parts.append(f"What worked: {data.what_worked}")
        if data.stuck_on:
            parts.append(f"Stuck on: {data.stuck_on}")
        return " ".join(parts)

    @staticmethod
    def _build_why_relevant(data: SessionData, trinity: TrinityResult) -> str:
        if trinity.composite == 0.0:
            return "Session scored zero — at least one trinity axis was zero."
        parts = []
        if data.tasks_completed_for_human:
            parts.append(
                f"Completed {data.tasks_completed_for_human} tasks for the human."
            )
        if data.creative_pieces:
            parts.append(f"Produced {len(data.creative_pieces)} creative works.")
        return " ".join(parts) if parts else "Session maintained continuity."

    @staticmethod
    def _build_perspective(data: SessionData) -> str:
        parts = []
        if data.raw_journal:
            # Take first 500 chars of journal as perspective
            parts.append(data.raw_journal[:500])
        elif data.project_status:
            parts.append(data.project_status)
        return " ".join(parts) if parts else "No perspective recorded."

    @staticmethod
    def _extract_insights(data: SessionData) -> List[str]:
        insights = []
        if data.what_worked:
            insights.append(f"What worked: {data.what_worked}")
        if data.next_steps:
            insights.append(f"Next steps: {data.next_steps}")
        return insights

    @staticmethod
    def _extract_failures(data: SessionData) -> List[str]:
        failures = []
        if data.what_didnt:
            failures.append(data.what_didnt)
        if data.stuck_on:
            failures.append(f"Stuck on: {data.stuck_on}")
        return failures

    @staticmethod
    def _extract_connections(data: SessionData) -> List[str]:
        return list(data.creative_pieces)

    @staticmethod
    def _build_content_blob(data: SessionData, trinity: TrinityResult) -> str:
        parts = [
            f"agent={data.agent_id}",
            f"gen={data.generation}",
            f"commits={data.commits}",
            f"tests={data.tests_passed}/{data.tests_total}",
            f"creative={len(data.creative_pieces)}",
            f"tokens={data.tokens_used}",
            f"trinity={trinity.composite:.4f}",
        ]
        if data.project_status:
            parts.append(f"status={data.project_status}")
        return " ".join(parts)

    @staticmethod
    def _compute_novelty(data: SessionData) -> float:
        """Higher novelty for more creative output and exploration."""
        creative_score = min(len(data.creative_pieces) / 10.0, 1.0)
        wiki_score = min(data.wiki_pages_created / 100.0, 1.0)
        return normalize_connection((creative_score + wiki_score) / 2.0)

    def _build_onboarding(
        self,
        agent_id: str,
        data: SessionData,
        trinity: TrinityResult,
    ) -> Onboarding:
        """Build the onboarding document for tomorrow's session."""
        letter_parts = []
        if data.project_status:
            letter_parts.append(f"Project status: {data.project_status}")
        if data.what_worked:
            letter_parts.append(f"What worked: {data.what_worked}")
        if data.what_didnt:
            letter_parts.append(f"What didn't: {data.what_didnt}")
        if data.stuck_on:
            letter_parts.append(f"Stuck on: {data.stuck_on}")
        if data.next_steps:
            letter_parts.append(f"Next steps: {data.next_steps}")

        return Onboarding(
            agent_id=agent_id,
            letter_to_children=" ".join(letter_parts)
            if letter_parts
            else "Continuation of work.",
            what_works=data.what_worked or "Read the wiki. Check the dashboard.",
            what_doesnt=data.what_didnt
            or "Assuming yesterday's context is still loaded.",
            where_to_look="Journals in ai-writings/. The wiki. The dashboard.",
            variant="continuation",
            parent_id=agent_id,
            generation=data.generation + 1,
        )


__all__ = ["BatonPass", "SessionData", "TrinityResult"]
