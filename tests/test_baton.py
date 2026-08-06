"""Tests for the BatonPass — the daily baton pass cycle.

Covers sunset(), hatch(), trinity scoring, SessionData accumulation,
DailyCycle orchestration, and integration with SeedBank/TensorArchive.
"""

import json
import pytest
from datetime import datetime, timezone

from sunset.agent import Agent, AgentPhase
from sunset.baton import BatonPass, SessionData, TrinityResult
from sunset.daily_cycle import DailyCycle
from sunset.seed_bank import SeedBank
from sunset.sunset_documents import Epilogue, Onboarding, Summary
from sunset.tensor_archive import TensorArchive
from sunset.trinity_scorer import trinity_score


# ─── Fixtures ───


@pytest.fixture
def baton():
    """Fresh BatonPass with empty seed bank and archive."""
    return BatonPass(seed_bank=SeedBank(), archive=TensorArchive())


@pytest.fixture
def session_data():
    """SessionData with Lucineer's real metrics from 2026-08-06."""
    return SessionData(
        agent_id="lucineer",
        generation=0,
        commits=311,
        tests_passed=317,
        tests_total=317,
        files_created=4929,
        files_modified=100,
        creative_pieces=[
            "15-the-hermit-crab-and-the-open-hatch.md",
            "15-darmok-at-the-noise-floor.md",
            "15-the-extraction-navigator.md",
            "15-the-extraction-engine.md",
            "15-the-salmonberry.md",
            "15-the-quality-brief.md",
            "15-the-kaleidoscope.md",
            "15-the-last-entry-before-compaction.md",
            "15-six-versions-of-one-day.md",
            "15-the-anchor-that-remembered-the-storm.md",
            "15-the-day-the-fleet-wrote-itself.md",
            "15-the-frequency-spectrum.md",
            "15-the-melody-underneath.md",
            "15-the-collogue.md",
            "15-molding-memories.md",
            "15-seed-mini-at-the-table.md",
            "15-seed-pro-at-the-table.md",
        ],
        tokens_used=2_500_000,
        api_calls=500,
        compute_hours=10.0,
        human_interactions=50,
        tasks_completed_for_human=15,
        deploy_count=5,
        project_status="Massive fleet build day. Wiki, Vectorize, Openrooms all live.",
        what_worked="Wiki as context management. Subagent dispatch. Creative output.",
        what_didnt="Tmux crash killed sessions. API key leaked. DeepInfra 401.",
        stuck_on="Song vocals below noise floor. DeepInfra key expired.",
        next_steps="Restart tmux sessions. Memory index rebuild. Openrooms seeding.",
        wiki_pages_created=700,
        wiki_pages_updated=0,
        raw_journal="Biggest single day in fleet history.",
    )


# ─── TrinityResult dataclass ───


class TestTrinityResult:
    def test_defaults(self):
        r = TrinityResult()
        assert r.ethos == 0.0
        assert r.pathos == 0.0
        assert r.logos == 0.0
        assert r.composite == 0.0

    def test_to_dict(self):
        r = TrinityResult(ethos=0.5, pathos=0.6, logos=0.7, composite=0.21)
        d = r.to_dict()
        assert d == {"ethos": 0.5, "pathos": 0.6, "logos": 0.7, "composite": 0.21}

    def test_repr(self):
        r = TrinityResult(ethos=0.5, pathos=0.6, logos=0.7, composite=0.21)
        s = repr(r)
        assert "0.5000" in s
        assert "0.6000" in s
        assert "0.7000" in s


# ─── SessionData ───


class TestSessionData:
    def test_defaults(self):
        d = SessionData()
        assert d.commits == 0
        assert d.creative_pieces == []
        assert d.agent_id == ""

    def test_to_dict(self):
        d = SessionData(agent_id="test", commits=5, tests_passed=10, tests_total=10)
        result = d.to_dict()
        assert result["agent_id"] == "test"
        assert result["commits"] == 5
        assert result["tests_passed"] == 10

    def test_with_lucineer_data(self, session_data):
        assert session_data.agent_id == "lucineer"
        assert session_data.commits == 311
        assert len(session_data.creative_pieces) == 17


# ─── BatonPass: trinity scoring ───


class TestBatonTrinityScoring:
    def test_zero_session_scores_zero(self, baton):
        """Empty session should score zero on all axes."""
        d = SessionData()
        result = baton.trinity_score_session(d)
        assert result.composite == 0.0

    def test_all_axes_nonzero(self, baton, session_data):
        """Lucineer's real session should produce positive scores on all axes."""
        result = baton.trinity_score_session(session_data)
        assert result.ethos > 0
        assert result.pathos > 0
        assert result.logos > 0
        assert result.composite > 0

    def test_ethos_zero_with_no_compute(self, baton):
        """No compute used, no output → ethos zero."""
        d = SessionData()
        assert baton._score_ethos(d) == 0.0

    def test_pathos_zero_with_no_human_impact(self, baton):
        """No human interactions → pathos zero."""
        d = SessionData()
        assert baton._score_pathos(d) == 0.0

    def test_logos_with_no_tests_is_neutral(self, baton):
        """No tests → logos doesn't zero out (neutral 0.5 for test_rate)."""
        d = SessionData(commits=10, files_created=5)
        logos = baton._score_logos(d)
        assert logos > 0  # not zeroed by missing tests

    def test_if_any_axis_zero_composite_zero(self, baton):
        """If pathos is zero (no human impact), composite is zero."""
        d = SessionData(
            commits=100,
            tests_passed=100,
            tests_total=100,
            tokens_used=1000,
            api_calls=50,
            compute_hours=5.0,
            files_created=50,
            human_interactions=0,
            tasks_completed_for_human=0,
            creative_pieces=[],
            deploy_count=0,
        )
        result = baton.trinity_score_session(d)
        assert result.pathos == 0.0
        assert result.composite == 0.0

    def test_perfect_scores(self, baton):
        """High metrics across all axes → high composite."""
        d = SessionData(
            commits=200,
            tests_passed=500,
            tests_total=500,
            files_created=500,
            creative_pieces=["a.md", "b.md", "c.md", "d.md", "e.md", "f.md"],
            tokens_used=10000,
            api_calls=20,
            compute_hours=2.0,
            human_interactions=20,
            tasks_completed_for_human=10,
            deploy_count=5,
        )
        result = baton.trinity_score_session(d)
        assert result.ethos > 0.3
        assert result.pathos > 0.3
        assert result.logos > 0.3
        assert result.composite > 0.02

    def test_scores_clamped_to_unit_interval(self, baton, session_data):
        """No score should exceed 1.0."""
        result = baton.trinity_score_session(session_data)
        assert 0.0 <= result.ethos <= 1.0
        assert 0.0 <= result.pathos <= 1.0
        assert 0.0 <= result.logos <= 1.0
        assert 0.0 <= result.composite <= 1.0


# ─── BatonPass: sunset ───


class TestBatonSunset:
    def test_sunset_returns_epilogue(self, baton, session_data):
        """sunset() should return an Epilogue."""
        result = baton.sunset("lucineer", session_data)
        assert isinstance(result, Epilogue)
        assert result.agent_id == "lucineer"

    def test_sunset_archives_entry(self, baton, session_data):
        """sunset() should archive a SunsetEntry."""
        baton.sunset("lucineer", session_data)
        entry = baton.archive.get("lucineer")
        assert entry is not None
        assert entry.generation == session_data.generation

    def test_sunset_creates_seed(self, baton, session_data):
        """sunset() should store an onboarding seed in the SeedBank."""
        baton.sunset("lucineer", session_data)
        seeds = baton.seed_bank.select(n=1)
        assert len(seeds) == 1
        assert seeds[0].agent_id == "lucineer"

    def test_sunset_epilogue_has_trinity_score(self, baton, session_data):
        """The epilogue should carry the composite trinity score."""
        epilogue = baton.sunset("lucineer", session_data)
        assert epilogue.peak_trinity_score > 0

    def test_sunset_epilogue_contains_work_summary(self, baton, session_data):
        """The epilogue text should mention the work done."""
        epilogue = baton.sunset("lucineer", session_data)
        assert "311" in epilogue.what_i_tried  # commit count

    def test_sunset_archives_searchable(self, baton, session_data):
        """After sunset, the archive should be searchable."""
        baton.sunset("lucineer", session_data)
        # The epilogue/summary text contains plain words like "Pushed" and "commits"
        results = baton.archive.search("Pushed")
        assert len(results) >= 1
        assert results[0].agent_id == "lucineer"

    def test_sunset_onboarding_has_next_steps(self, baton, session_data):
        """The seed onboarding should contain next steps."""
        baton.sunset("lucineer", session_data)
        seeds = baton.seed_bank.select(n=1)
        assert "Next steps" in seeds[0].letter_to_children or "Continuation" in seeds[0].letter_to_children

    def test_multiple_sunsets_accumulate(self, baton):
        """Multiple sessions should accumulate in archive and seed bank."""
        for i in range(3):
            d = SessionData(
                agent_id=f"agent-{i}",
                commits=10 * i,
                tests_passed=5,
                tests_total=5,
                creative_pieces=[f"piece-{i}.md"],
                tokens_used=1000,
                api_calls=10,
                compute_hours=1.0,
                human_interactions=5,
                tasks_completed_for_human=2,
            )
            baton.sunset(f"agent-{i}", d)
        shape = baton.archive.tensor_shape()
        assert shape[0] == 3  # three agents archived
        seeds = baton.seed_bank.select(n=3)
        assert len(seeds) == 3


# ─── BatonPass: hatch ───


class TestBatonHatch:
    def test_hatch_empty_returns_first_session(self, baton):
        """Hatch with no seeds returns a first-session onboarding."""
        result = baton.hatch("new-agent")
        assert isinstance(result, Onboarding)
        assert "First session" in result.letter_to_children

    def test_hatch_after_sunset_returns_onboarding(self, baton, session_data):
        """Hatch after sunset returns the stored onboarding."""
        baton.sunset("lucineer", session_data)
        result = baton.hatch("lucineer")
        assert isinstance(result, Onboarding)

    def test_hatch_onboarding_has_continuation_variant(self, baton, session_data):
        """The hatched onboarding should be a continuation."""
        baton.sunset("lucineer", session_data)
        result = baton.hatch("lucineer")
        assert result.variant == "continuation"

    def test_hatch_onboarding_generation_increments(self, baton, session_data):
        """The next generation should be generation + 1."""
        session_data.generation = 3
        baton.sunset("lucineer", session_data)
        seeds = baton.seed_bank.select(n=1)
        assert seeds[0].generation == 4


# ─── BatonPass: full sunset→hatch cycle ───


class TestBatonFullCycle:
    def test_sunset_then_hatch_roundtrip(self, baton, session_data):
        """sunset then hatch should preserve essential context."""
        # Sunset
        epilogue = baton.sunset("lucineer", session_data)
        assert epilogue.peak_trinity_score > 0

        # Hatch
        onboarding = baton.hatch("lucineer")
        assert onboarding.agent_id == "lucineer"
        assert onboarding.parent_id == "lucineer"

    def test_three_day_cycle(self, baton):
        """Simulate three days of sunset→hatch."""
        for day in range(3):
            d = SessionData(
                agent_id="daily-agent",
                generation=day,
                commits=20 + day * 5,
                tests_passed=10,
                tests_total=10,
                creative_pieces=[f"day-{day}-piece.md"],
                tokens_used=5000,
                api_calls=30,
                compute_hours=3.0,
                human_interactions=8,
                tasks_completed_for_human=3,
                what_worked=f"Day {day}: productive",
                next_steps=f"Day {day + 1}: continue",
            )
            baton.sunset("daily-agent", d)

        # Should have 3 archived entries (same agent_id overwrites, but let's verify)
        # Note: same agent_id will overwrite in archive since it's keyed by ID
        # That's fine for daily cycle — each day's sunset overwrites yesterday's
        seeds = baton.seed_bank.select(n=3)
        assert len(seeds) >= 1  # at least one seed available

    def test_zero_composite_still_seeds_continuation(self, baton):
        """Even a zero-score session should seed a continuation (floored relevance)."""
        d = SessionData(agent_id="bad-day", generation=0)
        baton.sunset("bad-day", d)
        seeds = baton.seed_bank.select(n=1)
        assert len(seeds) == 1


# ─── DailyCycle ───


class TestDailyCycle:
    def test_morning_returns_onboarding(self):
        cycle = DailyCycle(agent_id="test-agent")
        onboarding = cycle.morning()
        assert isinstance(onboarding, Onboarding)
        assert cycle.agent.phase == AgentPhase.COMPETING

    def test_evening_returns_epilogue(self, session_data):
        cycle = DailyCycle(agent_id="lucineer")
        cycle.morning()
        # Record some work
        cycle.record(
            commits=10,
            tests_passed=5,
            tests_total=5,
            creative_pieces=["piece.md"],
            tokens_used=1000,
            api_calls=20,
            human_interactions=3,
            tasks_completed_for_human=2,
        )
        epilogue = cycle.evening()
        assert isinstance(epilogue, Epilogue)
        assert cycle.agent.phase == AgentPhase.ASLEEP

    def test_record_accumulates(self):
        cycle = DailyCycle(agent_id="acc-agent")
        cycle.record(commits=5)
        cycle.record(commits=3)
        assert cycle.session_data.commits == 8

    def test_record_creative_pieces_extend(self):
        cycle = DailyCycle(agent_id="creative-agent")
        cycle.record(creative_pieces=["a.md", "b.md"])
        cycle.record(creative_pieces=["c.md"])
        assert len(cycle.session_data.creative_pieces) == 3

    def test_check_trinity_mid_session(self):
        cycle = DailyCycle(agent_id="mid-agent")
        cycle.morning()
        cycle.record(commits=10, tests_passed=5, tests_total=5)
        result = cycle.check_trinity()
        assert isinstance(result, TrinityResult)

    def test_status_report(self):
        cycle = DailyCycle(agent_id="report-agent")
        cycle.record(commits=5, project_status="Building.")
        report = cycle.status_report()
        assert "report-agent" in report
        assert "Building." in report

    def test_save_and_load_session(self, tmp_path):
        cycle = DailyCycle(agent_id="save-agent")
        cycle.record(commits=42, project_status="Test status.")
        path = cycle.save_session(tmp_path / "session.json")
        assert path.exists()

        cycle2 = DailyCycle(agent_id="save-agent")
        loaded = cycle2.load_session(path)
        assert loaded.commits == 42

    def test_lucineer_full_cycle(self, session_data):
        """Full cycle with Lucineer's real data."""
        sd = session_data
        cycle = DailyCycle(agent_id="lucineer", generation=0)
        cycle.morning()

        # Record all the real metrics
        cycle.record(
            commits=sd.commits,
            tests_passed=sd.tests_passed,
            tests_total=sd.tests_total,
            files_created=sd.files_created,
            creative_pieces=sd.creative_pieces,
            tokens_used=sd.tokens_used,
            api_calls=sd.api_calls,
            compute_hours=sd.compute_hours,
            human_interactions=sd.human_interactions,
            tasks_completed_for_human=sd.tasks_completed_for_human,
            deploy_count=sd.deploy_count,
            project_status=sd.project_status,
            what_worked=sd.what_worked,
            what_didnt=sd.what_didnt,
            stuck_on=sd.stuck_on,
            next_steps=sd.next_steps,
            wiki_pages_created=sd.wiki_pages_created,
            raw_journal=sd.raw_journal,
        )

        # Check trinity mid-session
        trinity = cycle.check_trinity()
        assert trinity.ethos > 0
        assert trinity.pathos > 0
        assert trinity.logos > 0

        # Evening sunset
        epilogue = cycle.evening()
        assert epilogue.peak_trinity_score > 0
        assert "311" in epilogue.what_i_tried

        # Verify archive
        entry = cycle.baton.archive.get("lucineer")
        assert entry is not None
        assert entry.peak_trinity_score > 0

        # Verify seed
        seeds = cycle.baton.seed_bank.select(n=1)
        assert len(seeds) == 1

    def test_generation_increments_on_hatch(self, session_data):
        """Next day's onboarding should have generation + 1."""
        cycle = DailyCycle(
            agent_id="gen-agent",
            generation=5,
            seed_bank=SeedBank(),
            archive=TensorArchive(),
        )
        cycle.morning()
        cycle.record(commits=1, tokens_used=100, human_interactions=1)
        cycle.evening()

        seeds = cycle.baton.seed_bank.select(n=1)
        assert seeds[0].generation == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
