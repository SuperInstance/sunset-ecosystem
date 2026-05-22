"""Tests for Intent Confirmation Protocol."""

import pytest

from logos.intent_protocol import (
    DESTRUCTIVE_ACTIONS,
    FleetState,
    Intent,
    IntentConfirmationProtocol,
)
from logos.decision_journal import DecisionJournal


@pytest.fixture
def protocol():
    state = FleetState(
        total_agents=1247,
        active_agents=900,
        rooms=["Tide-Pool", "Harbor", "Abyss"],
        avg_fitness=0.48,
        top_fitness_threshold=0.5,
    )
    return IntentConfirmationProtocol(fleet_state=state)


@pytest.fixture
def empty_protocol():
    state = FleetState(total_agents=0, active_agents=0, rooms=[])
    return IntentConfirmationProtocol(fleet_state=state)


class TestParseIntent:
    def test_make_it_faster(self, protocol):
        intent = protocol.parse_intent("make it faster")
        assert intent.action == "optimize"
        assert intent.target == "performance"
        assert intent.scope == "unspecified"
        assert intent.urgency == "normal"
        assert intent.raw_command == "make it faster"

    def test_sunset_agent_42(self, protocol):
        intent = protocol.parse_intent("sunset agent 42")
        assert intent.action == "sunset"
        assert intent.scope == "agent:42"
        assert intent.target == "agents"

    def test_breed_top_10(self, protocol):
        intent = protocol.parse_intent("breed top 10")
        assert intent.action == "breed"
        assert intent.scope == "top:10"
        assert intent.target == "agents"

    def test_migrate_room_harbor(self, protocol):
        intent = protocol.parse_intent("migrate agents to room Harbor")
        assert intent.action == "migrate"
        assert intent.scope == "room:Harbor"

    def test_critical_urgency(self, protocol):
        intent = protocol.parse_intent("optimize memory immediately")
        assert intent.urgency == "critical"
        assert intent.target == "memory"


class TestMeasureAmbiguity:
    def test_unspecified_scope_high_ambiguity(self, protocol):
        intent = protocol.parse_intent("make it faster")
        score = protocol.measure_ambiguity(intent)
        assert score > 0.7

    def test_all_scope_moderate_ambiguity(self, protocol):
        intent = protocol.parse_intent("optimize all agents")
        score = protocol.measure_ambiguity(intent)
        assert 0.3 <= score < 0.7

    def test_specific_agent_low_ambiguity(self, protocol):
        intent = protocol.parse_intent("sunset agent 42")
        score = protocol.measure_ambiguity(intent)
        assert score < 0.7

    def test_sunset_all_max_ambiguity(self, protocol):
        intent = protocol.parse_intent("sunset all agents")
        score = protocol.measure_ambiguity(intent)
        assert score == 1.0

    def test_top_10_medium_ambiguity(self, protocol):
        intent = protocol.parse_intent("breed top 10")
        score = protocol.measure_ambiguity(intent)
        # "top:10" is explicit but still affects multiple agents
        assert 0.3 <= score < 0.7


class TestRequireConfirmation:
    def test_make_it_faster_requires_confirmation(self, protocol):
        intent = protocol.parse_intent("make it faster")
        assert protocol.require_confirmation(intent) is True

    def test_sunset_agent_42_no_confirmation(self, protocol):
        intent = protocol.parse_intent("sunset agent 42")
        assert protocol.require_confirmation(intent) is False

    def test_breed_top_10_optional(self, protocol):
        intent = protocol.parse_intent("breed top 10")
        # medium ambiguity — below threshold, non-destructive
        assert protocol.require_confirmation(intent) is False

    def test_sunset_all_always_confirms(self, protocol):
        intent = protocol.parse_intent("sunset all")
        # destructive action: ALWAYS requires confirmation regardless of ambiguity
        assert protocol.require_confirmation(intent) is True

    def test_destructive_specific_scope_no_confirmation(self, protocol):
        # Single-agent sunset: low ambiguity + narrow scope → no confirmation
        intent = Intent(
            action="sunset",
            target="agents",
            scope="agent:42",
            urgency="normal",
            raw_command="sunset agent 42",
        )
        assert intent.is_destructive() is True
        assert protocol.require_confirmation(intent) is False

    def test_destructive_broad_scope_always_confirms(self, protocol):
        # Fleet-wide sunset: always confirms
        intent = Intent(
            action="sunset",
            target="agents",
            scope="all",
            urgency="normal",
            raw_command="sunset all",
        )
        assert intent.is_destructive() is True
        assert protocol.require_confirmation(intent) is True

    def test_destructive_action_registry(self):
        assert "sunset" in DESTRUCTIVE_ACTIONS
        assert "kill" in DESTRUCTIVE_ACTIONS
        assert "optimize" not in DESTRUCTIVE_ACTIONS


class TestGenerateConfirmation:
    def test_prompt_contains_fleet_stats(self, protocol):
        intent = protocol.parse_intent("make it faster")
        prompt = protocol.generate_confirmation(intent)
        assert "1,247 total agents" in prompt
        assert "900 active" in prompt

    def test_prompt_lists_options(self, protocol):
        intent = protocol.parse_intent("make it faster")
        prompt = protocol.generate_confirmation(intent)
        assert "(a) All agents globally" in prompt
        assert "(b)" in prompt
        assert "(c) Agents with fitness" in prompt
        assert "(d) Something else?" in prompt

    def test_destructive_warning(self, protocol):
        intent = protocol.parse_intent("sunset all")
        prompt = protocol.generate_confirmation(intent)
        assert "DESTRUCTIVE" in prompt

    def test_empty_fleet_graceful(self, empty_protocol):
        intent = empty_protocol.parse_intent("make it faster")
        prompt = empty_protocol.generate_confirmation(intent)
        assert "All agents globally" in prompt
        # Should not crash even with 0 agents


class TestLogDecision:
    def test_logs_to_journal(self, protocol):
        journal = DecisionJournal()
        intent = protocol.parse_intent("make it faster")
        protocol.log_decision(
            intent=intent,
            confirmed=False,
            scope="unspecified",
            journal=journal,
        )
        assert len(journal.all_entries()) == 1
        entry = journal.all_entries()[0]
        assert entry.why == "make it faster"
        assert entry.what == "optimize → unspecified"
        assert entry.scope == "unspecified"

    def test_logs_without_journal_is_noop(self, protocol):
        intent = protocol.parse_intent("make it faster")
        # Should not raise even when journal=None
        protocol.log_decision(intent=intent, confirmed=True, scope="all")


class TestFleetState:
    def test_agents_above_fitness_zero(self):
        state = FleetState(total_agents=0, avg_fitness=0.0)
        assert state.agents_above_fitness(0.5) == 0

    def test_agents_above_fitness_estimate(self):
        state = FleetState(total_agents=100, avg_fitness=0.6)
        # heuristic: 0.6 / 0.5 * 100 = 120, clamped to 100
        assert state.agents_above_fitness(0.5) == 100


class TestIntegrationPatterns:
    """Integration-style tests showing how the protocol is used in practice."""

    def test_full_flow_make_it_faster(self, protocol):
        """End-to-end: ambiguous command → confirmation required → generate prompt."""
        intent = protocol.parse_intent("make it faster")
        assert protocol.measure_ambiguity(intent) > protocol.AMBIGUITY_THRESHOLD
        assert protocol.require_confirmation(intent) is True
        prompt = protocol.generate_confirmation(intent)
        assert "Do you want to optimize" in prompt

    def test_full_flow_sunset_agent_42(self, protocol):
        """End-to-end: scoped destructive command → low ambiguity, narrow scope → no confirmation."""
        intent = protocol.parse_intent("sunset agent 42")
        # ambiguity is low, scope is a single agent → no confirmation needed
        assert protocol.measure_ambiguity(intent) < protocol.AMBIGUITY_THRESHOLD
        assert protocol.require_confirmation(intent) is False

    def test_full_flow_sunset_all(self, protocol):
        """Destructive + all scope → always confirms, max ambiguity."""
        intent = protocol.parse_intent("sunset all")
        assert protocol.measure_ambiguity(intent) == 1.0
        assert protocol.require_confirmation(intent) is True
        prompt = protocol.generate_confirmation(intent)
        assert "DESTRUCTIVE" in prompt
