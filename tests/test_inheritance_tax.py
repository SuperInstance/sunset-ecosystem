"""Tests for inheritance tax mechanism.

Verifies the five core claims from docs/RESEARCH_ECONOMICS.md:
1. Low inheritance  (0%)  → 0% tax
2. Medium inheritance (45%) → 15% tax
3. High inheritance (95%) → 50% tax
4. Dynasty score: earned vs inherited slots
5. Tax revenue funds global pool (social welfare)
"""

from __future__ import annotations

import pytest

from swarm.inheritance_tax import Agent, InheritanceTax


class TestTaxBrackets:
    """Tax brackets from RESEARCH_ECONOMICS.md."""

    def test_low_inheritance_zero_tax(self):
        """Low inheritance (0%) → 0% tax."""
        tax = InheritanceTax()
        rate = tax.compute_tax(
            parent_slots=100,
            child_slots=0,
            parent_fitness=0.5,
            child_fitness=0.5,
        )
        assert rate == 0.0

    def test_medium_inheritance_15_percent(self):
        """Medium inheritance (45%) → 15% tax."""
        tax = InheritanceTax()
        rate = tax.compute_tax(
            parent_slots=100,
            child_slots=45,
            parent_fitness=0.5,
            child_fitness=0.5,
        )
        # 45% in [0.3, 0.6] bracket → 15% base, no fitness delta
        assert rate == pytest.approx(0.15, abs=0.001)

    def test_high_inheritance_50_percent(self):
        """High inheritance (95%) → 50% tax."""
        tax = InheritanceTax()
        rate = tax.compute_tax(
            parent_slots=100,
            child_slots=95,
            parent_fitness=0.5,
            child_fitness=0.5,
        )
        # 95% in [0.9, 1.0] bracket → 50% base, no fitness delta
        assert rate == pytest.approx(0.50, abs=0.001)


class TestDynastyScore:
    """Dynastic lineage measurement."""

    def test_earned_slots_zero_dynasty(self):
        """Agent with no parent → all earned → dynasty score 0."""
        population = [
            Agent(agent_id=1, slots=10, fitness=0.5, parent_id=None),
        ]
        tax = InheritanceTax()
        score = tax.dynasty_score(1, population)
        assert score == 0.0

    def test_inherited_slots_high_dynasty(self):
        """Agent with parent → some inherited → dynasty score > 0."""
        population = [
            Agent(agent_id=1, slots=10, fitness=0.8, parent_id=None),
            Agent(agent_id=2, slots=8, fitness=0.5, parent_id=1),
        ]
        tax = InheritanceTax()
        score = tax.dynasty_score(2, population)
        # inherited = min(8, 10) = 8, earned = 0, total = 8
        assert score == 1.0

    def test_partial_inheritance(self):
        """Agent with more slots than parent → partially inherited."""
        population = [
            Agent(agent_id=1, slots=5, fitness=0.8, parent_id=None),
            Agent(agent_id=2, slots=10, fitness=0.5, parent_id=1),
        ]
        tax = InheritanceTax()
        score = tax.dynasty_score(2, population)
        # inherited = min(10, 5) = 5, earned = 5, total = 10
        assert score == pytest.approx(0.5, abs=0.001)


class TestGlobalPool:
    """Tax revenue funds global pool for social welfare."""

    def test_tax_revenue_funds_global_pool(self):
        """Tax revenue accumulates in global pool and funds new agents."""
        tax = InheritanceTax()

        # High inheritance triggers 50% tax
        parent_after, child_after = tax.apply_tax(
            parent_slots=100,
            child_slots=95,
            parent_fitness=0.5,
            child_fitness=0.5,
        )

        # 50% of 100 = 50 in global pool
        assert tax.global_pool == 50
        assert parent_after == 50
        assert child_after == 95

        # New agent requests 30 slots from pool
        granted = tax.fund_new_agent(30)
        assert granted == 30
        assert tax.global_pool == 20

        # Another request for 30 — only 20 left
        granted2 = tax.fund_new_agent(30)
        assert granted2 == 20
        assert tax.global_pool == 0

    def test_new_agent_gets_slots_from_tax(self):
        """A newly created agent receives slots funded by prior tax revenue."""
        tax = InheritanceTax()

        # Simulate a taxed breeding event
        tax.apply_tax(
            parent_slots=100,
            child_slots=95,
            parent_fitness=0.5,
            child_fitness=0.5,
        )
        assert tax.global_pool == 50

        # New unrelated agent requests startup slots
        startup = tax.fund_new_agent(InheritanceTax.DEFAULT_SLOTS)
        assert startup == 10  # full DEFAULT_SLOTS granted
        assert tax.global_pool == 40


class TestFitnessMultiplier:
    """Rich parents pay progressively more."""

    def test_rich_parent_taxed_more(self):
        """Higher fitness delta increases tax rate."""
        tax = InheritanceTax()

        # Same inheritance, equal fitness
        rate_equal = tax.compute_tax(
            parent_slots=100,
            child_slots=95,
            parent_fitness=0.5,
            child_fitness=0.5,
        )

        # Same inheritance, rich parent / poor child
        rate_rich = tax.compute_tax(
            parent_slots=100,
            child_slots=95,
            parent_fitness=1.0,
            child_fitness=0.0,
        )

        # Rich parent should pay more (multiplier = 1 + 1.0 = 2.0)
        assert rate_rich == pytest.approx(rate_equal * 2.0, abs=0.001)
        assert rate_rich > rate_equal

    def test_fitness_multiplier_capped_at_100_percent(self):
        """Tax rate never exceeds 100%."""
        tax = InheritanceTax()
        rate = tax.compute_tax(
            parent_slots=100,
            child_slots=95,
            parent_fitness=1.0,
            child_fitness=0.0,
        )
        assert rate <= 1.0
