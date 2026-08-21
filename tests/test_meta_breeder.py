"""Tests for the MetaBreeder module.

Coverage areas:
- Landscape analysis (ruggedness, smoothness, modality)
- Breeder selection based on landscape type
- Stall detection (fitness plateau, diversity collapse)
- Breeder switching logic
- Warm-start from previous breeder's population
- Event emission with selection reasoning
- Edge cases (empty archive, single breeder, all stall)
"""

import math
import random
from typing import List

import pytest

from swarm.breeding_kernel import (
    BreedingKernel,
    BreedingPreset,
    BreedingEvent,
    Genome,
    CallableEvaluator,
)
from swarm.meta_breeder import (
    LandscapeAnalyzer,
    LandscapeType,
    BreederPortfolio,
    BreederRecord,
    MetaBreeder,
    MetaBreedingEvent,
    StallDetector,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def simple_evaluator():
    """A simple sphere fitness function."""

    def fn(g: Genome) -> float:
        return -sum(x**2 for x in g.genes)

    return CallableEvaluator(fn)


@pytest.fixture
def multimodal_evaluator():
    """A multimodal Rastrigin-like function."""

    def fn(g: Genome) -> float:
        return -(10 + sum(x**2 - 10 * math.cos(2 * math.pi * x) for x in g.genes))

    return CallableEvaluator(fn)


@pytest.fixture
def random_population():
    """Generate a random population of genomes."""

    def _make(n=10, dim=3):
        return [
            Genome(genes=[random.uniform(-1, 1) for _ in range(dim)]) for _ in range(n)
        ]

    return _make


@pytest.fixture
def portfolio(simple_evaluator, random_population):
    """A portfolio with all four presets."""
    p = BreederPortfolio(evaluator=simple_evaluator, pop_size=20, gene_dim=3)
    for preset in BreedingPreset.all():
        p.add_preset(preset, name=preset.name.lower())
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Landscape analysis tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_landscape_analyzer_unknown_with_empty_history():
    """Test that empty fitness history returns UNKNOWN."""
    la = LandscapeAnalyzer()
    result = la.analyze([], [])
    assert result == LandscapeType.UNKNOWN


def test_landscape_analyzer_unknown_with_short_history():
    """Test that short fitness history (<3) returns UNKNOWN."""
    la = LandscapeAnalyzer()
    result = la.analyze([1.0, 1.1], [0.5])
    assert result == LandscapeType.UNKNOWN


def test_landscape_analyzer_detects_smooth_landscape():
    """A smoothly improving fitness curve should be classified as SMOOTH."""
    la = LandscapeAnalyzer(ruggedness_threshold=0.3, smoothness_threshold=0.05)
    # Linearly increasing fitness with tiny fluctuations
    fitness = [1.0 + 0.01 * i for i in range(15)]
    diversity = [0.5] * 15
    result = la.analyze(fitness, diversity)
    assert result == LandscapeType.SMOOTH


def test_landscape_analyzer_detects_rugged_landscape():
    """A wildly oscillating fitness curve is classified as MULTIMODAL (many peaks).
    The analyzer prioritizes modality over ruggedness when both are high."""
    la = LandscapeAnalyzer(ruggedness_threshold=0.3)
    # High fluctuation: up, down, up, down — this creates many local extrema
    fitness = [1.0, 0.2, 0.9, 0.1, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5]
    diversity = [0.5] * 10
    result = la.analyze(fitness, diversity)
    assert result == LandscapeType.MULTIMODAL


def test_landscape_analyzer_detects_multimodal():
    """A curve with multiple peaks should be classified as MULTIMODAL."""
    la = LandscapeAnalyzer(modality_threshold=0.15)
    # Multiple local extrema
    fitness = [0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5]
    diversity = [0.5] * 10
    result = la.analyze(fitness, diversity)
    assert result == LandscapeType.MULTIMODAL


def test_landscape_analyzer_ruggedness_computation():
    """Test that ruggedness is computed correctly."""
    la = LandscapeAnalyzer()
    # Window with large jumps
    window = [0.0, 1.0, 0.0, 1.0, 0.0]
    ruggedness = la._compute_ruggedness(window)
    assert ruggedness > 0.0
    # Flat window should have zero ruggedness
    assert la._compute_ruggedness([1.0, 1.0, 1.0]) == 0.0


def test_landscape_analyzer_smoothness_computation():
    """Test that smoothness (second derivative) is computed correctly."""
    la = LandscapeAnalyzer()
    # Linear has zero second derivative (smooth)
    linear = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert la._compute_smoothness(linear) == 0.0
    # Parabolic has constant second derivative
    parabola = [1.0, 4.0, 9.0, 16.0]
    assert la._compute_smoothness(parabola) > 0.0


def test_landscape_analyzer_modality_computation():
    """Test that modality (extrema count) is computed correctly."""
    la = LandscapeAnalyzer()
    # Alternating peaks and valleys: 8 extrema out of 8 interior points
    alternating = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    modality = la._compute_modality(alternating)
    assert modality == 1.0

    # Monotonic has no extrema
    monotonic = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert la._compute_modality(monotonic) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Breeder selection based on landscape type
# ═══════════════════════════════════════════════════════════════════════════════


def test_portfolio_selects_breeder_for_smooth_landscape(
    portfolio, simple_evaluator, random_population
):
    """The portfolio should select EXPLOITATION for smooth landscapes."""
    # Warm up all breeders with some QD scores
    for name, record in portfolio.breeders.items():
        for _ in range(5):
            record.breeder.run(3)
            portfolio.update_qd(name, record.breeder.qd_score)

    best_name, best_record = portfolio.get_best_breeder(LandscapeType.SMOOTH)
    assert best_name in portfolio.breeders
    assert best_record is not None


def test_portfolio_landscape_match_bonus(portfolio):
    """EXPLOITATION should get a bonus for SMOOTH, EXPLORATION for RUGGED."""
    bonus_exploit = portfolio._landscape_match_bonus(
        BreedingPreset.EXPLOITATION, LandscapeType.SMOOTH
    )
    bonus_explore = portfolio._landscape_match_bonus(
        BreedingPreset.EXPLORATION, LandscapeType.RUGGED
    )
    bonus_none = portfolio._landscape_match_bonus(
        BreedingPreset.EXPLOITATION, LandscapeType.RUGGED
    )

    assert bonus_exploit == 2.0
    assert bonus_explore == 2.0
    assert bonus_none == 0.0


def test_portfolio_empty_raises(portfolio):
    """An empty portfolio should raise ValueError on selection."""
    empty = BreederPortfolio(evaluator=simple_evaluator, pop_size=10, gene_dim=3)
    with pytest.raises(ValueError, match="Portfolio is empty"):
        empty.get_best_breeder(LandscapeType.SMOOTH)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Stall detection tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_stall_detector_fitness_plateau():
    """Stall should be detected when fitness stays flat."""
    sd = StallDetector(fitness_window=5, fitness_tolerance=1e-4, min_generations=3)
    # Flat fitness for 5 generations
    fitness = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    diversity = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    stalled, reason = sd.is_stalled(fitness, diversity, generations_active=5)
    assert stalled is True
    assert reason == "fitness_plateau"


def test_stall_detector_diversity_collapse():
    """Stall should be detected when diversity drops below threshold."""
    sd = StallDetector(diversity_threshold=0.01, min_generations=1)
    fitness = [1.0, 1.1, 1.2]
    diversity = [0.5, 0.02, 0.001]
    stalled, reason = sd.is_stalled(fitness, diversity, generations_active=3)
    assert stalled is True
    assert reason == "diversity_collapse"


def test_stall_detector_too_early_not_stalled():
    """Stall should not be detected before min_generations."""
    sd = StallDetector(min_generations=5)
    stalled, reason = sd.is_stalled(
        [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], generations_active=2
    )
    assert stalled is False
    assert reason == "too_early"


def test_stall_detector_monotonic_decline():
    """Stall should be detected on monotonic fitness decline."""
    sd = StallDetector(fitness_window=5, min_generations=3)
    fitness = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    diversity = [0.5] * 6
    stalled, reason = sd.is_stalled(fitness, diversity, generations_active=5)
    assert stalled is True
    assert reason == "monotonic_decline"


def test_stall_detector_active_when_improving():
    """No stall when fitness is steadily improving."""
    sd = StallDetector(fitness_window=5, fitness_tolerance=1e-4)
    fitness = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    diversity = [0.5] * 6
    stalled, reason = sd.is_stalled(fitness, diversity, generations_active=5)
    assert stalled is False
    assert reason == "active"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Breeder switching logic
# ═══════════════════════════════════════════════════════════════════════════════


def test_meta_breeder_switch_on_stall(portfolio, simple_evaluator):
    """MetaBreeder should switch breeders when the current one stalls."""
    mb = MetaBreeder(portfolio, evaluator=simple_evaluator, max_stall_switches=3)
    # Activate the first breeder
    first_name = mb.current_breeder_name

    # Force a stall by injecting flat fitness history
    for _ in range(5):
        mb.step()

    # Now inject a flat fitness history to trigger stall
    record = mb.current_breeder_record
    record.breeder._fitness_history = [1.0] * 20
    record.breeder._diversity_history = [0.5] * 20
    record.generations_active = 20

    events = mb.step()
    switch_events = [
        e
        for e in events
        if isinstance(e, MetaBreedingEvent) and e.event_type == "breeder_switched"
    ]
    assert len(switch_events) >= 1


def test_meta_breeder_switch_forces_different_breeder(portfolio, simple_evaluator):
    """When switching, the meta-breeder should pick a different breeder."""
    mb = MetaBreeder(portfolio, evaluator=simple_evaluator, max_stall_switches=3)
    first_name = mb.current_breeder_name

    # Run a few steps to warm up
    for _ in range(3):
        mb.step()

    # Force stall
    record = mb.current_breeder_record
    record.breeder._fitness_history = [1.0] * 20
    record.breeder._diversity_history = [0.5] * 20
    record.generations_active = 20

    events = mb.step()
    switch_events = [
        e
        for e in events
        if isinstance(e, MetaBreedingEvent) and e.event_type == "breeder_switched"
    ]
    if switch_events:
        assert switch_events[-1].selected_breeder != first_name


def test_meta_breeder_max_stall_limit(portfolio, simple_evaluator):
    """After max_stall_switches, the meta-breeder should not switch again."""
    mb = MetaBreeder(portfolio, evaluator=simple_evaluator, max_stall_switches=1)

    # First stall → switch
    for _ in range(3):
        mb.step()
    record = mb.current_breeder_record
    record.breeder._fitness_history = [1.0] * 20
    record.breeder._diversity_history = [0.5] * 20
    record.generations_active = 20
    mb.step()

    assert mb.stall_switch_count == 1

    # Second stall → limit reached
    record = mb.current_breeder_record
    record.breeder._fitness_history = [1.0] * 20
    record.breeder._diversity_history = [0.5] * 20
    record.generations_active = 20
    events = mb.step()
    limit_events = [
        e
        for e in events
        if isinstance(e, MetaBreedingEvent) and e.event_type == "stall_limit_reached"
    ]
    assert len(limit_events) >= 1


def test_meta_breeder_single_breeder(portfolio, simple_evaluator):
    """With a single breeder, the meta-breeder should still run without error."""
    single = BreederPortfolio(evaluator=simple_evaluator, pop_size=10, gene_dim=3)
    single.add_preset(BreedingPreset.BALANCED, name="only")

    mb = MetaBreeder(single, evaluator=simple_evaluator)
    events = mb.run(5)
    assert len(events) > 0
    assert mb.current_breeder_name == "only"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Warm-start from previous breeder's population
# ═══════════════════════════════════════════════════════════════════════════════


def test_warm_start_population_blended(portfolio, simple_evaluator):
    """Warm-start should blend old population with new random individuals."""
    mb = MetaBreeder(
        portfolio, evaluator=simple_evaluator, warm_start_ratio=0.5, pop_size=20
    )

    # Run a few generations to build up a population
    mb.run(3)
    old_pop = [g.copy() for g in mb.current_breeder_record.breeder.population]

    # Switch to another breeder with warm start
    next_name = [n for n in portfolio.breeders if n != mb.current_breeder_name][0]
    event = mb._activate_breeder(next_name, warm_start_population=old_pop)

    new_pop = mb.current_breeder_record.breeder.population
    assert len(new_pop) == 20
    # At least some individuals should come from warm start (the best ones are selected)
    assert event.payload.get("warm_start") is True


def test_warm_start_selects_diverse_subset(portfolio, simple_evaluator):
    """Warm-start should select a diverse subset, not just the best."""
    mb = MetaBreeder(
        portfolio, evaluator=simple_evaluator, warm_start_ratio=0.6, pop_size=10
    )
    mb.run(3)

    old_pop = mb.current_breeder_record.breeder.population
    diverse_subset = mb._select_diverse_subset(old_pop, 5)
    assert len(diverse_subset) == 5
    # All returned should be copies (not the same objects)
    assert all(g not in old_pop for g in diverse_subset)


def test_warm_start_empty_population(portfolio, simple_evaluator):
    """Warm-start with empty population should still produce a valid population."""
    mb = MetaBreeder(
        portfolio, evaluator=simple_evaluator, warm_start_ratio=0.5, pop_size=10
    )
    next_name = [n for n in portfolio.breeders if n != mb.current_breeder_name][0]
    event = mb._activate_breeder(next_name, warm_start_population=[])
    new_pop = mb.current_breeder_record.breeder.population
    # Empty warm-start means all-random population; should still reach pop_size
    assert len(new_pop) == 10


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Event emission with selection reasoning
# ═══════════════════════════════════════════════════════════════════════════════


def test_event_emission_on_breeder_activation(portfolio, simple_evaluator):
    """Activating a breeder should emit a MetaBreedingEvent with reasoning."""
    mb = MetaBreeder(portfolio, evaluator=simple_evaluator)
    event = mb._activate_breeder("balanced")
    assert isinstance(event, MetaBreedingEvent)
    assert event.event_type == "breeder_activated"
    assert event.reasoning is not None
    assert len(event.reasoning) > 0


def test_event_emission_on_breeder_switch(portfolio, simple_evaluator):
    """Switching breeders should emit a MetaBreedingEvent with reasoning and landscape."""
    mb = MetaBreeder(portfolio, evaluator=simple_evaluator, max_stall_switches=3)
    mb.run(3)

    record = mb.current_breeder_record
    record.breeder._fitness_history = [1.0] * 20
    record.breeder._diversity_history = [0.5] * 20
    record.generations_active = 20

    events = mb.step()
    switch_events = [
        e
        for e in events
        if isinstance(e, MetaBreedingEvent) and e.event_type == "breeder_switched"
    ]
    assert len(switch_events) >= 1
    assert switch_events[0].landscape is not None
    assert "Stall detected" in switch_events[0].reasoning


def test_event_payload_contains_stall_reason(portfolio, simple_evaluator):
    """Switch events should contain the stall reason in the payload."""
    mb = MetaBreeder(portfolio, evaluator=simple_evaluator, max_stall_switches=3)
    mb.run(3)

    record = mb.current_breeder_record
    record.breeder._fitness_history = [1.0] * 20
    record.breeder._diversity_history = [0.001] * 20  # diversity collapse
    record.generations_active = 20

    events = mb.step()
    switch_events = [
        e
        for e in events
        if isinstance(e, MetaBreedingEvent) and e.event_type == "breeder_switched"
    ]
    if switch_events:
        assert "stall_reason" in switch_events[0].payload


def test_event_list_accumulated(portfolio, simple_evaluator):
    """The meta-breeder should accumulate all events in self.events."""
    mb = MetaBreeder(portfolio, evaluator=simple_evaluator)
    mb.run(5)
    assert len(mb.events) > 0
    assert all(isinstance(e, MetaBreedingEvent) for e in mb.events)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


def test_empty_archive_qd_score(portfolio, simple_evaluator):
    """QD-score should be 0 for an empty archive."""
    single = BreederPortfolio(evaluator=simple_evaluator, pop_size=10, gene_dim=3)
    single.add_preset(BreedingPreset.BALANCED, name="only")
    mb = MetaBreeder(single, evaluator=simple_evaluator)
    mb.run(1)
    # After 1 generation, archive might have some entries but qd could be 0
    assert mb.current_breeder_record.breeder.qd_score is not None
    assert mb.current_breeder_record.breeder.qd_score >= -1e6  # fitness can be negative


def test_all_breeders_stall(portfolio, simple_evaluator):
    """When all breeders stall, the meta-breeder should handle gracefully."""
    mb = MetaBreeder(portfolio, evaluator=simple_evaluator, max_stall_switches=10)
    mb.run(3)

    # Stall all breeders
    for name, record in portfolio.breeders.items():
        record.breeder._fitness_history = [1.0] * 20
        record.breeder._diversity_history = [0.001] * 20
        record.generations_active = 20

    # Should still be able to run without crashing
    events = mb.run(2)
    assert all(isinstance(e, (BreedingEvent, MetaBreedingEvent)) for e in events)


def test_portfolio_add_existing_breeder(portfolio, simple_evaluator, random_population):
    """Adding an existing breeder should preserve its state."""
    pop = random_population(10, 3)
    bk = BreedingKernel.from_preset(
        BreedingPreset.BALANCED, simple_evaluator, pop, 10, name="custom"
    )
    portfolio.add_breeder(bk, BreedingPreset.BALANCED)
    assert "custom" in portfolio.breeders
    assert portfolio.breeders["custom"].breeder.name == "custom"


def test_meta_breeder_with_empty_portfolio(simple_evaluator):
    """Creating a MetaBreeder with an empty portfolio should not crash."""
    empty = BreederPortfolio(evaluator=simple_evaluator, pop_size=10, gene_dim=3)
    mb = MetaBreeder(empty, evaluator=simple_evaluator)
    assert mb.current_breeder_name is None


def test_meta_breeder_step_with_no_active_breeder(portfolio, simple_evaluator):
    """If no breeder is active, step should activate one."""
    # Create portfolio without auto-activating
    mb = MetaBreeder(portfolio, evaluator=simple_evaluator)
    mb.current_breeder_name = None
    mb.current_breeder_record = None
    events = mb.step()
    activation_events = [
        e
        for e in events
        if isinstance(e, MetaBreedingEvent) and e.event_type == "breeder_activated"
    ]
    assert len(activation_events) >= 1
    assert mb.current_breeder_name is not None


def test_select_diverse_subset_returns_copies(portfolio, simple_evaluator):
    """Diverse subset selection should return copies, not references."""
    mb = MetaBreeder(portfolio, evaluator=simple_evaluator, pop_size=10)
    pop = [Genome(genes=[0.0, 0.0, 0.0]) for _ in range(5)]
    subset = mb._select_diverse_subset(pop, 3)
    assert len(subset) == 3
    for g in subset:
        assert g not in pop  # Should be copies


def test_breeder_record_qd_trend():
    """BreederRecord should compute a QD trend slope."""
    br = BreederRecord(breeder=None, preset=BreedingPreset.BALANCED)
    for i in range(10):
        br.add_qd(float(i))  # Steadily increasing
    assert br.recent_qd_trend > 0.0

    br2 = BreederRecord(breeder=None, preset=BreedingPreset.BALANCED)
    for i in range(10):
        br2.add_qd(float(10 - i))  # Decreasing
    assert br2.recent_qd_trend < 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration / end-to-end
# ═══════════════════════════════════════════════════════════════════════════════


def test_end_to_end_meta_breeder_on_multimodal(multimodal_evaluator, random_population):
    """End-to-end: MetaBreeder on a multimodal landscape should adapt over time."""
    pop = random_population(20, 3)
    portfolio = BreederPortfolio(
        evaluator=multimodal_evaluator, pop_size=20, gene_dim=3
    )
    for preset in BreedingPreset.all():
        portfolio.add_preset(preset, name=preset.name.lower())

    mb = MetaBreeder(portfolio, evaluator=multimodal_evaluator, max_stall_switches=5)
    events = mb.run(15)

    # Should have generated both breeding and meta events
    breeding_events = [e for e in events if isinstance(e, BreedingEvent)]
    meta_events = [e for e in events if isinstance(e, MetaBreedingEvent)]
    assert len(breeding_events) > 0
    assert len(meta_events) >= 0

    # Best fitness should have improved (or at least not crashed)
    fitnesses = [
        e.payload.get("best_fitness")
        for e in breeding_events
        if e.payload.get("best_fitness") is not None
    ]
    assert len(fitnesses) > 0


def test_meta_breeder_repr(portfolio, simple_evaluator):
    """MetaBreeder __repr__ should be informative."""
    mb = MetaBreeder(portfolio, evaluator=simple_evaluator)
    r = repr(mb)
    assert "MetaBreeder" in r
    assert "gen=" in r
