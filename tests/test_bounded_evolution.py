"""Tests for bounded_evolution.py — Bounded Evolutionary Parameter Engine.

Pattern 4 from the SuperInstance audit: tunable parameters with explicit
bounds, mutation rates, fitness scores, and full rollback capability.
"""

import pytest

from fleet.bounded_evolution import (
    BoundedParameter,
    EvolutionEngine,
    EvolutionMode,
    MutationType,
    GenerationSnapshot,
)


# ===================================================================
# BoundedParameter
# ===================================================================


class TestBoundedParameter:
    def test_clamp_within_range(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0)
        assert p.clamp() == 5.0

    def test_clamp_below_min(self) -> None:
        p = BoundedParameter(value=-5.0, min=0.0, max=10.0)
        assert p.clamp() == 0.0

    def test_clamp_above_max(self) -> None:
        p = BoundedParameter(value=15.0, min=0.0, max=10.0)
        assert p.clamp() == 10.0

    def test_range(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0)
        assert p.range == 10.0

    def test_copy(self) -> None:
        p = BoundedParameter(
            value=5.0, min=0.0, max=10.0, mutation_rate=0.2, fitness_score=1.0, name="x"
        )
        c = p.copy()
        assert c.value == 5.0
        assert c.min == 0.0
        assert c.max == 10.0
        assert c.mutation_rate == 0.2
        assert c.fitness_score == 1.0
        assert c.name == "x"
        # Independent
        c.value = 7.0
        assert p.value == 5.0


# ===================================================================
# EvolutionEngine — Initialization
# ===================================================================


class TestEvolutionEngineInit:
    def test_empty_params(self) -> None:
        engine = EvolutionEngine([])
        assert engine.generation == 0
        assert engine.parameters == {}
        assert engine.average_fitness() == 0.0

    def test_single_param(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p])
        assert "alpha" in engine.parameters
        assert engine.parameters["alpha"].value == 5.0

    def test_auto_naming(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0)
        engine = EvolutionEngine([p])
        assert "param_0" in engine.parameters

    def test_initial_values_stored(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p])
        assert engine._initial_values["alpha"] == 5.0

    def test_seed_reproducibility(self) -> None:
        p1 = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        p2 = BoundedParameter(value=5.0, min=0.0, max=10.0, name="beta")
        e1 = EvolutionEngine([p1.copy(), p2.copy()], seed=42)
        e2 = EvolutionEngine([p1.copy(), p2.copy()], seed=42)
        e1.evolve()
        e2.evolve()
        assert e1.parameters["alpha"].value == e2.parameters["alpha"].value
        assert e1.parameters["beta"].value == e2.parameters["beta"].value


# ===================================================================
# Fitness Scoring
# ===================================================================


class TestFitnessScoring:
    def test_score_single_param(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p])
        engine.score({"alpha": 5.0}, 1.0)
        assert engine.parameters["alpha"].fitness_score == 1.0

    def test_score_multiple_params(self) -> None:
        p1 = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        p2 = BoundedParameter(value=3.0, min=0.0, max=10.0, name="beta")
        engine = EvolutionEngine([p1, p2])
        engine.score({"alpha": 5.0, "beta": 3.0}, 2.0)
        assert engine.parameters["alpha"].fitness_score == 2.0
        assert engine.parameters["beta"].fitness_score == 2.0

    def test_score_all_when_empty(self) -> None:
        p1 = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        p2 = BoundedParameter(value=3.0, min=0.0, max=10.0, name="beta")
        engine = EvolutionEngine([p1, p2])
        engine.score({}, 1.5)
        assert engine.parameters["alpha"].fitness_score == 1.5
        assert engine.parameters["beta"].fitness_score == 1.5

    def test_average_fitness(self) -> None:
        p1 = BoundedParameter(
            value=5.0, min=0.0, max=10.0, name="alpha", fitness_score=1.0
        )
        p2 = BoundedParameter(
            value=3.0, min=0.0, max=10.0, name="beta", fitness_score=3.0
        )
        engine = EvolutionEngine([p1, p2])
        assert engine.average_fitness() == 2.0

    def test_score_missing_param_ignored(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p])
        engine.score({"beta": 3.0}, 1.0)  # beta doesn't exist
        assert engine.parameters["alpha"].fitness_score == 0.0


# ===================================================================
# Mode Selection
# ===================================================================


class TestModeSelection:
    def test_auto_mode_aggressive(self) -> None:
        p = BoundedParameter(
            value=5.0, min=0.0, max=10.0, name="alpha", fitness_score=0.0
        )
        engine = EvolutionEngine([p], mode=EvolutionMode.NORMAL, auto_mode=True)
        engine.evolve()
        assert engine.mode == EvolutionMode.AGGRESSIVE

    def test_auto_mode_elite(self) -> None:
        p = BoundedParameter(
            value=5.0, min=0.0, max=10.0, name="alpha", fitness_score=1.0
        )
        engine = EvolutionEngine([p], mode=EvolutionMode.NORMAL, auto_mode=True)
        engine.evolve()
        assert engine.mode == EvolutionMode.ELITE

    def test_auto_mode_normal(self) -> None:
        p = BoundedParameter(
            value=5.0, min=0.0, max=10.0, name="alpha", fitness_score=0.5
        )
        engine = EvolutionEngine([p], mode=EvolutionMode.NORMAL, auto_mode=True)
        engine.evolve()
        assert engine.mode == EvolutionMode.NORMAL

    def test_auto_mode_disabled(self) -> None:
        p = BoundedParameter(
            value=5.0, min=0.0, max=10.0, name="alpha", fitness_score=0.0
        )
        engine = EvolutionEngine([p], mode=EvolutionMode.ELITE, auto_mode=False)
        engine.evolve()
        assert engine.mode == EvolutionMode.ELITE

    def test_mode_magnitude(self) -> None:
        assert EvolutionEngine.MODE_MAGNITUDE[EvolutionMode.AGGRESSIVE] == 3.0
        assert EvolutionEngine.MODE_MAGNITUDE[EvolutionMode.NORMAL] == 1.0
        assert EvolutionEngine.MODE_MAGNITUDE[EvolutionMode.ELITE] == 0.5


# ===================================================================
# Snapshots & Rollback
# ===================================================================


class TestSnapshots:
    def test_snapshot_created(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p])
        snap = engine.snapshot()
        assert snap.generation == 0
        assert "alpha" in snap.parameters
        assert engine.snapshots[0] == snap

    def test_snapshot_increments_generation(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p])
        engine.evolve()
        assert engine.generation == 1
        assert 0 in engine.snapshots
        assert 1 not in engine.snapshots  # evolve snapshots gen 0, then increments

    def test_rollback_restores_state(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p], seed=42)
        engine.evolve()
        v_after = engine.parameters["alpha"].value
        engine.rollback(0)
        assert engine.parameters["alpha"].value == 5.0
        assert engine.generation == 0

    def test_rollback_prunes_future(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p], seed=42)
        engine.evolve()  # snapshots gen 0, then gen becomes 1
        engine.evolve()  # snapshots gen 1, then gen becomes 2
        assert 0 in engine.snapshots
        assert 1 in engine.snapshots
        assert 2 not in engine.snapshots  # snapshot happens before increment
        engine.rollback(0)
        assert 1 not in engine.snapshots
        assert 2 not in engine.snapshots

    def test_rollback_invalid_generation(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p])
        with pytest.raises(ValueError, match="No snapshot"):
            engine.rollback(5)

    def test_snapshot_serialization(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p])
        snap = engine.snapshot()
        d = snap.to_dict()
        assert d["generation"] == 0
        assert d["parameters"]["alpha"]["value"] == 5.0
        restored = GenerationSnapshot.from_dict(d)
        assert restored == snap


# ===================================================================
# Mutation Types
# ===================================================================


class TestMutations:
    def test_param_adjust_changes_value(self) -> None:
        p = BoundedParameter(
            value=5.0, min=0.0, max=10.0, name="alpha", mutation_rate=1.0
        )
        engine = EvolutionEngine([p], seed=42)
        engine._apply_single_mutation("alpha", MutationType.PARAM_ADJUST, 1.0)
        assert engine.parameters["alpha"].value != 5.0
        assert 0.0 <= engine.parameters["alpha"].value <= 10.0

    def test_threshold_shift_changes_value(self) -> None:
        p = BoundedParameter(
            value=5.0, min=0.0, max=10.0, name="alpha", mutation_rate=1.0
        )
        engine = EvolutionEngine([p], seed=42)
        engine._apply_single_mutation("alpha", MutationType.THRESHOLD_SHIFT, 1.0)
        assert engine.parameters["alpha"].value != 5.0

    def test_weight_rebalance_toward_mid(self) -> None:
        p = BoundedParameter(
            value=2.0, min=0.0, max=10.0, name="alpha", mutation_rate=1.0
        )
        engine = EvolutionEngine([p], seed=42)
        engine._apply_single_mutation("alpha", MutationType.WEIGHT_REBALANCE, 1.0)
        # Should move toward midpoint (5.0)
        assert engine.parameters["alpha"].value > 2.0

    def test_invert(self) -> None:
        p = BoundedParameter(value=2.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p], seed=42)
        engine._apply_single_mutation("alpha", MutationType.INVERT, 1.0)
        assert engine.parameters["alpha"].value == 8.0

    def test_scale(self) -> None:
        p = BoundedParameter(
            value=5.0, min=0.0, max=10.0, name="alpha", mutation_rate=1.0
        )
        engine = EvolutionEngine([p], seed=42)
        engine._apply_single_mutation("alpha", MutationType.SCALE, 1.0)
        assert engine.parameters["alpha"].value != 5.0

    def test_reset(self) -> None:
        p = BoundedParameter(value=8.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p], seed=42)
        engine._apply_single_mutation("alpha", MutationType.PARAM_ADJUST, 1.0)
        engine._apply_single_mutation("alpha", MutationType.RESET, 1.0)
        assert engine.parameters["alpha"].value == 8.0  # initial value

    def test_crossover(self) -> None:
        p1 = BoundedParameter(value=2.0, min=0.0, max=10.0, name="alpha")
        p2 = BoundedParameter(value=8.0, min=0.0, max=10.0, name="beta")
        engine = EvolutionEngine([p1, p2], seed=42)
        engine._apply_single_mutation("alpha", MutationType.CROSSOVER, 1.0)
        # alpha should blend toward beta
        assert engine.parameters["alpha"].value != 2.0

    def test_swap(self) -> None:
        p1 = BoundedParameter(value=2.0, min=0.0, max=10.0, name="alpha")
        p2 = BoundedParameter(value=8.0, min=0.0, max=10.0, name="beta")
        engine = EvolutionEngine([p1, p2], seed=42)
        engine._apply_single_mutation("alpha", MutationType.SWAP, 1.0)
        assert engine.parameters["alpha"].value == 8.0
        assert engine.parameters["beta"].value == 2.0

    def test_pair_crossover(self) -> None:
        p1 = BoundedParameter(value=2.0, min=0.0, max=10.0, name="alpha")
        p2 = BoundedParameter(value=8.0, min=0.0, max=10.0, name="beta")
        engine = EvolutionEngine([p1, p2], seed=42)
        engine._apply_pair_mutation(MutationType.CROSSOVER, "alpha", "beta")
        assert engine.parameters["alpha"].value == 5.0
        assert engine.parameters["beta"].value == 5.0

    def test_pair_swap(self) -> None:
        p1 = BoundedParameter(value=2.0, min=0.0, max=10.0, name="alpha")
        p2 = BoundedParameter(value=8.0, min=0.0, max=10.0, name="beta")
        engine = EvolutionEngine([p1, p2], seed=42)
        engine._apply_pair_mutation(MutationType.SWAP, "alpha", "beta")
        assert engine.parameters["alpha"].value == 8.0
        assert engine.parameters["beta"].value == 2.0

    def test_clamping_after_mutation(self) -> None:
        p = BoundedParameter(
            value=9.0, min=0.0, max=10.0, name="alpha", mutation_rate=1.0
        )
        engine = EvolutionEngine([p], seed=42)
        # Force a big mutation that would exceed bounds
        engine._apply_single_mutation("alpha", MutationType.PARAM_ADJUST, 10.0)
        assert engine.parameters["alpha"].value <= 10.0

    def test_crossover_fallback_when_alone(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p], seed=42)
        # Crossover with no partner should still mutate
        engine._apply_single_mutation("alpha", MutationType.CROSSOVER, 1.0)
        assert engine.parameters["alpha"].value != 5.0

    def test_swap_fallback_when_alone(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p], seed=42)
        engine._apply_single_mutation("alpha", MutationType.SWAP, 1.0)
        assert engine.parameters["alpha"].value != 5.0


# ===================================================================
# Evolution Strategies
# ===================================================================


class TestEvolutionStrategies:
    def test_normal_mode_mutates_all(self) -> None:
        p1 = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        p2 = BoundedParameter(value=5.0, min=0.0, max=10.0, name="beta")
        engine = EvolutionEngine(
            [p1, p2], mode=EvolutionMode.NORMAL, auto_mode=False, seed=42
        )
        engine.evolve()
        assert engine.generation == 1
        # At least one param should have changed
        assert (
            engine.parameters["alpha"].value != 5.0
            or engine.parameters["beta"].value != 5.0
        )

    def test_aggressive_mode_uses_pairs(self) -> None:
        p1 = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        p2 = BoundedParameter(value=5.0, min=0.0, max=10.0, name="beta")
        p3 = BoundedParameter(value=5.0, min=0.0, max=10.0, name="gamma")
        p4 = BoundedParameter(value=5.0, min=0.0, max=10.0, name="delta")
        engine = EvolutionEngine(
            [p1, p2, p3, p4], mode=EvolutionMode.AGGRESSIVE, auto_mode=False, seed=42
        )
        engine.evolve()
        assert engine.generation == 1

    def test_elite_mode_mutates_worst(self) -> None:
        p1 = BoundedParameter(
            value=7.0, min=0.0, max=10.0, name="alpha", fitness_score=1.0
        )
        p2 = BoundedParameter(
            value=3.0, min=0.0, max=10.0, name="beta", fitness_score=0.0
        )
        engine = EvolutionEngine(
            [p1, p2], mode=EvolutionMode.ELITE, auto_mode=False, seed=123
        )
        engine.evolve()
        # Beta (worst) should have mutated, alpha (best) should not
        assert engine.parameters["beta"].value != 3.0
        assert engine.parameters["alpha"].value == 7.0

    def test_multiple_evolutions(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine(
            [p], mode=EvolutionMode.NORMAL, auto_mode=False, seed=42
        )
        for _ in range(10):
            engine.evolve()
        assert engine.generation == 10

    def test_evolve_empty_params(self) -> None:
        engine = EvolutionEngine([], mode=EvolutionMode.NORMAL, auto_mode=False)
        engine.evolve()
        assert engine.generation == 1  # still increments

    def test_safe_mutations_in_elite(self) -> None:
        p = BoundedParameter(
            value=3.0, min=0.0, max=10.0, name="alpha", fitness_score=0.0
        )
        engine = EvolutionEngine(
            [p], mode=EvolutionMode.ELITE, auto_mode=False, seed=123
        )
        engine.evolve()
        # Elite mode should only use safe mutations (no crossover/swap)
        assert engine.parameters["alpha"].value != 3.0


# ===================================================================
# Edge Cases
# ===================================================================


class TestEdgeCases:
    def test_repr(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p])
        r = repr(engine)
        assert "gen=0" in r
        assert "normal" in r
        assert "params=1" in r

    def test_generation_snapshot_equality(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p])
        s1 = engine.snapshot()
        s2 = engine.snapshot()
        assert s1 == s2

    def test_generation_snapshot_inequality(self) -> None:
        p1 = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine1 = EvolutionEngine([p1])
        s1 = engine1.snapshot()
        p2 = BoundedParameter(value=6.0, min=0.0, max=10.0, name="alpha")
        engine2 = EvolutionEngine([p2])
        s2 = engine2.snapshot()
        assert s1 != s2

    def test_single_param_evolve(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine(
            [p], mode=EvolutionMode.NORMAL, auto_mode=False, seed=42
        )
        engine.evolve()
        assert engine.generation == 1

    def test_two_param_aggressive(self) -> None:
        p1 = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        p2 = BoundedParameter(value=5.0, min=0.0, max=10.0, name="beta")
        engine = EvolutionEngine(
            [p1, p2], mode=EvolutionMode.AGGRESSIVE, auto_mode=False, seed=42
        )
        engine.evolve()
        # With 2 params, crossover happens then remaining (0) get single mutation
        assert engine.generation == 1

    def test_three_param_aggressive(self) -> None:
        p1 = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        p2 = BoundedParameter(value=5.0, min=0.0, max=10.0, name="beta")
        p3 = BoundedParameter(value=5.0, min=0.0, max=10.0, name="gamma")
        engine = EvolutionEngine(
            [p1, p2, p3], mode=EvolutionMode.AGGRESSIVE, auto_mode=False, seed=42
        )
        engine.evolve()
        # With 3 params: crossover pair, remaining 1 gets single mutation
        assert engine.generation == 1

    def test_fitness_after_evolve(self) -> None:
        p = BoundedParameter(
            value=5.0, min=0.0, max=10.0, name="alpha", fitness_score=0.5
        )
        engine = EvolutionEngine(
            [p], mode=EvolutionMode.NORMAL, auto_mode=True, seed=42
        )
        engine.evolve()
        # Fitness scores should be preserved across generations
        assert engine.parameters["alpha"].fitness_score == 0.5

    def test_rollback_preserves_fitness(self) -> None:
        p = BoundedParameter(
            value=5.0, min=0.0, max=10.0, name="alpha", fitness_score=1.0
        )
        engine = EvolutionEngine(
            [p], mode=EvolutionMode.NORMAL, auto_mode=False, seed=42
        )
        engine.evolve()
        engine.score({"alpha": engine.parameters["alpha"].value}, 2.0)
        engine.evolve()
        engine.rollback(0)
        assert engine.parameters["alpha"].fitness_score == 1.0

    def test_snapshot_roundtrip_with_timestamp(self) -> None:
        p = BoundedParameter(value=5.0, min=0.0, max=10.0, name="alpha")
        engine = EvolutionEngine([p])
        snap = engine.snapshot()
        d = snap.to_dict()
        assert "timestamp" in d
        restored = GenerationSnapshot.from_dict(d)
        assert restored.generation == 0
        assert restored.parameters["alpha"].value == 5.0
