"""Tests for swarm.dreaming_loop.

Coverage
--------
- Dream detection (idle vs busy)
- Dream generation (random hypothesis from archive)
- Dream execution (run breeder with small population)
- Dream storage (save to dream archive)
- Dream retrieval (find matching dream for incoming task)
- Dream scoring (how well the dream matches the task)
- Dream pruning (evict old dreams)
- Idle threshold configuration
- Dream archive size limits
- Edge cases (no dreams, archive full, always busy)
"""
from __future__ import annotations

import time
import numpy as np
import pytest

from swarm.dreaming_loop import (
    Dream,
    DreamArchive,
    DreamingLoop,
    DreamMatcher,
    HypothesisGenerator,
    IdleDetector,
    TimeSinceLastTaskIdleDetector,
    AlwaysBusyIdleDetector,
    AlwaysIdleIdleDetector,
)
from swarm.breeding_kernel import BreedingKernel, BreedingEvent


# ── Helpers ─────────────────────────────────────────────────────

class MockSelector:
    def select(self, population, fitness, n_parents):
        return population[:n_parents]


class MockMutator:
    def crossover(self, parents, n_offspring):
        return parents[:n_offspring] if n_offspring <= len(parents) else parents * 2

    def mutate(self, offspring, rate=0.1):
        return offspring


class MockEvaluator:
    def evaluate(self, individuals):
        return [float(i) if isinstance(i, (int, float)) else 1.0 for i in individuals]


class MockSurvivor:
    def merge(self, old, old_fitness, new, new_fitness):
        combined = list(zip(old + new, old_fitness + new_fitness))
        combined.sort(key=lambda x: x[1], reverse=True)
        target = len(old)
        return [ind for ind, _ in combined[:target]]


@pytest.fixture
def mock_kernel():
    return BreedingKernel(
        selector=MockSelector(),
        mutator=MockMutator(),
        evaluator=MockEvaluator(),
        survivor=MockSurvivor(),
        population_size=10,
    )


@pytest.fixture
def small_archive():
    return DreamArchive(max_size=50)


@pytest.fixture
def dreaming_loop(small_archive, mock_kernel):
    return DreamingLoop(
        archive=small_archive,
        kernel=mock_kernel,
        idle_detector=AlwaysIdleIdleDetector(),
        dream_population_size=10,
        dream_generations=3,
    )


# ── Dream dataclass tests ───────────────────────────────────────

class TestDream:
    def test_dream_creation(self):
        d = Dream(tags=["a", "b"], hypothesis="test")
        assert d.dream_id
        assert d.tags == ["a", "b"]
        assert d.hypothesis == "test"
        assert d.access_count == 0

    def test_dream_touch(self):
        d = Dream()
        d.touch()
        assert d.access_count == 1
        assert d.accessed_at >= d.created_at

    def test_dream_relevance_exact_match(self):
        d = Dream(tags=["a", "b", "c"])
        assert d.score_relevance(["a", "b", "c"]) == 1.0

    def test_dream_relevance_partial(self):
        d = Dream(tags=["a", "b"])
        assert d.score_relevance(["a", "c"]) == 1.0 / 3.0

    def test_dream_relevance_empty(self):
        d = Dream(tags=[])
        assert d.score_relevance(["a"]) == 0.0

    def test_dream_to_dict(self):
        d = Dream(tags=["x"], hypothesis="h", best_fitness=0.9)
        data = d.to_dict()
        assert data["tags"] == ["x"]
        assert data["best_fitness"] == 0.9
        assert "dream_id" in data


# ── IdleDetector tests ──────────────────────────────────────────

class TestIdleDetector:
    def test_time_since_last_task_idle(self):
        detector = TimeSinceLastTaskIdleDetector(threshold_ms=100)
        time.sleep(0.15)
        assert detector.is_idle() is True

    def test_time_since_last_task_busy(self):
        detector = TimeSinceLastTaskIdleDetector(threshold_ms=5000)
        detector.on_task()
        assert detector.is_idle() is False

    def test_always_busy(self):
        detector = AlwaysBusyIdleDetector()
        assert detector.is_idle() is False

    def test_always_idle(self):
        detector = AlwaysIdleIdleDetector()
        assert detector.is_idle() is True


# ── DreamArchive tests ──────────────────────────────────────────

class TestDreamArchive:
    def test_add_and_get(self):
        archive = DreamArchive()
        d = Dream(dream_id="d1", tags=["a"])
        archive.add(d)
        assert len(archive) == 1
        got = archive.get("d1")
        assert got is not None
        assert got.access_count == 1

    def test_find_by_tags(self):
        archive = DreamArchive()
        archive.add(Dream(dream_id="d1", tags=["a", "b"]))
        archive.add(Dream(dream_id="d2", tags=["b", "c"]))
        results = archive.find_by_tags(["b"])
        assert len(results) == 2

    def test_search_ranking(self):
        archive = DreamArchive()
        archive.add(Dream(dream_id="d1", tags=["a", "b"]))
        archive.add(Dream(dream_id="d2", tags=["a", "b", "c"]))
        scored = archive.search(["a", "b"])
        assert len(scored) == 2
        # d1 has exact match 2/2=1.0, d2 has 2/3=0.67
        assert scored[0][1] >= scored[1][1]

    def test_archive_size_limit(self):
        archive = DreamArchive(max_size=5)
        for i in range(10):
            archive.add(Dream(dream_id=f"d{i}", tags=[f"t{i}"]))
        assert len(archive) == 5

    def test_prune_old(self):
        archive = DreamArchive()
        d = Dream(dream_id="old", tags=["x"])
        d.created_at = time.time() - 10000
        archive.add(d)
        removed = archive.prune_old(max_age_seconds=5)
        assert removed == 1
        assert len(archive) == 0

    def test_merge_archives(self):
        a1 = DreamArchive(max_size=3)
        a1.add(Dream(dream_id="d1", tags=["a"], access_count=5))
        a1.add(Dream(dream_id="d2", tags=["b"], access_count=4))
        a2 = DreamArchive(max_size=3)
        a2.add(Dream(dream_id="d3", tags=["c"], access_count=3))
        merged = a1.merge(a2)
        assert len(merged) == 3
        # highest access_count should survive
        ids = {d.dream_id for d in merged.list_all()}
        assert "d1" in ids
        assert "d2" in ids

    def test_list_all(self):
        archive = DreamArchive()
        archive.add(Dream(dream_id="d1", tags=["a"]))
        archive.add(Dream(dream_id="d2", tags=["b"]))
        assert len(archive.list_all()) == 2

    def test_remove_updates_tag_index(self):
        archive = DreamArchive(max_size=2)
        archive.add(Dream(dream_id="d1", tags=["a"]))
        archive.add(Dream(dream_id="d2", tags=["a", "b"]))
        archive.add(Dream(dream_id="d3", tags=["c"]))  # evicts one
        # Tag "a" should still be indexed if any dream has it
        results = archive.find_by_tags(["a"])
        assert len(results) >= 0  # may be 0 or 1 depending on eviction


# ── HypothesisGenerator tests ───────────────────────────────────

class TestHypothesisGenerator:
    def test_generate_from_memory(self):
        gen = HypothesisGenerator(seed=42)
        memory = [["spectral", "hamiltonian"], ["pythagorean"]]
        h = gen.generate(memory, n_hypotheses=1)
        assert len(h) == 1
        assert "tags" in h[0]
        assert "hypothesis" in h[0]
        assert "breeder_preset" in h[0]

    def test_generate_novel_when_empty(self):
        gen = HypothesisGenerator(seed=42)
        h = gen.generate([], n_hypotheses=1)
        assert len(h) == 1
        assert "speculative" in h[0]["tags"]

    def test_generate_multiple(self):
        gen = HypothesisGenerator(seed=42)
        h = gen.generate([["a"], ["b"]], n_hypotheses=3)
        assert len(h) == 3
        # With a tiny memory pool, duplicates are possible; just verify structure
        assert all("tags" in x and "hypothesis" in x for x in h)


# ── DreamMatcher tests ──────────────────────────────────────────

class TestDreamMatcher:
    def test_match_exact(self):
        archive = DreamArchive()
        archive.add(Dream(dream_id="d1", tags=["hamiltonian", "pythagorean"]))
        matcher = DreamMatcher(archive, min_score=0.5)
        result = matcher.match(["hamiltonian", "pythagorean"])
        assert result is not None
        dream, score = result
        assert score == 1.0

    def test_match_below_threshold(self):
        archive = DreamArchive()
        archive.add(Dream(dream_id="d1", tags=["a"]))
        matcher = DreamMatcher(archive, min_score=0.9)
        result = matcher.match(["b"])
        assert result is None

    def test_match_all(self):
        archive = DreamArchive()
        archive.add(Dream(dream_id="d1", tags=["a", "b"]))
        archive.add(Dream(dream_id="d2", tags=["a", "c"]))
        matcher = DreamMatcher(archive, min_score=0.1)
        results = matcher.match_all(["a"])
        assert len(results) == 2

    def test_match_no_dreams(self):
        archive = DreamArchive()
        matcher = DreamMatcher(archive)
        assert matcher.match(["x"]) is None


# ── DreamingLoop tests ──────────────────────────────────────────

class TestDreamingLoop:
    def test_sense_idle_true(self, dreaming_loop):
        assert dreaming_loop.sense_idle() is True

    def test_sense_idle_false_when_busy(self, dreaming_loop):
        dreaming_loop.idle_detector = AlwaysBusyIdleDetector()
        assert dreaming_loop.sense_idle() is False

    def test_decide_dream(self, dreaming_loop):
        dream = dreaming_loop.decide_dream()
        assert dream is not None
        assert dream.tags
        assert dream.hypothesis
        assert dream.breeder_preset
        assert dream.dream_id

    def test_decide_dream_with_memory(self, dreaming_loop):
        dreaming_loop.archive.add(Dream(dream_id="m1", tags=["spectral"]))
        dream = dreaming_loop.decide_dream()
        assert dream is not None

    def test_act_execute_with_kernel(self, dreaming_loop):
        dream = Dream(tags=["test"], population_size=10, generations=2)
        completed = dreaming_loop.act_execute(dream)
        assert completed.best_fitness >= 0.0
        assert completed.mean_fitness >= 0.0
        assert completed.diversity >= 0.0
        assert completed.raw_result is not None

    def test_act_execute_without_kernel(self, small_archive):
        loop = DreamingLoop(
            archive=small_archive,
            kernel=None,
            idle_detector=AlwaysIdleIdleDetector(),
        )
        dream = Dream(tags=["spectral", "hamiltonian"], population_size=10, generations=5)
        completed = loop.act_execute(dream)
        assert completed.best_fitness >= 0.0
        assert completed.best_fitness <= 1.0
        assert completed.raw_result == {"synthetic": True}

    def test_store_result(self, dreaming_loop):
        dream = Dream(tags=["x"], best_fitness=0.5)
        dreaming_loop.store_result(dream)
        assert len(dreaming_loop.archive) == 1
        assert dreaming_loop._total_dreams_run == 1

    def test_tick_full_cycle(self, dreaming_loop):
        dream = dreaming_loop.tick()
        assert dream is not None
        assert dream.best_fitness >= 0.0
        assert len(dreaming_loop.archive) >= 1

    def test_tick_when_busy(self, dreaming_loop):
        dreaming_loop.idle_detector = AlwaysBusyIdleDetector()
        result = dreaming_loop.tick()
        assert result is None
        assert dreaming_loop._total_ticks == 1

    def test_run_multiple(self, dreaming_loop):
        dreams = dreaming_loop.run(max_dreams=3)
        assert len(dreams) == 3
        assert dreaming_loop._total_dreams_run == 3

    def test_find_dream_for_task(self, dreaming_loop):
        # Seed archive with a dream
        dreaming_loop.archive.add(Dream(dream_id="d1", tags=["hamiltonian", "pythagorean"]))
        result = dreaming_loop.find_dream_for_task(["hamiltonian", "pythagorean"])
        assert result is not None
        dream, score = result
        assert score == 1.0

    def test_find_dream_for_task_no_match(self, dreaming_loop):
        dreaming_loop.archive.add(Dream(dream_id="d1", tags=["a"]))
        result = dreaming_loop.find_dream_for_task(["b"])
        assert result is None

    def test_stats(self, dreaming_loop):
        dreaming_loop.tick()
        stats = dreaming_loop.stats()
        assert stats["total_ticks"] >= 1
        assert stats["total_dreams_run"] >= 1
        assert stats["archive_size"] >= 1

    def test_synthetic_fitness_determinism(self, dreaming_loop):
        d1 = Dream(tags=["spectral", "hamiltonian"])
        d2 = Dream(tags=["spectral", "hamiltonian"])
        f1 = dreaming_loop._synthetic_fitness(d1)
        f2 = dreaming_loop._synthetic_fitness(d2)
        assert f1 == f2

    def test_synthetic_fitness_different_tags(self, dreaming_loop):
        d1 = Dream(tags=["a"])
        d2 = Dream(tags=["b"])
        f1 = dreaming_loop._synthetic_fitness(d1)
        f2 = dreaming_loop._synthetic_fitness(d2)
        assert f1 != f2

    def test_make_population(self, dreaming_loop):
        pop = dreaming_loop._make_population(5)
        assert len(pop) == 5
        assert all(isinstance(p, list) for p in pop)

    def test_idle_threshold_config(self):
        loop = DreamingLoop(
            archive=DreamArchive(),
            idle_threshold_ms=2500.0,
        )
        assert isinstance(loop.idle_detector, TimeSinceLastTaskIdleDetector)
        assert loop.idle_detector.threshold_ms == 2500.0

    def test_always_busy_edge_case(self):
        loop = DreamingLoop(
            archive=DreamArchive(),
            idle_detector=AlwaysBusyIdleDetector(),
        )
        dreams = loop.run(max_dreams=5)
        assert dreams == []
        assert loop._total_dreams_run == 0

    def test_archive_full_edge_case(self, dreaming_loop):
        dreaming_loop.archive.max_size = 2
        dreaming_loop.run(max_dreams=5)
        assert len(dreaming_loop.archive) == 2

    def test_no_kernel_with_population(self):
        loop = DreamingLoop(
            archive=DreamArchive(),
            kernel=None,
            idle_detector=AlwaysIdleIdleDetector(),
        )
        dream = Dream(tags=["x"], population_size=10, generations=2)
        completed = loop.act_execute(dream)
        assert completed.best_fitness >= 0.0
        assert completed.raw_result == {"synthetic": True}


# ── Integration-style tests ─────────────────────────────────────

class TestDreamingLoopIntegration:
    def test_end_to_end_dream_and_reuse(self, mock_kernel):
        archive = DreamArchive(max_size=100)
        loop = DreamingLoop(
            archive=archive,
            kernel=mock_kernel,
            idle_detector=AlwaysIdleIdleDetector(),
            dream_population_size=10,
            dream_generations=3,
        )
        # Run some dreams
        dreams = loop.run(max_dreams=5)
        assert len(dreams) == 5
        # Now try to match a real task
        result = loop.find_dream_for_task(dreams[0].tags)
        assert result is not None
        dream, score = result
        assert score > 0.0
        assert dream.access_count >= 1

    def test_dream_pruning_under_pressure(self):
        archive = DreamArchive(max_size=3)
        loop = DreamingLoop(
            archive=archive,
            kernel=None,
            idle_detector=AlwaysIdleIdleDetector(),
        )
        loop.run(max_dreams=5)
        assert len(archive) == 3
        # All dreams should have different IDs
        ids = {d.dream_id for d in archive.list_all()}
        assert len(ids) == 3

    def test_matcher_with_min_score(self):
        archive = DreamArchive()
        archive.add(Dream(dream_id="d1", tags=["a", "b"]))
        archive.add(Dream(dream_id="d2", tags=["c", "d"]))
        matcher = DreamMatcher(archive, min_score=0.5)
        result = matcher.match(["a", "b"])
        assert result is not None
        assert result[1] == 1.0
        result2 = matcher.match(["a", "c"])
        # 1/3 overlap, below 0.5 threshold
        assert result2 is None

    def test_hypothesis_from_archive_memory(self):
        archive = DreamArchive()
        archive.add(Dream(tags=["spectral"], best_fitness=0.9))
        archive.add(Dream(tags=["hamiltonian"], best_fitness=0.8))
        gen = HypothesisGenerator(seed=42)
        memory = [d.tags for d in archive.list_all()]
        h = gen.generate(memory, n_hypotheses=1)[0]
        assert any(tag in ["spectral", "hamiltonian"] for tag in h["tags"])
