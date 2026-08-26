"""Tests for BreedOptimizer — intelligent breeding optimization.

Reference: fleet/breed_optimizer.py
"""

from __future__ import annotations

import random

import pytest

from fleet.breed_optimizer import (
    AnomalyResult,
    BreedOptimizer,
    BreedingArchive,
    OffspringPrediction,
    ParentPair,
)


class TestWassersteinDistance:
    def test_identical_distributions(self) -> None:
        opt = BreedOptimizer()
        dist = opt.wasserstein_distance([1, 2, 3], [1, 2, 3])
        assert dist == 0.0

    def test_different_distributions(self) -> None:
        opt = BreedOptimizer()
        dist = opt.wasserstein_distance([0, 0, 0], [1, 1, 1])
        assert dist > 0.0
        assert dist == 1.0

    def test_empty_distributions(self) -> None:
        opt = BreedOptimizer()
        assert opt.wasserstein_distance([], [1, 2]) == float("inf")
        assert opt.wasserstein_distance([1, 2], []) == float("inf")

    def test_unequal_lengths(self) -> None:
        opt = BreedOptimizer()
        dist = opt.wasserstein_distance([1, 2, 3, 4, 5], [1, 2])
        assert dist >= 0.0

    def test_single_element(self) -> None:
        opt = BreedOptimizer()
        dist = opt.wasserstein_distance([5], [10])
        assert dist == 5.0

    def test_diversity_score(self) -> None:
        opt = BreedOptimizer()
        score = opt.diversity_score([0.1, 0.2], [0.8, 0.9])
        assert 0.0 < score <= 1.0

    def test_diversity_identical(self) -> None:
        opt = BreedOptimizer()
        score = opt.diversity_score([0.5, 0.5], [0.5, 0.5])
        assert score == 0.0


class TestParentSelection:
    def test_select_parents_basic(self) -> None:
        opt = BreedOptimizer()
        pool = [
            {"id": "a", "traits": [0.1, 0.2, 0.3]},
            {"id": "b", "traits": [0.8, 0.9, 0.7]},
            {"id": "c", "traits": [0.5, 0.5, 0.5]},
        ]
        pairs = opt.select_parents(pool, k=2)
        assert len(pairs) > 0
        assert len(pairs) <= 2
        assert isinstance(pairs[0], ParentPair)

    def test_select_parents_insufficient_pool(self) -> None:
        opt = BreedOptimizer()
        pool = [{"id": "a", "traits": [0.1]}]
        pairs = opt.select_parents(pool, k=3)
        assert pairs == []

    def test_select_parents_no_traits(self) -> None:
        opt = BreedOptimizer()
        pool = [
            {"id": "a"},
            {"id": "b", "traits": [0.1]},
        ]
        pairs = opt.select_parents(pool, k=3)
        # Should skip pairs without traits
        assert len(pairs) == 0 or all(p.diversity_score >= 0 for p in pairs)

    def test_parent_pair_fields(self) -> None:
        opt = BreedOptimizer()
        pool = [
            {"id": "a", "traits": [0.1, 0.2]},
            {"id": "b", "traits": [0.9, 0.8]},
        ]
        pairs = opt.select_parents(pool, k=1)
        assert len(pairs) == 1
        p = pairs[0]
        assert p.parent_a in ["a", "b"]
        assert p.parent_b in ["a", "b"]
        assert p.parent_a != p.parent_b
        assert p.diversity_score >= 0.0
        assert p.wasserstein_distance >= 0.0


class TestOffspringPrediction:
    def test_predict_offspring(self) -> None:
        opt = BreedOptimizer()
        pred = opt._predict_offspring([0.5, 0.6], [0.7, 0.8])
        assert isinstance(pred, OffspringPrediction)
        assert 0.0 <= pred.expected_fitness <= 1.0
        assert 0.0 <= pred.confidence <= 1.0
        assert 0.0 <= pred.novelty_score <= 1.0
        assert pred.risk_level in ["low", "medium", "high"]

    def test_predict_offspring_empty(self) -> None:
        opt = BreedOptimizer()
        pred = opt._predict_offspring([], [0.5])
        assert pred.expected_fitness == 0.0
        assert pred.confidence == 0.0

    def test_predict_diverse_parents(self) -> None:
        opt = BreedOptimizer()
        pred = opt._predict_offspring([0.1, 0.1], [0.9, 0.9])
        assert pred.novelty_score > 0.5  # Diverse parents = high novelty
        assert pred.risk_level == "high"  # High distance = high risk


class TestAnomalyDetection:
    def test_detect_no_anomalies_short_history(self) -> None:
        opt = BreedOptimizer()
        anomalies = opt.detect_anomalies([], threshold=2.0)
        assert anomalies == []

    def test_detect_fitness_anomaly(self) -> None:
        opt = BreedOptimizer()
        history = [
            {
                "parent_a": "a",
                "parent_b": "b",
                "offspring_fitness": 0.8,
                "diversity": 0.5,
            },
            {
                "parent_a": "a",
                "parent_b": "b",
                "offspring_fitness": 0.85,
                "diversity": 0.5,
            },
            {
                "parent_a": "a",
                "parent_b": "b",
                "offspring_fitness": 0.82,
                "diversity": 0.5,
            },
            {
                "parent_a": "a",
                "parent_b": "b",
                "offspring_fitness": 0.81,
                "diversity": 0.5,
            },
            {
                "parent_a": "a",
                "parent_b": "b",
                "offspring_fitness": 0.83,
                "diversity": 0.5,
            },
            {
                "parent_a": "a",
                "parent_b": "b",
                "offspring_fitness": 0.1,
                "diversity": 0.5,
            },  # Anomaly
        ]
        anomalies = opt.detect_anomalies(history, threshold=2.0)
        assert len(anomalies) > 0
        assert any(a.is_anomaly for a in anomalies)
        assert any("Fitness" in a.reason for a in anomalies)

    def test_detect_inbreeding(self) -> None:
        opt = BreedOptimizer()
        history = [
            {
                "parent_a": "a",
                "parent_b": "b",
                "offspring_fitness": 0.5,
                "diversity": 0.5,
            }
            for _ in range(10)
        ]
        anomalies = opt.detect_anomalies(history, threshold=2.0)
        assert len(anomalies) > 0
        assert any("Inbreeding" in a.reason for a in anomalies)

    def test_detect_diversity_collapse(self) -> None:
        opt = BreedOptimizer()
        history = [
            {
                "parent_a": "a",
                "parent_b": "b",
                "offspring_fitness": 0.5,
                "diversity": 0.5,
            },
            {
                "parent_a": "c",
                "parent_b": "d",
                "offspring_fitness": 0.5,
                "diversity": 0.5,
            },
            {
                "parent_a": "e",
                "parent_b": "f",
                "offspring_fitness": 0.5,
                "diversity": 0.05,
            },
            {
                "parent_a": "g",
                "parent_b": "h",
                "offspring_fitness": 0.5,
                "diversity": 0.04,
            },
            {
                "parent_a": "i",
                "parent_b": "j",
                "offspring_fitness": 0.5,
                "diversity": 0.03,
            },
            {
                "parent_a": "k",
                "parent_b": "l",
                "offspring_fitness": 0.5,
                "diversity": 0.02,
            },
            {
                "parent_a": "m",
                "parent_b": "n",
                "offspring_fitness": 0.5,
                "diversity": 0.01,
            },
        ]
        anomalies = opt.detect_anomalies(history, threshold=2.0)
        assert len(anomalies) > 0
        assert any("Diversity collapse" in a.reason for a in anomalies)

    def test_anomaly_result_fields(self) -> None:
        result = AnomalyResult(
            is_anomaly=True,
            anomaly_score=2.5,
            reason="Test",
            affected_parents=["a", "b"],
        )
        assert result.is_anomaly is True
        assert result.anomaly_score == 2.5
        assert result.affected_parents == ["a", "b"]


class TestBreedingArchive:
    def test_archive_add(self) -> None:
        archive = BreedingArchive()
        archive.add((0.5, 0.5), {"fitness": 0.8, "id": "agent1"})
        assert len(archive.cells) > 0

    def test_archive_coverage(self) -> None:
        archive = BreedingArchive()
        assert archive.coverage == 0.0
        archive.add((0.5, 0.5), {"fitness": 0.8})
        assert archive.coverage > 0.0

    def test_archive_get_best(self) -> None:
        archive = BreedingArchive()
        archive.add((0.5, 0.5), {"fitness": 0.8})
        archive.add((0.5, 0.5), {"fitness": 0.9})
        best = archive.get_best_in_cell((5, 5))  # 0.5 * 10 = 5
        assert best is not None
        assert best["fitness"] == 0.9

    def test_archive_sample_diverse(self) -> None:
        archive = BreedingArchive()
        for i in range(20):
            archive.add((i / 20.0, i / 20.0), {"fitness": 0.5 + i / 40.0})
        samples = archive.sample_diverse(k=5)
        assert len(samples) > 0
        assert len(samples) <= 5

    def test_archive_qd_score(self) -> None:
        archive = BreedingArchive()
        assert archive.qd_score == 0.0
        archive.add((0.5, 0.5), {"fitness": 0.8})
        assert archive.qd_score > 0.0


class TestOptimizeArchive:
    def test_optimize_archive(self) -> None:
        opt = BreedOptimizer()
        archive = BreedingArchive()
        for i in range(50):
            archive.add(
                (random.random(), random.random()),
                {"fitness": 0.5 + random.random() * 0.5, "traits": [random.random()]},
            )
        result = opt.optimize_archive(archive, iterations=20)
        assert isinstance(result, BreedingArchive)
        assert result.coverage >= archive.coverage  # Should not decrease

    def test_optimize_empty_archive(self) -> None:
        opt = BreedOptimizer()
        archive = BreedingArchive()
        result = opt.optimize_archive(archive, iterations=10)
        assert result.coverage == 0.0


class TestHistoryManagement:
    def test_record_breeding(self) -> None:
        opt = BreedOptimizer()
        opt.record_breeding("a", "b", 0.8, 0.5, [0.1, 0.2])
        assert len(opt.get_history()) == 1
        assert opt.get_history()[0]["parent_a"] == "a"
        assert opt.get_history()[0]["offspring_fitness"] == 0.8

    def test_get_stats_empty(self) -> None:
        opt = BreedOptimizer()
        stats = opt.get_stats()
        assert stats["count"] == 0
        assert stats["mean_fitness"] == 0.0

    def test_get_stats_with_history(self) -> None:
        opt = BreedOptimizer()
        opt.record_breeding("a", "b", 0.8, 0.5)
        opt.record_breeding("c", "d", 0.6, 0.3)
        stats = opt.get_stats()
        assert stats["count"] == 2
        assert stats["mean_fitness"] == 0.7
        assert stats["mean_diversity"] == 0.4

    def test_stats_archive_metrics(self) -> None:
        opt = BreedOptimizer()
        opt.archive.add((0.5, 0.5), {"fitness": 0.9})
        stats = opt.get_stats()
        assert stats["archive_coverage"] > 0.0
        assert stats["archive_qd_score"] > 0.0


class TestDeadlineManagement:
    def test_set_deadline_without_tminus(self) -> None:
        opt = BreedOptimizer()
        deadline = opt.set_breeding_deadline(60.0, 120.0)
        assert deadline == 60.0

    def test_set_deadline_child_smaller(self) -> None:
        opt = BreedOptimizer()
        deadline = opt.set_breeding_deadline(120.0, 60.0)
        assert deadline == 60.0


class TestReport:
    def test_generate_report(self) -> None:
        opt = BreedOptimizer(node_id="test-node")
        opt.record_breeding("a", "b", 0.8, 0.5)
        report = opt.generate_report()
        assert report["node_id"] == "test-node"
        assert report["history_size"] == 1
        assert report["swarm_connected"] is False
        assert report["cache_connected"] is False
        assert report["tminus_connected"] is False

    def test_report_with_connections(self) -> None:
        # Can't easily mock, but we can test the structure
        opt = BreedOptimizer(node_id="connected")
        report = opt.generate_report()
        assert isinstance(report["swarm_connected"], bool)
        assert isinstance(report["cache_connected"], bool)
        assert isinstance(report["tminus_connected"], bool)


class TestEdgeCases:
    def test_select_parents_all_same(self) -> None:
        opt = BreedOptimizer()
        pool = [
            {"id": "a", "traits": [0.5, 0.5]},
            {"id": "b", "traits": [0.5, 0.5]},
        ]
        pairs = opt.select_parents(pool, k=1)
        assert len(pairs) == 1
        assert pairs[0].diversity_score == 0.0

    def test_detect_anomalies_empty_history(self) -> None:
        opt = BreedOptimizer()
        anomalies = opt.detect_anomalies([], threshold=2.0)
        assert anomalies == []

    def test_wasserstein_with_negatives(self) -> None:
        opt = BreedOptimizer()
        dist = opt.wasserstein_distance([-1, 0, 1], [0, 0, 0])
        assert dist >= 0.0

    def test_archive_add_none(self) -> None:
        archive = BreedingArchive()
        # Adding None should not crash, but get_best_in_cell should handle it
        archive.add((0.5, 0.5), None)
        best = archive.get_best_in_cell((5, 5))
        # None is stored but get_best_in_cell filters None
        assert best is None

    def test_archive_handles_none_cell(self) -> None:
        archive = BreedingArchive()
        archive.add((0.5, 0.5), None)
        archive.add((0.5, 0.5), {"fitness": 0.9})
        best = archive.get_best_in_cell((5, 5))
        assert best is not None
        assert best["fitness"] == 0.9
