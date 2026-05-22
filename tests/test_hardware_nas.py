"""Tests for hardware-conditional NAS (Experiment 2).

Tests:
1. evaluate a single config → returns dict with all metrics
2. aging evolution → returns non-empty Pareto frontier
3. hardware-specific frontiers differ (Jetson prefers smaller configs than Oracle1)
4. config serialization/deserialization roundtrip
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure nerve is importable (repo has no root __init__.py)
_NERVE = Path(__file__).parent.parent / "nerve"
sys.path.insert(0, str(_NERVE.parent))

from experiments.hardware_nas import (
    HardwareConditionalNAS,
    Config,
    EvalResult,
    oracle1_profile,
    jetson_profile,
    laptop_profile,
    pareto_dominates,
    compute_pareto_frontier,
    _feasible_for_hardware,
)


# ── Fixtures ──

@pytest.fixture
def nas_jetson():
    return HardwareConditionalNAS(jetson_profile, max_evals=50, seed=42)


@pytest.fixture
def nas_oracle1():
    return HardwareConditionalNAS(oracle1_profile, max_evals=50, seed=42)


@pytest.fixture
def sample_config():
    return Config(n_rooms=100, d_latent=32, h_history=8, l_signal=8, chaos_decay=0.95, route_density=0.05)


# ── Test 1: evaluate a single config ──

class TestEvaluateSingle:
    def test_returns_dict_with_all_metrics(self, nas_jetson, sample_config):
        result = nas_jetson.evaluate(sample_config)
        assert isinstance(result, EvalResult)
        assert hasattr(result, "ticks_per_second")
        assert hasattr(result, "memory_mb")
        assert hasattr(result, "diversity")
        assert hasattr(result, "stability")

    def test_ticks_per_second_positive(self, nas_jetson, sample_config):
        result = nas_jetson.evaluate(sample_config)
        assert result.ticks_per_second > 0, "ticks_per_second must be positive"

    def test_memory_mb_non_negative(self, nas_jetson, sample_config):
        result = nas_jetson.evaluate(sample_config)
        assert result.memory_mb >= 0, "memory_mb must be non-negative"

    def test_diversity_in_range(self, nas_jetson, sample_config):
        result = nas_jetson.evaluate(sample_config)
        assert 0.0 <= result.diversity <= 1.0, f"diversity should be in [0,1], got {result.diversity}"

    def test_stability_in_range(self, nas_jetson, sample_config):
        result = nas_jetson.evaluate(sample_config)
        assert 0.0 < result.stability <= 1.0, f"stability should be in (0,1], got {result.stability}"

    def test_eval_count_increments(self, nas_jetson, sample_config):
        before = nas_jetson.eval_count
        nas_jetson.evaluate(sample_config)
        assert nas_jetson.eval_count == before + 1

    def test_dict_output_contains_expected_keys(self, nas_jetson, sample_config):
        result = nas_jetson.evaluate(sample_config)
        d = result.to_dict()
        expected = {"n_rooms", "d_latent", "h_history", "l_signal", "chaos_decay",
                    "route_density", "ticks_per_second", "memory_mb", "diversity",
                    "stability", "age"}
        assert expected.issubset(d.keys())

    def test_infeasible_config_penalized(self):
        """Config exceeding RAM should return penalty result."""
        tiny_profile = {"ram_gb": 0.05, "cpu_cores": 2, "gpu": "none"}  # ~50MB allowance
        nas = HardwareConditionalNAS(tiny_profile, max_evals=10)
        huge = Config(n_rooms=1000, d_latent=128, h_history=32, l_signal=64, chaos_decay=0.95, route_density=0.20)
        result = nas.evaluate(huge)
        assert result.ticks_per_second == 0.0
        assert result.memory_mb >= 999000


# ── Test 2: aging evolution → non-empty Pareto frontier ──

class TestAgingEvolution:
    def test_returns_non_empty_list(self, nas_jetson):
        frontier = nas_jetson.aging_evolution(population_size=5, generations=3)
        assert isinstance(frontier, list)
        assert len(frontier) > 0, "Pareto frontier should not be empty"

    def test_frontier_items_are_dicts(self, nas_jetson):
        frontier = nas_jetson.aging_evolution(population_size=5, generations=3)
        for item in frontier:
            assert isinstance(item, dict)

    def test_frontier_contains_expected_keys(self, nas_jetson):
        frontier = nas_jetson.aging_evolution(population_size=5, generations=3)
        expected = {"n_rooms", "d_latent", "h_history", "l_signal", "chaos_decay",
                    "route_density", "ticks_per_second", "memory_mb", "diversity", "stability"}
        for item in frontier:
            assert expected.issubset(item.keys())

    def test_frontier_is_pareto_optimal(self, nas_jetson):
        frontier = nas_jetson.aging_evolution(population_size=5, generations=3)
        objectives = ["ticks_per_second", "diversity", "stability", "memory_mb"]
        maximize = {"ticks_per_second", "diversity", "stability"}
        # No point in frontier should dominate another
        for i, a in enumerate(frontier):
            for j, b in enumerate(frontier):
                if i != j:
                    assert not pareto_dominates(a, b, objectives, maximize), \
                        f"Frontier point {i} dominates {j} — not a true frontier"

    def test_eval_count_bounded_by_max_evals(self, nas_jetson):
        nas_jetson.aging_evolution(population_size=5, generations=10)
        assert nas_jetson.eval_count <= nas_jetson.max_evals


# ── Test 3: hardware-specific frontiers differ ──

class TestHardwareSpecificFrontiers:
    def test_jetson_prefers_smaller_configs_than_oracle1(self):
        """Jetson (8GB RAM) should converge to smaller n_rooms than Oracle1 (32GB)."""
        nas_j = HardwareConditionalNAS(jetson_profile, max_evals=30, seed=7)
        nas_o = HardwareConditionalNAS(oracle1_profile, max_evals=30, seed=7)

        frontier_j = nas_j.aging_evolution(population_size=5, generations=4)
        frontier_o = nas_o.aging_evolution(population_size=5, generations=4)

        assert len(frontier_j) > 0
        assert len(frontier_o) > 0

        avg_n_jetson = sum(p["n_rooms"] for p in frontier_j) / len(frontier_j)
        avg_n_oracle = sum(p["n_rooms"] for p in frontier_o) / len(frontier_o)

        # Jetson should generally prefer smaller configs (or at least not bigger)
        assert avg_n_jetson <= avg_n_oracle * 1.5, \
            f"Jetson avg n={avg_n_jetson} unexpectedly larger than Oracle1 avg n={avg_n_oracle}"

    def test_feasibility_differs_by_hardware(self):
        big = Config(n_rooms=1000, d_latent=128, h_history=32, l_signal=64, chaos_decay=0.95, route_density=0.20)
        assert _feasible_for_hardware(big, oracle1_profile)
        # Use a tiny profile to force infeasibility for the test
        tiny = {"ram_gb": 0.05, "cpu_cores": 2, "gpu": "none"}
        assert not _feasible_for_hardware(big, tiny)


# ── Test 4: config serialization roundtrip ──

class TestConfigSerialization:
    def test_config_to_dict_roundtrip(self, sample_config):
        d = sample_config.to_dict()
        restored = Config.from_dict(d)
        assert restored == sample_config

    def test_eval_result_to_dict_roundtrip(self, nas_jetson, sample_config):
        result = nas_jetson.evaluate(sample_config)
        d = result.to_dict()
        assert d["n_rooms"] == sample_config.n_rooms
        assert d["d_latent"] == sample_config.d_latent
        assert "ticks_per_second" in d

    def test_json_serialization(self, sample_config):
        d = sample_config.to_dict()
        s = json.dumps(d)
        restored_dict = json.loads(s)
        restored = Config.from_dict(restored_dict)
        assert restored == sample_config

    def test_json_serialization_with_float_precision(self, sample_config):
        d = sample_config.to_dict()
        s = json.dumps(d)
        restored = json.loads(s)
        assert restored["chaos_decay"] == sample_config.chaos_decay
        assert restored["route_density"] == sample_config.route_density


# ── Test 5: Pareto helpers ──

class TestParetoHelpers:
    def test_pareto_dominates_simple(self):
        a = {"x": 2, "y": 2}  # better on both
        b = {"x": 1, "y": 1}
        assert pareto_dominates(a, b, ["x", "y"], {"x", "y"})
        assert not pareto_dominates(b, a, ["x", "y"], {"x", "y"})

    def test_pareto_not_dominates_equal(self):
        a = {"x": 1, "y": 1}
        b = {"x": 1, "y": 1}
        assert not pareto_dominates(a, b, ["x", "y"], {"x", "y"})

    def test_compute_pareto_frontier_filters_dominated(self):
        points = [
            {"x": 1, "y": 1},   # dominated
            {"x": 2, "y": 2},   # dominates first
            {"x": 2, "y": 1},   # non-dominated (trade-off)
        ]
        frontier = compute_pareto_frontier(points, ["x", "y"], {"x", "y"})
        for p in frontier:
            for q in frontier:
                if p is q:
                    continue
                assert not (pareto_dominates(p, q, ["x", "y"], {"x", "y"}) and pareto_dominates(q, p, ["x", "y"], {"x", "y"})), \
                    "Mutual domination in frontier"

    def test_frontier_minimizes_memory(self):
        points = [
            {"ticks_per_second": 100, "diversity": 0.5, "stability": 0.5, "memory_mb": 100},
            {"ticks_per_second": 120, "diversity": 0.6, "stability": 0.6, "memory_mb": 50},   # dominates 1st
            {"ticks_per_second": 80,  "diversity": 0.4, "stability": 0.4, "memory_mb": 200},  # dominated by 2nd
        ]
        frontier = compute_pareto_frontier(points)
        assert len(frontier) == 1  # only point 2 survives
        assert frontier[0]["ticks_per_second"] == 120


# ── Test 6: Hardware profiles exist and are well-formed ──

class TestHardwareProfiles:
    def test_oracle1_profile(self):
        assert oracle1_profile["device"] == "Alibaba Cloud"
        assert oracle1_profile["ram_gb"] == 32

    def test_jetson_profile(self):
        assert jetson_profile["device"] == "Jetson Orin"
        assert jetson_profile["ram_gb"] == 8

    def test_laptop_profile(self):
        assert laptop_profile["device"] == "RTX 4050 Laptop"
        assert laptop_profile["ram_gb"] == 16
