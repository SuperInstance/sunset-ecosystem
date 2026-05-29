import pytest
import numpy as np
from fleet.benchmark_suite import BenchmarkResult, BenchmarkSuite
from swarm.differential_breeder import DifferentialBreeder


class TestBenchmarkResult:
    def test_to_dict(self):
        r = BenchmarkResult(
            problem="sphere",
            breeder="de",
            best_fitness=0.9,
            generations=100,
            time_seconds=1.0,
            final_diversity=0.5,
            success=True,
        )
        d = r.to_dict()
        assert d["problem"] == "sphere"
        assert d["success"] is True


class TestBenchmarkSuite:
    def test_init(self):
        bs = BenchmarkSuite()
        assert bs.max_generations == 100
        assert bs.results == []

    def test_get_problems(self):
        bs = BenchmarkSuite()
        problems = bs.get_problems()
        assert "sphere" in problems
        assert "rastrigin" in problems
        assert "ackley" in problems

    def test_sphere(self):
        bs = BenchmarkSuite()
        genome = np.array([0.0, 0.0, 0.0])
        fitness = bs._sphere(genome)
        assert fitness == 0.0

    def test_onemax(self):
        bs = BenchmarkSuite()
        genome = np.array([1.0, 1.0, 1.0])
        fitness = bs._onemax(genome)
        assert fitness == 3.0

    def test_leading_ones(self):
        bs = BenchmarkSuite()
        genome = np.array([1.0, 1.0, 0.0, 1.0])
        fitness = bs._leading_ones(genome)
        assert fitness == 2.0

    def test_run_breeder(self):
        bs = BenchmarkSuite(max_generations=10)
        breeder = DifferentialBreeder(population_size=10, dimensions=3, bounds=(-1, 1))
        result = bs.run_breeder("de_test", breeder, "sphere", dimensions=3)
        assert result.problem == "sphere"
        assert result.breeder == "de_test"
        assert result.success is True
        assert result.generations == 10

    def test_run_all(self):
        bs = BenchmarkSuite(max_generations=5)
        breeder = DifferentialBreeder(population_size=10, dimensions=3, bounds=(-1, 1))
        results = bs.run_all("de_test", breeder)
        assert len(results) == len(bs.get_problems())

    def test_compare_breeders(self):
        bs = BenchmarkSuite(max_generations=5)
        breeder1 = DifferentialBreeder(population_size=10, dimensions=3, bounds=(-1, 1))
        results1 = bs.run_all("de_1", breeder1)
        comparison = bs.compare_breeders({"de_1": results1})
        assert "sphere" in comparison

    def test_get_leaderboard(self):
        bs = BenchmarkSuite(max_generations=5)
        breeder = DifferentialBreeder(population_size=10, dimensions=3, bounds=(-1, 1))
        bs.run_all("de", breeder)
        leaderboard = bs.get_leaderboard()
        assert len(leaderboard) > 0

    def test_export_json(self):
        bs = BenchmarkSuite(max_generations=5)
        breeder = DifferentialBreeder(population_size=10, dimensions=3, bounds=(-1, 1))
        bs.run_all("de", breeder)
        j = bs.export_json()
        assert "sphere" in j
        assert "results" in j

    def test_to_dict(self):
        bs = BenchmarkSuite(max_generations=5)
        breeder = DifferentialBreeder(population_size=10, dimensions=3, bounds=(-1, 1))
        bs.run_all("de", breeder)
        d = bs.to_dict()
        assert "problems" in d
        assert d["results"] > 0
