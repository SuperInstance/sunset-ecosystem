"""Tests for OpenConstruct Bridge — harness integration.

Covers ConstructManifest, HarnessAdapter, BuildCoordinator, ProgressStreamer,
and validation gates.
"""

import json
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from fleet.openconstruct_bridge import (
    ConstructManifest,
    BreedingEvent,
    ValidationGate,
    HarnessAdapter,
    BuildCoordinator,
    ProgressStreamer,
    exact_arithmetic_gate,
    spectral_real_gate,
    robustness_gate,
)


class TestConstructManifest:
    def test_basic(self):
        m = ConstructManifest(
            name="test-construct",
            breeder_type="pythagorean",
            goal="Test goal",
        )
        assert m.name == "test-construct"
        assert m.breeder_type == "pythagorean"
        assert m.population_size == 50

    def test_to_dict(self):
        m = ConstructManifest(name="t", breeder_type="b", goal="g")
        d = m.to_dict()
        assert d["name"] == "t"
        assert d["breeder_type"] == "b"

    def test_from_dict(self):
        d = {
            "name": "t",
            "breeder_type": "b",
            "goal": "g",
            "population_size": 100,
        }
        m = ConstructManifest.from_dict(d)
        assert m.population_size == 100

    def test_with_qd(self):
        m = ConstructManifest(
            name="qd-test",
            breeder_type="pythagorean",
            goal="Test QD",
            qd_dimensions=[(3, 4, 5), (5, 12, 13)],
            qd_resolution=10,
        )
        assert len(m.qd_dimensions) == 2
        assert m.qd_resolution == 10


class TestBreedingEvent:
    def test_creation(self):
        e = BreedingEvent(
            generation=5,
            best_fitness=1.5,
            mean_fitness=1.0,
            population_size=50,
            elapsed_seconds=10.0,
        )
        assert e.generation == 5
        assert e.best_fitness == 1.5

    def test_to_json(self):
        e = BreedingEvent(
            generation=1,
            best_fitness=2.0,
            mean_fitness=1.5,
            population_size=20,
            elapsed_seconds=5.0,
        )
        s = e.to_json()
        d = json.loads(s)
        assert d["generation"] == 1
        assert d["best_fitness"] == 2.0

    def test_qd_metrics(self):
        e = BreedingEvent(
            generation=0,
            best_fitness=0.0,
            mean_fitness=0.0,
            population_size=0,
            elapsed_seconds=0.0,
            qd_coverage=0.8,
            qd_score=150.0,
            num_bins=12,
        )
        assert e.qd_coverage == 0.8
        assert e.qd_score == 150.0

    def test_consensus_metrics(self):
        e = BreedingEvent(
            generation=0,
            best_fitness=0.0,
            mean_fitness=0.0,
            population_size=0,
            elapsed_seconds=0.0,
            nodes_agreed=3,
            total_nodes=5,
        )
        assert e.nodes_agreed == 3
        assert e.total_nodes == 5


class TestValidationGate:
    def test_pass(self):
        gate = ValidationGate(
            name="always_pass",
            check_fn=lambda g: (True, "OK"),
            required=True,
        )
        passed, msg = gate.validate("anything")
        assert passed is True
        assert msg == "OK"

    def test_fail(self):
        gate = ValidationGate(
            name="always_fail",
            check_fn=lambda g: (False, "BAD"),
            required=True,
        )
        passed, msg = gate.validate("anything")
        assert passed is False
        assert msg == "BAD"

    def test_hard_vs_soft(self):
        hard = ValidationGate("hard", lambda g: (False, "x"), required=True)
        soft = ValidationGate("soft", lambda g: (False, "x"), required=False)
        assert hard.required is True
        assert soft.required is False


class TestHarnessAdapter:
    def test_create_pythagorean(self):
        m = ConstructManifest(
            name="test",
            breeder_type="pythagorean",
            goal="g",
            population_size=10,
        )
        adapter = HarnessAdapter(m)
        # Don't actually run, just verify creation
        assert adapter.manifest == m
        assert len(adapter.gates) == 0

    def test_create_spectral(self):
        m = ConstructManifest(
            name="test",
            breeder_type="spectral",
            goal="g",
            population_size=10,
        )
        adapter = HarnessAdapter(m)
        assert adapter.manifest.breeder_type == "spectral"

    def test_create_adversarial(self):
        m = ConstructManifest(
            name="test",
            breeder_type="adversarial",
            goal="g",
            population_size=20,
        )
        adapter = HarnessAdapter(m)
        assert adapter.manifest.breeder_type == "adversarial"

    def test_add_gate(self):
        m = ConstructManifest(name="t", breeder_type="b", goal="g")
        adapter = HarnessAdapter(m)
        gate = ValidationGate("test", lambda g: (True, "OK"))
        adapter.add_gate(gate)
        assert len(adapter.gates) == 1

    def test_run_breeding_pythagorean(self):
        m = ConstructManifest(
            name="test",
            breeder_type="pythagorean",
            goal="g",
            population_size=10,
            generations=3,
        )
        adapter = HarnessAdapter(m)
        
        def task_fn(phenotype):
            return float(np.sum(phenotype ** 2))
        
        events = list(adapter.run_breeding(task_fn, generations=3))
        assert len(events) == 3
        assert events[0].generation == 0
        assert events[2].generation == 2
        assert events[-1].best_fitness >= 0

    def test_run_breeding_spectral(self):
        m = ConstructManifest(
            name="test",
            breeder_type="spectral",
            goal="g",
            population_size=10,
            generations=3,
        )
        adapter = HarnessAdapter(m)
        
        def task_fn(phenotype):
            return float(np.sum(phenotype ** 2))
        
        events = list(adapter.run_breeding(task_fn, generations=3))
        assert len(events) == 3
        assert all(e.population_size > 0 for e in events)

    def test_run_with_gates(self):
        m = ConstructManifest(
            name="test",
            breeder_type="pythagorean",
            goal="g",
            population_size=10,
            generations=2,
            validation_gates=True,
        )
        adapter = HarnessAdapter(m)
        adapter.add_gate(ValidationGate("pass", lambda g: (True, "OK")))
        adapter.add_gate(ValidationGate("fail", lambda g: (False, "FAIL")))
        
        def task_fn(phenotype):
            return float(np.sum(phenotype ** 2))
        
        events = list(adapter.run_breeding(task_fn, generations=2))
        assert len(events) == 2
        # Should track gate results
        assert events[0].flux_passed == 1  # One gate passes
        assert events[0].flux_failed == 1  # One gate fails

    def test_get_best(self):
        m = ConstructManifest(
            name="test",
            breeder_type="pythagorean",
            goal="g",
            population_size=10,
            generations=2,
        )
        adapter = HarnessAdapter(m)
        
        def task_fn(phenotype):
            return float(np.sum(phenotype ** 2))
        
        list(adapter.run_breeding(task_fn, generations=2))
        best, fitness = adapter.get_best()
        assert fitness >= 0

    def test_export_manifest(self):
        m = ConstructManifest(
            name="test",
            breeder_type="pythagorean",
            goal="g",
            population_size=10,
            generations=2,
        )
        adapter = HarnessAdapter(m)
        
        def task_fn(phenotype):
            return float(np.sum(phenotype ** 2))
        
        list(adapter.run_breeding(task_fn, generations=2))
        exported = adapter.export_manifest()
        d = json.loads(exported)
        assert "results" in d
        assert d["results"]["generations_completed"] == 2


class TestBuildCoordinator:
    def test_single_node(self):
        m = ConstructManifest(name="t", breeder_type="b", goal="g")
        coord = BuildCoordinator(m, node_id="node-0", total_nodes=1)
        
        event = BreedingEvent(
            generation=0,
            best_fitness=1.0,
            mean_fitness=0.5,
            population_size=10,
            elapsed_seconds=1.0,
        )
        result = coord.propose_generation(event)
        assert result is True  # Single node always agrees

    def test_multi_node(self):
        m = ConstructManifest(name="t", breeder_type="b", goal="g")
        coord = BuildCoordinator(m, node_id="node-0", total_nodes=4)
        assert coord._consensus is not None

    def test_node_config(self):
        coord = BuildCoordinator(
            ConstructManifest(name="t", breeder_type="b", goal="g"),
            node_id="test-node",
            total_nodes=7,
        )
        assert coord.node_id == "test-node"
        assert coord.total_nodes == 7


class TestProgressStreamer:
    def test_stream(self):
        m = ConstructManifest(
            name="test",
            breeder_type="pythagorean",
            goal="g",
            population_size=10,
            generations=3,
        )
        adapter = HarnessAdapter(m)
        streamer = ProgressStreamer(adapter)
        
        received = []
        def callback(event):
            received.append(event)
        
        streamer.subscribe(callback)
        
        def task_fn(phenotype):
            return float(np.sum(phenotype ** 2))
        
        events = list(streamer.stream(task_fn, generations=3))
        assert len(events) == 3
        assert len(received) == 3
        assert received[0].generation == 0

    def test_multiple_listeners(self):
        m = ConstructManifest(
            name="test",
            breeder_type="pythagorean",
            goal="g",
            population_size=10,
            generations=2,
        )
        adapter = HarnessAdapter(m)
        streamer = ProgressStreamer(adapter)
        
        received1 = []
        received2 = []
        streamer.subscribe(lambda e: received1.append(e))
        streamer.subscribe(lambda e: received2.append(e))
        
        def task_fn(phenotype):
            return float(np.sum(phenotype ** 2))
        
        list(streamer.stream(task_fn, generations=2))
        assert len(received1) == 2
        assert len(received2) == 2


class TestBuiltinGates:
    def test_exact_arithmetic_gate_pythagorean(self):
        from swarm.pythagorean_evolution import PythagoreanGenome, PythagoreanTriple
        genome = PythagoreanGenome(triples=[PythagoreanTriple(3, 4, 5)])
        passed, msg = exact_arithmetic_gate(genome)
        assert passed is True

    def test_exact_arithmetic_gate_other(self):
        passed, msg = exact_arithmetic_gate("not a genome")
        assert passed is False

    def test_spectral_real_gate(self):
        from swarm.spectral_breeding import SpectralGenome
        genome = SpectralGenome.random(32)
        passed, msg = spectral_real_gate(genome)
        assert passed is True
        assert "Real phenotype" in msg

    def test_spectral_real_gate_non_spectral(self):
        passed, msg = spectral_real_gate("not spectral")
        assert passed is True  # Passes through

    def test_robustness_gate_pass(self):
        genome = Mock()
        genome.robustness = 0.8
        passed, msg = robustness_gate(genome)
        assert passed is True

    def test_robustness_gate_fail(self):
        genome = Mock()
        genome.robustness = 0.3
        passed, msg = robustness_gate(genome)
        assert passed is False


class TestIntegration:
    def test_full_harness_workflow(self):
        """End-to-end: manifest → adapter → breeding → gates → export."""
        # 1. Define construct
        manifest = ConstructManifest(
            name="robust-solver",
            breeder_type="pythagorean",
            goal="Evolve robust solver",
            population_size=15,
            generations=5,
            qd_dimensions=[(3, 4, 5)],
            validation_gates=True,
        )
        
        # 2. Create adapter with gates
        adapter = HarnessAdapter(manifest)
        adapter.add_gate(ValidationGate("always_pass", lambda g: (True, "OK")))
        
        # 3. Create coordinator (simulating multi-node)
        coord = BuildCoordinator(manifest, node_id="node-0", total_nodes=1)
        
        # 4. Run breeding
        def task_fn(phenotype):
            return float(np.sum(phenotype ** 2))
        
        events = []
        for event in adapter.run_breeding(task_fn, generations=5):
            events.append(event)
            coord.propose_generation(event)
        
        # 5. Verify
        assert len(events) == 5
        assert events[-1].generation == 4
        assert events[-1].flux_passed == 1
        
        # 6. Export
        exported = adapter.export_manifest()
        result = json.loads(exported)
        assert result["results"]["generations_completed"] == 5
        
        # 7. Get best
        best, fitness = adapter.get_best()
        assert fitness >= 0
