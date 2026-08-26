"""Tests for OpenConstruct Bridge — harness integration system.

Covers ConstructManifest, BreederFactory, HarnessAdapter, BuildCoordinator,
ProgressStreamer, ValidationGate, and all integration points.
"""

import json
import time
from typing import List

import numpy as np
import pytest

from fleet.openconstruct_bridge import (
    BreedingEvent,
    BreedingEventType,
    ConstructManifest,
    BreederFactory,
    HarnessAdapter,
    BuildCoordinator,
    ProgressStreamer,
    ValidationGate,
    GATE_REGISTRY,
    exact_arithmetic_gate,
    holonomic_consistency_gate,
    spectral_real_gate,
    robustness_gate,
)
from swarm.pythagorean_evolution import PythagoreanGenome, PythagoreanTriple
from swarm.spectral_breeding import SpectralGenome


class TestConstructManifest:
    def test_basic_manifest(self):
        m = ConstructManifest(
            name="test-solver",
            breeder_type="pythagorean",
            goal="Test goal",
            population_size=10,
            generations=5,
        )
        assert m.name == "test-solver"
        assert m.breeder_type == "pythagorean"
        assert m.population_size == 10

    def test_manifest_defaults(self):
        m = ConstructManifest(name="default-test", breeder_type="standard", goal="test")
        assert m.population_size == 50
        assert m.generations == 100
        assert m.mutation_rate == 0.1
        assert m.crossover_rate == 0.5

    def test_to_json(self):
        m = ConstructManifest(
            name="json-test",
            breeder_type="spectral",
            goal="test json",
            qd_dimensions=[(3, 4, 5)],
        )
        json_str = m.to_json()
        data = json.loads(json_str)
        assert data["name"] == "json-test"
        assert data["breeder_type"] == "spectral"
        assert data["qd_dimensions"] == [[3, 4, 5]]

    def test_from_json(self):
        m = ConstructManifest(name="roundtrip", breeder_type="adversarial", goal="test")
        json_str = m.to_json()
        m2 = ConstructManifest.from_json(json_str)
        assert m2.name == m.name
        assert m2.breeder_type == m.breeder_type
        assert m2.population_size == m.population_size

    def test_manifest_with_resources(self):
        m = ConstructManifest(
            name="resource-test",
            breeder_type="standard",
            goal="test",
            resources={"nodes": 4, "gpu": True},
        )
        assert m.resources["nodes"] == 4


class TestBreederFactory:
    def test_pythagorean_creation(self):
        m = ConstructManifest(
            name="p", breeder_type="pythagorean", goal="g", population_size=5
        )
        breeder = BreederFactory.create(m)
        assert breeder is not None
        assert hasattr(breeder, "population_size")

    def test_spectral_creation(self):
        m = ConstructManifest(
            name="s", breeder_type="spectral", goal="g", population_size=5
        )
        breeder = BreederFactory.create(m)
        assert breeder is not None
        assert hasattr(breeder, "spectrum_size")

    def test_adversarial_creation(self):
        m = ConstructManifest(
            name="a", breeder_type="adversarial", goal="g", population_size=5
        )
        breeder = BreederFactory.create(m)
        assert breeder is not None
        assert hasattr(breeder, "tester_pop_size")

    def test_standard_creation(self):
        m = ConstructManifest(
            name="st", breeder_type="standard", goal="g", population_size=5
        )
        breeder = BreederFactory.create(m)
        assert breeder is not None

    def test_unknown_breeder_type(self):
        m = ConstructManifest(name="u", breeder_type="unknown", goal="g")
        with pytest.raises(ValueError):
            BreederFactory.create(m)

    def test_register_custom(self):
        def custom_builder(manifest):
            return {"custom": True, "name": manifest.name}

        BreederFactory.register("custom", custom_builder)
        m = ConstructManifest(name="c", breeder_type="custom", goal="g")
        result = BreederFactory.create(m)
        assert result["custom"] is True


class TestValidationGate:
    def test_exact_arithmetic_pass(self):
        genome = PythagoreanGenome([PythagoreanTriple(3, 4, 5)])
        ok, msg = exact_arithmetic_gate(genome)
        assert ok is True
        assert "verified" in msg

    def test_exact_arithmetic_fail(self):
        # Create a genome and manually invalidate one triple
        genome = PythagoreanGenome([PythagoreanTriple(3, 4, 5)])

        # Manually create an invalid triple-like object
        class FakeTriple:
            a = 3
            b = 4
            c = 6

        genome.triples.append(FakeTriple())
        ok, msg = exact_arithmetic_gate(genome)
        assert ok is False
        assert "Invalid triple" in msg

    def test_holonomic_gate_skip(self):
        # When no constraint bridge, should pass with skip message
        genome = PythagoreanGenome([PythagoreanTriple(3, 4, 5)])
        ok, msg = holonomic_consistency_gate(genome)
        assert ok is True
        assert "skipped" in msg

    def test_spectral_real_gate(self):
        # Create a valid spectral genome with Hermitian symmetry
        spectrum = np.zeros(32, dtype=complex)
        spectrum[0] = 1.0
        for i in range(1, 16):
            spectrum[i] = complex(0.5, 0.3)
            spectrum[32 - i] = complex(0.5, -0.3)
        genome = SpectralGenome(spectrum)
        ok, msg = spectral_real_gate(genome)
        assert ok is True

    def test_spectral_real_gate_fail(self):
        # Violate Hermitian symmetry
        spectrum = np.zeros(32, dtype=complex)
        spectrum[1] = 1.0
        spectrum[31] = 2.0  # Should be conjugate
        genome = SpectralGenome(spectrum)
        ok, msg = spectral_real_gate(genome)
        assert ok is False

    def test_robustness_gate(self):
        genome = PythagoreanGenome([PythagoreanTriple(3, 4, 5)])
        ok, msg = robustness_gate(genome)
        assert ok is True

    def test_gate_registry(self):
        assert "exact_arithmetic" in GATE_REGISTRY
        assert "holonomic_consistency" in GATE_REGISTRY
        assert "spectral_real" in GATE_REGISTRY
        assert "robustness" in GATE_REGISTRY

    def test_hard_vs_soft_gate(self):
        hard_gate = ValidationGate("hard", lambda g: (False, "fail"), hard=True)
        soft_gate = ValidationGate("soft", lambda g: (False, "warn"), hard=False)
        assert hard_gate.hard is True
        assert soft_gate.hard is False


class TestProgressStreamer:
    def test_emit_and_history(self):
        streamer = ProgressStreamer()
        event = BreedingEvent(
            event_type=BreedingEventType.GENERATION_START,
            generation=0,
            timestamp=time.time(),
        )
        streamer.emit(event)
        assert len(streamer.history) == 1

    def test_callback(self):
        received: List[BreedingEvent] = []

        def callback(e):
            received.append(e)

        streamer = ProgressStreamer([callback])
        event = BreedingEvent(
            event_type=BreedingEventType.GENERATION_END,
            generation=1,
            timestamp=time.time(),
        )
        streamer.emit(event)
        assert len(received) == 1
        assert received[0].generation == 1

    def test_sse_format(self):
        streamer = ProgressStreamer()
        event = BreedingEvent(
            event_type=BreedingEventType.GENERATION_END,
            generation=2,
            timestamp=time.time(),
            best_fitness=1.5,
        )
        sse = streamer.sse_format(event)
        assert sse.startswith("data: ")
        assert "generation" in sse
        assert "best_fitness" in sse

    def test_filter_history(self):
        streamer = ProgressStreamer()
        for i in range(3):
            streamer.emit(
                BreedingEvent(
                    event_type=BreedingEventType.GENERATION_START,
                    generation=i,
                    timestamp=time.time(),
                )
            )
        streamer.emit(
            BreedingEvent(
                event_type=BreedingEventType.BREED_COMPLETE,
                generation=3,
                timestamp=time.time(),
            )
        )
        start_events = streamer.get_history(BreedingEventType.GENERATION_START)
        assert len(start_events) == 3


class TestBuildCoordinator:
    def test_single_node(self):
        m = ConstructManifest(
            name="coord-test",
            breeder_type="pythagorean",
            goal="test",
            population_size=5,
        )
        coord = BuildCoordinator(
            node_id="node-1",
            all_nodes=["node-1"],
            manifest=m,
        )
        assert coord.node_id == "node-1"
        assert coord.local_breeder is not None

    def test_multi_node_setup(self):
        m = ConstructManifest(
            name="multi-test",
            breeder_type="pythagorean",
            goal="test",
            population_size=5,
        )
        coord = BuildCoordinator(
            node_id="node-1",
            all_nodes=["node-1", "node-2", "node-3"],
            manifest=m,
        )
        assert len(coord.all_nodes) == 3
        assert coord.consensus is not None

    def test_propose_generation(self):
        m = ConstructManifest(
            name="prop-test",
            breeder_type="pythagorean",
            goal="test",
            population_size=5,
        )
        coord = BuildCoordinator(
            node_id="node-1",
            all_nodes=["node-1"],
            manifest=m,
        )
        coord.local_breeder.initialize()
        parents = coord.local_breeder.population
        result = coord.propose_generation(0, parents)
        assert "generation" in result
        assert "nodes_agreed" in result

    def test_coordinated_breeding(self):
        m = ConstructManifest(
            name="breed-test",
            breeder_type="pythagorean",
            goal="test",
            population_size=5,
            genome_length=3,
        )
        coord = BuildCoordinator(
            node_id="node-1",
            all_nodes=["node-1"],
            manifest=m,
        )

        def task_fn(matrix):
            return float(np.sum(matrix))

        events = list(coord.run_coordinated_breeding(task_fn, generations=2))
        assert len(events) > 0
        # Should have generation starts, ends, and complete
        assert any(e.event_type == BreedingEventType.GENERATION_START for e in events)
        assert any(e.event_type == BreedingEventType.GENERATION_END for e in events)
        assert any(e.event_type == BreedingEventType.BREED_COMPLETE for e in events)


class TestHarnessAdapter:
    def test_single_node_breeding(self):
        m = ConstructManifest(
            name="adapter-test",
            breeder_type="pythagorean",
            goal="test",
            population_size=5,
            genome_length=3,
            generations=2,
        )
        adapter = HarnessAdapter(m)

        def task_fn(matrix):
            return float(np.sum(matrix))

        events = list(adapter.run_breeding(task_fn, generations=2))
        assert len(events) > 0
        # Check that we got generation events
        assert any(e.event_type == BreedingEventType.GENERATION_START for e in events)
        assert any(e.event_type == BreedingEventType.BREED_COMPLETE for e in events)

    def test_adapter_with_gates(self):
        m = ConstructManifest(
            name="gate-test",
            breeder_type="pythagorean",
            goal="test",
            population_size=5,
            genome_length=3,
            generations=2,
            constraints=["exact_arithmetic"],
        )
        adapter = HarnessAdapter(m)
        assert len(adapter.gates) == 1
        assert adapter.gates[0].name == "exact_arithmetic"

    def test_adapter_with_streamer(self):
        m = ConstructManifest(
            name="stream-test",
            breeder_type="pythagorean",
            goal="test",
            population_size=5,
            genome_length=3,
            generations=2,
        )
        adapter = HarnessAdapter(m)

        def task_fn(matrix):
            return float(np.sum(matrix))

        events = list(adapter.run_breeding(task_fn, generations=2))
        assert len(adapter.streamer.history) > 0
        assert len(events) == len(adapter.streamer.history)

    def test_export_manifest(self):
        m = ConstructManifest(
            name="export-test",
            breeder_type="pythagorean",
            goal="test",
            population_size=5,
        )
        adapter = HarnessAdapter(m)

        # Run a quick breeding to populate history
        def task_fn(matrix):
            return float(np.sum(matrix))

        list(adapter.run_breeding(task_fn, generations=1))

        exported = adapter.export_manifest()
        data = json.loads(exported)
        assert data["name"] == "export-test"
        assert "results" in data
        assert data["results"]["history_count"] > 0

    def test_connect_disconnect(self):
        m = ConstructManifest(name="conn", breeder_type="standard", goal="test")
        adapter = HarnessAdapter(m)
        # These are currently placeholders
        adapter.connect("http://example.com")
        adapter.disconnect()

    def test_add_custom_gate(self):
        m = ConstructManifest(name="custom-gate", breeder_type="standard", goal="test")
        adapter = HarnessAdapter(m)
        custom_gate = ValidationGate("custom", lambda g: (True, "ok"), hard=False)
        adapter.add_gate(custom_gate)
        assert len(adapter.gates) == 1

    def test_manifest_with_all_breeder_types(self):
        for btype in ["pythagorean", "spectral", "adversarial", "standard"]:
            m = ConstructManifest(
                name=f"test-{btype}",
                breeder_type=btype,
                goal="test",
                population_size=3,
                generations=1,
            )
            adapter = HarnessAdapter(m)
            if btype == "adversarial":
                # AdversarialArena takes 2 args: solver_vector, tester_vector
                def task_fn(solver_vec, tester_vec):
                    return 1.0, 0.5
            else:

                def task_fn(genome_or_matrix):
                    return 1.0

            # Standard breeder (BreederDaemonV2) has a different API and needs
            # RoomGrid + ThermalBudget; skip runtime for it, just verify creation
            if btype == "standard":
                # Just verify the adapter was created successfully
                assert adapter is not None
                continue
            events = list(adapter.run_breeding(task_fn, generations=1))
            assert len(events) > 0


class TestIntegrationFlow:
    def test_full_pipeline(self):
        """End-to-end test: manifest → adapter → breeding → events → export."""
        m = ConstructManifest(
            name="full-pipeline",
            breeder_type="pythagorean",
            goal="Maximize triple count",
            population_size=5,
            genome_length=3,
            generations=3,
            constraints=["exact_arithmetic"],
        )
        adapter = HarnessAdapter(m)

        def task_fn(matrix):
            return float(np.sum(matrix))

        # Run breeding and collect all events
        events = list(adapter.run_breeding(task_fn, generations=3))

        # Verify event flow
        start_events = [
            e for e in events if e.event_type == BreedingEventType.GENERATION_START
        ]
        end_events = [
            e for e in events if e.event_type == BreedingEventType.GENERATION_END
        ]
        complete_events = [
            e for e in events if e.event_type == BreedingEventType.BREED_COMPLETE
        ]

        assert len(start_events) == 3
        assert len(end_events) == 3
        assert len(complete_events) == 1

        # Verify fitness progression
        best_fitnesses = [
            e.best_fitness for e in end_events if e.best_fitness is not None
        ]
        assert len(best_fitnesses) == 3
        assert all(f >= 0 for f in best_fitnesses)

        # Export and verify
        exported = adapter.export_manifest()
        data = json.loads(exported)
        assert data["name"] == "full-pipeline"
        assert data["results"]["history_count"] == len(events)

    def test_multi_node_pipeline(self):
        """End-to-end with multi-node BFT consensus."""
        m = ConstructManifest(
            name="multi-pipeline",
            breeder_type="pythagorean",
            goal="test multi",
            population_size=5,
            genome_length=3,
            generations=2,
        )
        coord = BuildCoordinator(
            node_id="node-1",
            all_nodes=["node-1", "node-2"],
            manifest=m,
        )
        adapter = HarnessAdapter(m, coordinator=coord)

        def task_fn(matrix):
            return float(np.sum(matrix))

        events = list(adapter.run_breeding(task_fn, generations=2))
        assert len(events) > 0
        # Should have consensus information
        end_events = [
            e for e in events if e.event_type == BreedingEventType.GENERATION_END
        ]
        assert len(end_events) == 2
        assert end_events[0].nodes_agreed is not None
        assert end_events[0].total_nodes is not None
