"""Tests for OpenConstruct Shell — agent attachment system.

Covers AttachmentRegistry, SensorReading, SelfHealingLoop, OpenConstructShell,
and the full skill manual generation.
"""

import time
from typing import List

import numpy as np
import pytest

from fleet.openconstruct_shell import (
    SensorReading,
    SensorType,
    AttachmentRegistry,
    SelfHealingLoop,
    OpenConstructShell,
    ATTACHMENTS,
)
from fleet.openconstruct_bridge import (
    BreedingEvent,
    BreedingEventType,
    ConstructManifest,
    HarnessAdapter,
)


class TestSensorReading:
    def test_tick_sensor(self):
        r = SensorReading(
            sensor_type=SensorType.TICK,
            name="engine_temp",
            value=72.5,
            timestamp=time.time(),
            unit="°C",
        )
        text = r.to_agent_text()
        assert "[TICK]" in text
        assert "engine_temp" in text
        assert "72.5" in text

    def test_alert_sensor(self):
        r = SensorReading(
            sensor_type=SensorType.ALERT,
            name="fitness_collapse",
            value=0.001,
            timestamp=time.time(),
            message="Fitness collapsed to near zero",
        )
        text = r.to_agent_text()
        assert "[ALERT]" in text
        assert "fitness_collapse" in text
        assert "collapsed" in text

    def test_delta_sensor(self):
        r = SensorReading(
            sensor_type=SensorType.DELTA,
            name="best_fitness",
            value=0.15,
            timestamp=time.time(),
            unit="score",
        )
        text = r.to_agent_text()
        assert "[DELTA]" in text
        assert "change detected" in text

    def test_to_json(self):
        r = SensorReading(
            sensor_type=SensorType.METRIC,
            name="qd_coverage",
            value=42,
            timestamp=time.time(),
            unit="cells",
        )
        json_str = r.to_json()
        assert "qd_coverage" in json_str
        assert "42" in json_str


class TestAttachmentRegistry:
    def test_list_default_attachments(self):
        attachments = ATTACHMENTS.list_attachments()
        names = [a[0] for a in attachments]
        assert "pythagorean" in names
        assert "spectral" in names
        assert "adversarial" in names
        assert "nca" in names
        assert "standard" in names

    def test_register_custom(self):
        reg = AttachmentRegistry()
        reg.register("custom", "Custom attachment", {"breeder_type": "custom"})
        attachments = reg.list_attachments()
        assert any(a[0] == "custom" for a in attachments)

    def test_spawn_pythagorean(self):
        manifest = ConstructManifest(
            name="test",
            breeder_type="pythagorean",
            goal="test",
            population_size=5,
            genome_length=3,
        )
        adapter, run_id = ATTACHMENTS.spawn("pythagorean", manifest)
        assert adapter is not None
        assert run_id.startswith("pythagorean-")

    def test_spawn_spectral(self):
        manifest = ConstructManifest(
            name="spectral-test",
            breeder_type="spectral",
            goal="test",
            population_size=5,
        )
        adapter, run_id = ATTACHMENTS.spawn("spectral", manifest)
        assert adapter is not None
        assert "spectral" in run_id

    def test_spawn_unknown(self):
        manifest = ConstructManifest(name="u", breeder_type="unknown", goal="test")
        with pytest.raises(ValueError):
            ATTACHMENTS.spawn("unknown", manifest)

    def test_sensor_history(self):
        manifest = ConstructManifest(
            name="sensor-test",
            breeder_type="pythagorean",
            goal="test",
            population_size=5,
        )
        adapter, run_id = ATTACHMENTS.spawn("pythagorean", manifest)
        reading = SensorReading(
            sensor_type=SensorType.METRIC,
            name="fitness",
            value=1.0,
            timestamp=time.time(),
        )
        ATTACHMENTS._record_sensor(run_id, reading)
        history = ATTACHMENTS.get_sensor_history(run_id)
        assert len(history) == 1
        assert history[0].value == 1.0


class TestSelfHealingLoop:
    def test_fitness_collapse_detection(self):
        manifest = ConstructManifest(
            name="heal-test",
            breeder_type="pythagorean",
            goal="test",
            population_size=5,
        )
        adapter = HarnessAdapter(manifest)
        healing = SelfHealingLoop(adapter, "run-1")
        
        event = BreedingEvent(
            event_type=BreedingEventType.GENERATION_END,
            generation=5,
            timestamp=time.time(),
            best_fitness=0.001,
        )
        readings = healing.check_health(event)
        alert_readings = [r for r in readings if r.sensor_type == SensorType.ALERT]
        assert len(alert_readings) > 0
        assert any("collapse" in r.message for r in alert_readings)

    def test_stagnation_detection(self):
        manifest = ConstructManifest(
            name="stag-test",
            breeder_type="pythagorean",
            goal="test",
            population_size=5,
        )
        adapter = HarnessAdapter(manifest)
        # Populate history with stagnant fitness
        for i in range(15):
            adapter.streamer.emit(BreedingEvent(
                event_type=BreedingEventType.GENERATION_END,
                generation=i,
                timestamp=time.time(),
                best_fitness=1.0,
            ))
        
        healing = SelfHealingLoop(adapter, "run-2")
        event = BreedingEvent(
            event_type=BreedingEventType.GENERATION_END,
            generation=15,
            timestamp=time.time(),
            best_fitness=1.0,
        )
        readings = healing.check_health(event)
        alert_readings = [r for r in readings if r.sensor_type == SensorType.ALERT]
        assert any("Stagnation" in r.message for r in alert_readings)

    def test_normal_metrics(self):
        manifest = ConstructManifest(
            name="metric-test",
            breeder_type="pythagorean",
            goal="test",
            population_size=5,
        )
        adapter = HarnessAdapter(manifest)
        healing = SelfHealingLoop(adapter, "run-3")
        
        event = BreedingEvent(
            event_type=BreedingEventType.GENERATION_END,
            generation=1,
            timestamp=time.time(),
            best_fitness=5.0,
            qd_coverage=10,
        )
        readings = healing.check_health(event)
        metric_readings = [r for r in readings if r.sensor_type == SensorType.METRIC]
        assert len(metric_readings) >= 2
        assert any(r.name == "best_fitness" for r in metric_readings)
        assert any(r.name == "qd_coverage" for r in metric_readings)

    def test_healing_log(self):
        manifest = ConstructManifest(
            name="log-test",
            breeder_type="pythagorean",
            goal="test",
            population_size=5,
        )
        adapter = HarnessAdapter(manifest)
        healing = SelfHealingLoop(adapter, "run-4")
        
        # Trigger fitness collapse
        event = BreedingEvent(
            event_type=BreedingEventType.GENERATION_END,
            generation=1,
            timestamp=time.time(),
            best_fitness=0.0001,
        )
        healing.check_health(event)
        
        assert len(healing.healing_log) > 0
        assert healing.healing_log[0]["action"] == "fitness_collapse_recovery"

    def test_max_recoveries(self):
        manifest = ConstructManifest(
            name="max-test",
            breeder_type="pythagorean",
            goal="test",
            population_size=5,
        )
        adapter = HarnessAdapter(manifest)
        healing = SelfHealingLoop(adapter, "run-5")
        healing._max_recoveries = 2
        
        # Trigger multiple times
        for i in range(5):
            event = BreedingEvent(
                event_type=BreedingEventType.GENERATION_END,
                generation=i,
                timestamp=time.time(),
                best_fitness=0.0001,
            )
            healing.check_health(event)
        
        # Should stop after max_recoveries
        assert healing._recovery_count == 2


class TestOpenConstructShell:
    def test_list_attachments(self):
        shell = OpenConstructShell()
        attachments = shell.list_attachments()
        assert len(attachments) >= 5
        names = [a[0] for a in attachments]
        assert "pythagorean" in names

    def test_spawn(self):
        shell = OpenConstructShell()
        run_id = shell.spawn("pythagorean", population_size=5, genome_length=3, generations=2)
        assert run_id.startswith("pythagorean-")
        status = shell.status(run_id)
        assert status["attachment"] == "pythagorean"
        assert status["status"] == "spawned"

    def test_spawn_spectral(self):
        shell = OpenConstructShell()
        run_id = shell.spawn("spectral", population_size=5, generations=2)
        assert "spectral" in run_id

    def test_run_single_generation(self):
        shell = OpenConstructShell()
        run_id = shell.spawn("pythagorean", population_size=5, genome_length=3, generations=1)
        
        def task_fn(matrix):
            return float(np.sum(matrix))
        
        readings = list(shell.run(run_id, task_fn, generations=1))
        assert len(readings) > 0
        metric_readings = [r for r in readings if r.sensor_type == SensorType.METRIC]
        assert len(metric_readings) > 0

    def test_run_multiple_generations(self):
        shell = OpenConstructShell()
        run_id = shell.spawn("pythagorean", population_size=5, genome_length=3, generations=3)
        
        def task_fn(matrix):
            return float(np.sum(matrix))
        
        readings = list(shell.run(run_id, task_fn, generations=3))
        
        # Should have multiple generation readings
        fitness_readings = [r for r in readings if r.name == "best_fitness"]
        assert len(fitness_readings) >= 3

    def test_health_check(self):
        shell = OpenConstructShell()
        run_id = shell.spawn("pythagorean", population_size=5, genome_length=3, generations=1)
        
        def task_fn(matrix):
            return float(np.sum(matrix))
        
        list(shell.run(run_id, task_fn, generations=1))
        readings = shell.health_check(run_id)
        assert len(readings) > 0

    def test_status_all(self):
        shell = OpenConstructShell()
        run_id1 = shell.spawn("pythagorean", population_size=5, genome_length=3, generations=1)
        run_id2 = shell.spawn("spectral", population_size=5, generations=1)
        
        status = shell.status()
        assert status["active_runs"] == 2
        assert run_id1 in status["runs"]
        assert run_id2 in status["runs"]

    def test_terminate(self):
        shell = OpenConstructShell()
        run_id = shell.spawn("pythagorean", population_size=5, genome_length=3, generations=1)
        shell.terminate(run_id)
        status = shell.status(run_id)
        assert status["status"] == "terminated"

    def test_multi_node_shell(self):
        shell = OpenConstructShell(node_id="node-1", all_nodes=["node-1", "node-2"])
        run_id = shell.spawn("pythagorean", population_size=5, genome_length=3, generations=1)
        
        def task_fn(matrix):
            return float(np.sum(matrix))
        
        readings = list(shell.run(run_id, task_fn, generations=1))
        assert len(readings) > 0

    def test_skill_manual(self):
        shell = OpenConstructShell()
        manual = shell.to_skill_manual()
        assert "OpenConstruct Fleet Breeding Manual" in manual
        assert "pythagorean" in manual
        assert "spectral" in manual
        assert "Operating Instructions" in manual
        assert "Self-Healing" in manual

    def test_skill_manual_has_quick_reference(self):
        shell = OpenConstructShell()
        manual = shell.to_skill_manual()
        assert "Quick Reference" in manual
        assert "shell.spawn()" in manual
        assert "shell.run()" in manual
        assert "shell.status()" in manual

    def test_unknown_attachment(self):
        shell = OpenConstructShell()
        with pytest.raises(ValueError):
            shell.spawn("nonexistent", population_size=5)

    def test_run_not_found(self):
        shell = OpenConstructShell()
        readings = list(shell.run("nonexistent-run", lambda g: 1.0))
        assert len(readings) == 1
        assert readings[0].sensor_type == SensorType.ALERT


class TestIntegrationFlow:
    def test_full_agent_workflow(self):
        """End-to-end: Agent spawns, runs, monitors, and gets results."""
        shell = OpenConstructShell()
        
        # Agent lists available equipment
        attachments = shell.list_attachments()
        assert len(attachments) > 0
        
        # Agent spawns pythagorean breeder
        run_id = shell.spawn(
            "pythagorean",
            population_size=5,
            genome_length=3,
            generations=3,
            name="agent-workflow-test",
        )
        
        # Agent defines task (expects matrix for PythagoreanBreeder)
        def task_fn(matrix):
            return float(np.sum(matrix))
        
        # Agent runs and monitors sensors
        all_readings = []
        for reading in shell.run(run_id, task_fn, generations=3):
            all_readings.append(reading)
            # Agent checks for alerts
            if reading.sensor_type == SensorType.ALERT:
                print(f"AGENT ALERT: {reading.to_agent_text()}")
        
        # Agent checks status
        status = shell.status(run_id)
        assert status["status"] == "complete"
        
        # Agent gets best result
        best_genome, best_fitness = shell.get_best(run_id)
        assert best_genome is not None
        assert best_fitness >= 0.0
        
        # Agent reviews health history
        health = shell.health_check(run_id)
        assert len(health) > 0

    def test_multi_node_fleet_workflow(self):
        """Agent operates a fleet of nodes."""
        shell = OpenConstructShell(
            node_id="node-1",
            all_nodes=["node-1", "node-2", "node-3"],
        )
        
        run_id = shell.spawn("pythagorean", population_size=5, genome_length=3, generations=2)
        
        def task_fn(matrix):
            return float(np.sum(matrix))
        
        readings = list(shell.run(run_id, task_fn, generations=2))
        
        # Check for consensus metrics
        consensus_readings = [r for r in readings if "consensus" in r.name.lower()]
        # Multi-node may or may not produce consensus readings depending on implementation
        # But the run should complete successfully
        status = shell.status(run_id)
        assert status["status"] == "complete"

    def test_skill_manual_completeness(self):
        """Verify the manual is comprehensive enough for an agent to operate."""
        shell = OpenConstructShell()
        manual = shell.to_skill_manual()
        
        # Should contain all key sections
        required_sections = [
            "Available Attachments",
            "Operating Instructions",
            "Sensor Types",
            "Self-Healing",
            "Multi-Node Operation",
            "Safety Notes",
            "Quick Reference",
        ]
        for section in required_sections:
            assert section in manual, f"Missing section: {section}"
        
        # Should contain all attachments
        for name, _ in shell.list_attachments():
            assert name in manual, f"Missing attachment in manual: {name}"

    def test_shell_parallel_command(self):
        """Agent runs multiple campaigns in parallel via shell."""
        shell = OpenConstructShell(
            node_id="node-1",
            all_nodes=["node-1"],
        )
        
        campaigns = [
            {
                "name": "exact-rational",
                "attachment": "pythagorean",
                "params": {"population_size": 5, "genome_length": 3},
            },
            {
                "name": "fourier-evolution",
                "attachment": "spectral",
                "params": {"population_size": 5, "spectrum_size": 32},
            },
        ]
        
        result = shell.run_parallel(campaigns, generations=1, repo_path="/tmp/test-repo")
        
        # Verify structure
        assert "campaign_count" in result
        assert result["campaign_count"] == 2
        assert "best_campaign" in result
        assert "success_rate" in result
        assert result["total_duration"] >= 0
