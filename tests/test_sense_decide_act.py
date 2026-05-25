"""Tests for the Sense→Decide→Act framework.

Covers core abstractions, SDALoop orchestration, Policy engine,
built-in adapters, and thread-safety.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from fleet.sense_decide_act import (
    Act,
    ActResult,
    BreedCoordinationAct,
    BreedCoordinationSense,
    Decision,
    Decide,
    FluxPresetDecide,
    GatewayDispatchDecide,
    HebbianMeshAct,
    HebbianMeshSense,
    Observation,
    Policy,
    SDALoop,
    SDAPipeline,
    Sense,
    TrapSense,
)


# ═══════════════════════════════════════════════════════════════
# Minimal concrete implementations for testing
# ═══════════════════════════════════════════════════════════════

class _DummySense(Sense):
    def __init__(self, metrics: dict[str, Any] | None = None, severity: str = "info") -> None:
        self.metrics = metrics or {"value": 42}
        self.severity = severity
        self.call_order: int | None = None

    def observe(self) -> Observation:
        self.call_order = _call_counter.increment()
        return Observation(
            timestamp=time.time(),
            source="dummy",
            metrics=dict(self.metrics),
            severity_hint=self.severity,
        )


class _DummyDecide(Decide):
    def __init__(self, action_type: str = "noop", confidence: float = 1.0) -> None:
        self.action_type = action_type
        self.confidence = confidence
        self.last_observation: Observation | None = None
        self.call_order: int | None = None

    def evaluate(self, observation: Observation) -> Decision:
        self.call_order = _call_counter.increment()
        self.last_observation = observation
        return Decision(
            action_type=self.action_type,
            confidence=self.confidence,
            payload={"observed": observation.source},
            reasoning="dummy decision",
        )


class _DummyAct(Act):
    def __init__(self, success: bool = True, side_effects: list[str] | None = None) -> None:
        self.success = success
        self.side_effects = side_effects or ["dummy_effect"]
        self.last_decision: Decision | None = None
        self.call_order: int | None = None

    def execute(self, decision: Decision) -> ActResult:
        self.call_order = _call_counter.increment()
        self.last_decision = decision
        return ActResult(
            success=self.success,
            latency_ms=1.0,
            side_effects=list(self.side_effects),
            new_observations=[
                Observation(
                    timestamp=time.time(),
                    source="dummy_feedback",
                    metrics={"triggered_by": decision.action_type},
                )
            ],
        )


class _CallCounter:
    """Thread-safe monotonic counter for ordering assertions."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    def reset(self) -> None:
        with self._lock:
            self._value = 0

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


_call_counter = _CallCounter()


# ═══════════════════════════════════════════════════════════════
# Core abstraction tests
# ═══════════════════════════════════════════════════════════════

class TestSense:
    def test_observe_returns_observation(self):
        """Sense.observe() must return an Observation instance."""
        sense = _DummySense()
        obs = sense.observe()
        assert isinstance(obs, Observation)
        assert obs.source == "dummy"
        assert obs.metrics["value"] == 42

    def test_observation_severity_hint_passthrough(self):
        """Severity hint from sense must survive in the observation."""
        sense = _DummySense(severity="critical")
        obs = sense.observe()
        assert obs.severity_hint == "critical"


class TestDecide:
    def test_evaluate_returns_decision_with_correct_confidence(self):
        """Decide.evaluate() must return a Decision with the requested confidence."""
        decide = _DummyDecide(action_type="test", confidence=0.85)
        obs = Observation(timestamp=0.0, source="test", metrics={})
        decision = decide.evaluate(obs)
        assert isinstance(decision, Decision)
        assert decision.confidence == 0.85
        assert decision.action_type == "test"

    def test_decide_receives_observation(self):
        """Decide must receive the exact Observation produced by Sense."""
        decide = _DummyDecide()
        obs = Observation(timestamp=1.0, source="sensor_a", metrics={"temp": 99})
        decision = decide.evaluate(obs)
        assert decide.last_observation is obs
        assert decision.payload["observed"] == "sensor_a"


class TestAct:
    def test_execute_returns_act_result(self):
        """Act.execute() must return an ActResult instance."""
        act = _DummyAct()
        decision = Decision(action_type="poke", confidence=1.0, payload={})
        result = act.execute(decision)
        assert isinstance(result, ActResult)
        assert result.success is True
        assert result.latency_ms == 1.0

    def test_act_result_side_effects_tracking(self):
        """ActResult must faithfully report side effects."""
        act = _DummyAct(side_effects=["file_written", "cache_cleared"])
        decision = Decision(action_type="flush", confidence=1.0, payload={})
        result = act.execute(decision)
        assert result.side_effects == ["file_written", "cache_cleared"]

    def test_act_result_new_observations_generation(self):
        """ActResult may carry new observations for feedback loops."""
        act = _DummyAct()
        decision = Decision(action_type="scan", confidence=1.0, payload={})
        result = act.execute(decision)
        assert len(result.new_observations) == 1
        assert result.new_observations[0].source == "dummy_feedback"


# ═══════════════════════════════════════════════════════════════
# Policy engine tests
# ═══════════════════════════════════════════════════════════════

class TestPolicy:
    def test_rule_matching(self):
        """Policy must return the first matching rule's decision."""
        policy = Policy()
        policy.add_rule(
            condition=lambda obs: obs.metrics.get("temp", 0) > 80,
            action_type="cool",
            confidence=0.9,
            reasoning="temperature too high",
        )
        policy.add_rule(
            condition=lambda obs: obs.metrics.get("temp", 0) > 50,
            action_type="fan",
            confidence=0.6,
            reasoning="temperature elevated",
        )

        obs = Observation(timestamp=0.0, source="thermal", metrics={"temp": 85})
        decision = policy.evaluate(obs)
        assert decision.action_type == "cool"
        assert decision.confidence == 0.9

    def test_no_match_returns_noop(self):
        """If no rule matches, Policy returns a noop with confidence 1.0."""
        policy = Policy()
        policy.add_rule(
            condition=lambda obs: obs.metrics.get("temp", 0) > 100,
            action_type="panic",
            confidence=1.0,
            reasoning="meltdown",
        )
        obs = Observation(timestamp=0.0, source="thermal", metrics={"temp": 20})
        decision = policy.evaluate(obs)
        assert decision.action_type == "noop"
        assert decision.confidence == 1.0


# ═══════════════════════════════════════════════════════════════
# SDALoop orchestration tests
# ═══════════════════════════════════════════════════════════════

class TestSDALoop:
    def test_tick_runs_full_pipeline_in_correct_order(self):
        """SDALoop.tick() must call sense → decide → act in that order."""
        _call_counter.reset()
        sense = _DummySense()
        decide = _DummyDecide()
        act = _DummyAct()

        loop = SDALoop()
        loop.register(sense, decide, act, name="test_pipe")
        loop.tick()

        assert sense.call_order == 1
        assert decide.call_order == 2
        assert act.call_order == 3

    def test_pipeline_registration_and_retrieval(self):
        """Register a pipeline and get it back by name."""
        loop = SDALoop()
        sense = _DummySense()
        decide = _DummyDecide()
        act = _DummyAct()

        pipe = loop.register(sense, decide, act, name="alpha")
        assert pipe.name == "alpha"

        retrieved = loop.get_pipeline("alpha")
        assert retrieved is pipe

    def test_list_pipelines(self):
        """list_pipelines returns all registered names."""
        loop = SDALoop()
        loop.register(_DummySense(), _DummyDecide(), _DummyAct(), name="a")
        loop.register(_DummySense(), _DummyDecide(), _DummyAct(), name="b")
        assert set(loop.list_pipelines()) == {"a", "b"}

    def test_metrics_collection_accurate(self):
        """After ticking, metrics must reflect what happened."""
        loop = SDALoop()
        loop.register(_DummySense(), _DummyDecide("go", 0.8), _DummyAct(), name="m1")
        loop.tick()

        metrics = loop.get_metrics()
        assert metrics["tick_count"] == 1
        assert metrics["pipeline_ticks"]["m1"] == 1
        assert metrics["decision_counts"]["go"] == 1
        assert metrics["act_success_rate"] == 1.0
        assert metrics["mean_latency_ms"] is not None
        assert metrics["mean_latency_ms"] >= 0.0

    def test_metrics_initial_state(self):
        """Before any ticks, metrics report zero/None as appropriate."""
        loop = SDALoop()
        metrics = loop.get_metrics()
        assert metrics["tick_count"] == 0
        assert metrics["mean_latency_ms"] is None
        assert metrics["max_latency_ms"] is None
        assert metrics["act_success_rate"] == 0.0

    def test_disabled_pipeline_skipped(self):
        """Disabled pipelines must return None on tick and not increment ticks."""
        loop = SDALoop()
        loop.register(_DummySense(), _DummyDecide(), _DummyAct(), name="on")
        loop.register(_DummySense(), _DummyDecide(), _DummyAct(), name="off")
        loop.disable("off")

        results = loop.tick()
        assert results["on"] is not None
        assert results["off"] is None

        metrics = loop.get_metrics()
        assert metrics["pipeline_ticks"]["on"] == 1
        assert "off" not in metrics["pipeline_ticks"] or metrics["pipeline_ticks"]["off"] == 0

    def test_decision_confidence_threshold_skips_act(self):
        """If decision.confidence < 0.5, Act must not run."""
        loop = SDALoop(confidence_threshold=0.5)
        decide = _DummyDecide(action_type="risky", confidence=0.3)
        act = _DummyAct()
        loop.register(_DummySense(), decide, act, name="low_conf")

        results = loop.tick()
        assert results["low_conf"] is None
        assert act.call_order is None  # never called

    def test_multiple_pipelines(self):
        """Tick must run all enabled pipelines independently."""
        loop = SDALoop()
        loop.register(_DummySense({"a": 1}), _DummyDecide("act_a", 1.0), _DummyAct(), name="p1")
        loop.register(_DummySense({"b": 2}), _DummyDecide("act_b", 1.0), _DummyAct(), name="p2")

        results = loop.tick()
        assert len(results) == 2
        assert results["p1"] is not None
        assert results["p2"] is not None

        metrics = loop.get_metrics()
        assert metrics["pipeline_ticks"]["p1"] == 1
        assert metrics["pipeline_ticks"]["p2"] == 1
        assert metrics["decision_counts"]["act_a"] == 1
        assert metrics["decision_counts"]["act_b"] == 1

    def test_pipeline_interval_ms_respected(self):
        """Pipelines with interval_ms must throttle."""
        loop = SDALoop()
        loop.register(
            _DummySense(), _DummyDecide(), _DummyAct(),
            name="throttled", interval_ms=5000.0,
        )

        r1 = loop.tick()
        assert r1["throttled"] is not None

        r2 = loop.tick()
        assert r2["throttled"] is None  # throttled

    def test_pipeline_name_override(self):
        """Registering the same name replaces the old pipeline."""
        loop = SDALoop()
        old_sense = _DummySense()
        new_sense = _DummySense()
        loop.register(old_sense, _DummyDecide(), _DummyAct(), name="x")
        loop.register(new_sense, _DummyDecide(), _DummyAct(), name="x")

        pipe = loop.get_pipeline("x")
        assert pipe.sense is new_sense

    def test_thread_safe_concurrent_ticks(self):
        """Multiple threads calling tick() must not corrupt metrics."""
        loop = SDALoop()
        loop.register(_DummySense(), _DummyDecide("safe", 1.0), _DummyAct(), name="concurrent")

        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(20):
                    loop.tick()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        metrics = loop.get_metrics()
        assert metrics["tick_count"] == 100
        assert metrics["pipeline_ticks"]["concurrent"] == 100

    def test_sense_failure_produces_error_act_result(self):
        """If Sense raises, the pipeline returns a failed ActResult."""
        class _BrokenSense(Sense):
            def observe(self) -> Observation:
                raise RuntimeError("sensor offline")

        loop = SDALoop()
        loop.register(_BrokenSense(), _DummyDecide(), _DummyAct(), name="broken")
        result = loop.tick()["broken"]
        assert result is not None
        assert result.success is False
        assert "sense_error" in result.side_effects

    def test_decide_failure_produces_error_act_result(self):
        """If Decide raises, the pipeline returns a failed ActResult."""
        class _BrokenDecide(Decide):
            def evaluate(self, observation: Observation) -> Decision:
                raise RuntimeError("policy corrupt")

        loop = SDALoop()
        loop.register(_DummySense(), _BrokenDecide(), _DummyAct(), name="broken_decide")
        result = loop.tick()["broken_decide"]
        assert result is not None
        assert result.success is False
        assert "decide_error" in result.side_effects

    def test_act_failure_tracks_success_rate(self):
        """Failed Act executions must reduce act_success_rate."""
        loop = SDALoop()
        loop.register(
            _DummySense(), _DummyDecide("boom", 1.0),
            _DummyAct(success=False), name="fail",
        )
        loop.tick()

        metrics = loop.get_metrics()
        assert metrics["act_success_rate"] == 0.0


# ═══════════════════════════════════════════════════════════════
# Built-in adapter tests
# ═══════════════════════════════════════════════════════════════

class TestTrapSense:
    def test_trap_sense_observes_registry(self):
        """TrapSense must produce an Observation from a TrapRegistry."""
        from fleet.operational_trap import TrapRegistry, ThermalTrap

        registry = TrapRegistry()
        # Create a minimal mock budget
        class MockBudget:
            _devices = {}

        trap = ThermalTrap(budget=MockBudget())
        registry.register(trap)

        sense = TrapSense(registry)
        obs = sense.observe()
        assert obs.source == "trap_sense"
        assert "trap_count" in obs.metrics
        assert obs.metrics["trap_count"] == 1


class TestGatewayDispatchDecide:
    def test_open_gateway_decides_dispatch(self):
        """OPEN gateway should decide to dispatch with high confidence."""
        from fleet.gateway_pacing import GatewayPacing

        gateway = GatewayPacing()
        decide = GatewayDispatchDecide(gateway)
        obs = Observation(timestamp=0.0, source="gateway", metrics={})
        decision = decide.evaluate(obs)
        assert decision.action_type == "dispatch"
        assert decision.confidence == 1.0

    def test_closed_gateway_decides_defer(self):
        """CLOSED gateway should decide to defer with low confidence."""
        from fleet.gateway_pacing import GatewayPacing

        gateway = GatewayPacing()
        # Force circuit closed by recording max consecutive timeouts
        for _ in range(gateway._max_consecutive_timeouts):
            gateway.record_timeout()

        decide = GatewayDispatchDecide(gateway)
        obs = Observation(timestamp=0.0, source="gateway", metrics={})
        decision = decide.evaluate(obs)
        assert decision.action_type == "defer"
        assert decision.confidence < 0.5


class TestHebbianMeshSense:
    def test_hebbian_mesh_sense_observes_stats(self):
        """HebbianMeshSense must read stats and chaos_factor from the layer."""
        from swarm.hebbian_mesh import HebbianMeshLayer

        # Minimal mock gossip
        class MockGossip:
            max_peers_per_round = 2
            local_table = None

        mesh = HebbianMeshLayer(MockGossip())
        sense = HebbianMeshSense(mesh)
        obs = sense.observe()
        assert obs.source == "hebbian_mesh"
        assert "peer_count" in obs.metrics
        assert "chaos_factor" in obs.metrics


class TestHebbianMeshAct:
    def test_hebbian_mesh_act_gossip(self):
        """HebbianMeshAct with action 'gossip' should select peers."""
        from swarm.hebbian_mesh import HebbianMeshLayer

        class MockGossip:
            max_peers_per_round = 2
            local_table = None

        mesh = HebbianMeshLayer(MockGossip())
        act = HebbianMeshAct(mesh, peer_pool=["A", "B", "C"])
        decision = Decision(
            action_type="gossip",
            confidence=1.0,
            payload={},
        )
        result = act.execute(decision)
        assert result.success is True
        assert any("selected_peers" in eff for eff in result.side_effects)


class TestFluxPresetDecide:
    def test_flux_preset_decide_continues_when_all_pass(self):
        """All constraints passing should yield 'continue' decision."""
        from sunset.flux_preset_library import FluxPresetLibrary

        lib = FluxPresetLibrary()
        decide = FluxPresetDecide(lib, preset_name="RangeCheck")
        obs = Observation(
            timestamp=0.0,
            source="flux",
            metrics={
                "thermal_headroom": 0.5,
                "chaos": 0.3,
                "weights": 3.0,
                "weight_bounds": (0.0, 10.0),
            },
        )
        decision = decide.evaluate(obs)
        assert decision.action_type == "continue"
        assert decision.confidence == 1.0

    def test_flux_preset_decide_blocks_on_critical(self):
        """Critical breach should yield 'block' decision."""
        from sunset.flux_preset_library import FluxPresetLibrary

        lib = FluxPresetLibrary()
        decide = FluxPresetDecide(lib, preset_name="ThermalCeiling")
        obs = Observation(
            timestamp=0.0,
            source="flux",
            metrics={"thermal_headroom": 1.5},  # way over ceiling
        )
        decision = decide.evaluate(obs)
        assert decision.action_type == "block"
        assert decision.confidence == 0.9


class TestBreedCoordinationSense:
    def test_breed_coordination_sense_observes_beat_phase(self):
        """BreedCoordinationSense must report beat phase and pool info."""
        from nerve.metronome_bridge import MetronomeBridge

        class MockGrid:
            n = 4
            chaos = [0.1, 0.2, 0.3, 0.4]
            activity = [1, 2, 3, 4]
            latents = []
            w = []

        class MockScheduler:
            class signal_source:
                @staticmethod
                def next_signal(beat):
                    return [0.0] * 64

        bridge = MetronomeBridge(MockGrid(), MockScheduler())
        sense = BreedCoordinationSense(bridge)
        obs = sense.observe()
        assert obs.source == "breed_coordination"
        assert "beat_phase" in obs.metrics
        assert "tick_counter" in obs.metrics


class TestBreedCoordinationAct:
    def test_breed_coordination_act_dispatch(self):
        """BreedCoordinationAct with 'dispatch_beat' should tick rooms."""
        from nerve.metronome_bridge import MetronomeBridge

        class MockGrid:
            n = 4
            chaos = [0.1, 0.2, 0.3, 0.4]
            activity = [1, 2, 3, 4]
            latents = []
            w = []

            def tick(self, signal):
                return {"fired": 0}

        class MockScheduler:
            class signal_source:
                @staticmethod
                def next_signal(beat):
                    return [0.0] * 64

        bridge = MetronomeBridge(MockGrid(), MockScheduler())
        act = BreedCoordinationAct(bridge)
        decision = Decision(
            action_type="dispatch_beat",
            confidence=1.0,
            payload={"beat_number": 0},
        )
        result = act.execute(decision)
        assert result.success is True
        assert any("dispatched" in eff for eff in result.side_effects)


# ═══════════════════════════════════════════════════════════════
# Integration: full SDA cycle with built-ins
# ═══════════════════════════════════════════════════════════════

class TestBuiltInPipelines:
    def test_all_built_in_pipelines_load_and_tick(self):
        """Register all 5 built-in pipelines and verify they tick without error."""
        from fleet.gateway_pacing import GatewayPacing
        from fleet.operational_trap import TrapRegistry, ThermalTrap
        from nerve.metronome_bridge import MetronomeBridge
        from sunset.flux_preset_library import FluxPresetLibrary
        from swarm.hebbian_mesh import HebbianMeshLayer

        loop = SDALoop()

        # 1. thermal_monitoring
        registry = TrapRegistry()
        registry.register(ThermalTrap(budget=object()))  # invalid budget → no trap fire
        loop.register(
            TrapSense(registry),
            Policy(),  # default noop
            _DummyAct(),
            name="thermal_monitoring",
        )

        # 2. dispatch_gating
        gateway = GatewayPacing()
        loop.register(
            _DummySense({"gateway": "ok"}),
            GatewayDispatchDecide(gateway),
            _DummyAct(),
            name="dispatch_gating",
        )

        # 3. mesh_exploration
        class MockGossip:
            max_peers_per_round = 2
            local_table = None

        mesh = HebbianMeshLayer(MockGossip())
        loop.register(
            HebbianMeshSense(mesh),
            Policy(),
            HebbianMeshAct(mesh, peer_pool=["A", "B"]),
            name="mesh_exploration",
        )

        # 4. flux_constraint
        lib = FluxPresetLibrary()
        loop.register(
            _DummySense({"thermal_headroom": 0.5}),
            FluxPresetDecide(lib, preset_name="FleetHealth"),
            _DummyAct(),
            name="flux_constraint",
        )

        # 5. breed_coordination
        class MockGrid:
            n = 4
            chaos = [0.1, 0.2, 0.3, 0.4]
            activity = [1, 2, 3, 4]
            latents = []
            w = []

        class MockScheduler:
            class signal_source:
                @staticmethod
                def next_signal(beat):
                    return [0.0] * 64

        bridge = MetronomeBridge(MockGrid(), MockScheduler())
        loop.register(
            BreedCoordinationSense(bridge),
            Policy(),
            BreedCoordinationAct(bridge),
            name="breed_coordination",
        )

        results = loop.tick()
        assert len(results) == 5
        # All should produce non-None results (none disabled, all confidence >= 0.5)
        for name, result in results.items():
            assert result is not None, f"Pipeline {name} returned None"

        metrics = loop.get_metrics()
        assert metrics["total_pipelines"] == 5
        assert all(metrics["pipeline_ticks"].get(n, 0) == 1 for n in loop.list_pipelines())
