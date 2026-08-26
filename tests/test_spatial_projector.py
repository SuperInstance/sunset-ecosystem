"""
Tests for A2A Spatial Projector System.

Covers: WorldState, Prediction, SpatialIndex, SpatialProjector,
FluxConstraint, WorldModelBridge.
"""

import math
import time
import pytest

from fleet.spatial_projector import (
    FluxConstraint,
    Prediction,
    SpatialIndex,
    SpatialProjector,
    WorldState,
    create_room_constraint,
    create_thermal_constraint,
    create_uncertainty_constraint,
)
from fleet.worldmodel_bridge import WorldModelBridge, SolverConfig, EnvironmentConfig


# ──────────────────────────── WorldState ────────────────────────────


class TestWorldState:
    def test_basic_creation(self):
        s = WorldState(position=(1.0, 2.0, 3.0))
        assert s.position == (1.0, 2.0, 3.0)
        assert s.confidence == 1.0
        assert s.agent_id is None

    def test_distance_2d(self):
        a = WorldState(position=(0.0, 0.0))
        b = WorldState(position=(3.0, 4.0))
        assert a.distance_to(b) == 5.0

    def test_distance_3d(self):
        a = WorldState(position=(0.0, 0.0, 0.0))
        b = WorldState(position=(1.0, 2.0, 2.0))
        assert a.distance_to(b) == 3.0

    def test_distance_dimension_mismatch(self):
        a = WorldState(position=(0.0, 0.0))
        b = WorldState(position=(0.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            a.distance_to(b)

    def test_to_vector(self):
        s = WorldState(position=(1.0, 2.0), velocity=(3.0, 4.0), orientation=0.5)
        vec = s.to_vector()
        assert vec == [1.0, 2.0, 3.0, 4.0, 0.5]

    def test_to_dict_roundtrip(self):
        s = WorldState(
            position=(1.0, 2.0),
            velocity=(3.0, 4.0),
            orientation=0.5,
            semantics={"room_type": "ethos"},
            confidence=0.95,
            agent_id="a1",
            room_id="ethos-thermal",
        )
        d = s.to_dict()
        s2 = WorldState.from_dict(d)
        assert s2.position == (1.0, 2.0)
        assert s2.semantics == {"room_type": "ethos"}
        assert s2.agent_id == "a1"

    def test_static_state_vector(self):
        s = WorldState(position=(5.0, 5.0))
        assert s.to_vector() == [5.0, 5.0]


# ──────────────────────────── Prediction ────────────────────────────


class TestPrediction:
    def test_final_state(self):
        states = [
            WorldState(position=(0.0, 0.0)),
            WorldState(position=(1.0, 1.0)),
            WorldState(position=(2.0, 2.0)),
        ]
        p = Prediction(trajectory=states)
        assert p.final_state.position == (2.0, 2.0)

    def test_mean_uncertainty(self):
        p = Prediction(
            trajectory=[WorldState(position=(0.0,))], uncertainty=[0.1, 0.2, 0.3]
        )
        assert abs(p.mean_uncertainty - 0.2) < 0.001

    def test_empty_uncertainty(self):
        p = Prediction(trajectory=[WorldState(position=(0.0,))], uncertainty=[])
        assert p.mean_uncertainty == 0.0

    def test_to_dict(self):
        p = Prediction(
            trajectory=[WorldState(position=(1.0, 2.0), agent_id="a1")],
            rewards=[1.0],
            uncertainty=[0.1],
            model_id="test",
            agent_id="a1",
        )
        d = p.to_dict()
        assert d["model_id"] == "test"
        assert d["agent_id"] == "a1"
        assert len(d["trajectory"]) == 1


# ──────────────────────────── SpatialIndex ────────────────────────────


class TestSpatialIndex:
    def test_insert_and_query(self):
        idx = SpatialIndex(dimension=2)
        s = WorldState(position=(0.0, 0.0), agent_id="a1", room_id="room1")
        idx.insert("p1", s)
        assert idx.get_latest("a1") == s

    def test_query_radius(self):
        idx = SpatialIndex(dimension=2)
        idx.insert("p1", WorldState(position=(0.0, 0.0), agent_id="a1"))
        idx.insert("p2", WorldState(position=(3.0, 4.0), agent_id="a2"))  # dist=5
        idx.insert("p3", WorldState(position=(10.0, 0.0), agent_id="a3"))  # dist=10

        center = WorldState(position=(0.0, 0.0))
        results = idx.query_radius(center, radius=6.0)
        # Exclude center itself (a1 at dist 0)
        others = [r for r in results if r[1].agent_id != "a1"]
        assert len(others) == 1  # Only a2 within 6
        assert others[0][1].agent_id == "a2"

    def test_query_knn(self):
        idx = SpatialIndex(dimension=2)
        for i in range(5):
            idx.insert(f"p{i}", WorldState(position=(float(i), 0.0), agent_id=f"a{i}"))

        center = WorldState(position=(0.0, 0.0))
        knn = idx.query_knn(center, k=3)
        assert len(knn) == 3
        # Closest should be a0, then a1, then a2
        assert [r[1].agent_id for r in knn] == ["a0", "a1", "a2"]

    def test_query_semantic(self):
        idx = SpatialIndex(dimension=2)
        idx.insert("p1", WorldState(position=(0.0, 0.0), semantics={"role": "breeder"}))
        idx.insert("p2", WorldState(position=(1.0, 1.0), semantics={"role": "solver"}))

        breeders = idx.query_semantic("role", "breeder")
        assert len(breeders) == 1
        assert breeders[0].semantics["role"] == "breeder"

    def test_room_filter(self):
        idx = SpatialIndex(dimension=2)
        idx.insert(
            "p1", WorldState(position=(0.0, 0.0), agent_id="a1", room_id="ethos")
        )
        idx.insert(
            "p2", WorldState(position=(1.0, 1.0), agent_id="a2", room_id="pathos")
        )

        center = WorldState(position=(0.0, 0.0))
        results = idx.query_radius(center, radius=10.0, room_filter="ethos")
        assert len(results) == 1
        assert results[0][1].agent_id == "a1"

    def test_remove(self):
        idx = SpatialIndex(dimension=2)
        idx.insert("p1", WorldState(position=(0.0, 0.0), agent_id="a1"))
        idx.remove("p1")
        assert idx.get_latest("a1") is None

    def test_snapshot(self):
        idx = SpatialIndex(dimension=2)
        idx.insert("p1", WorldState(position=(1.0, 2.0), agent_id="a1"))
        snap = idx.snapshot()
        assert snap["dimension"] == 2
        assert len(snap["entries"]) == 1
        assert "timestamp" in snap


# ──────────────────────────── SpatialProjector ────────────────────────────


class TestSpatialProjector:
    def test_project_and_query(self):
        proj = SpatialProjector("node-1", dimension=2)
        state = WorldState(position=(0.0, 0.0), semantics={"temp": 65.0})
        pid = proj.project_state("agent-1", "ethos-thermal", state)
        assert len(pid) == 16  # SHA-256 truncated

        latest = proj.get_agent_state("agent-1")
        assert latest is not None
        assert latest.agent_id == "agent-1"
        assert latest.room_id == "ethos-thermal"

    def test_query_neighbors(self):
        proj = SpatialProjector("node-1", dimension=2)
        proj.project_state("agent-1", "room1", WorldState(position=(0.0, 0.0)))
        proj.project_state("agent-2", "room1", WorldState(position=(3.0, 4.0)))
        proj.project_state("agent-3", "room1", WorldState(position=(100.0, 0.0)))

        neighbors = proj.query_neighbors("agent-1", radius=6.0)
        assert len(neighbors) == 1
        assert neighbors[0].agent_id == "agent-2"

    def test_query_neighbors_exclude_self(self):
        proj = SpatialProjector("node-1", dimension=2)
        proj.project_state("agent-1", "room1", WorldState(position=(0.0, 0.0)))
        proj.project_state("agent-1", "room1", WorldState(position=(1.0, 1.0)))

        neighbors = proj.query_neighbors("agent-1", radius=10.0)
        # Should exclude same agent
        assert all(n.agent_id != "agent-1" for n in neighbors)

    def test_predict_trajectory_with_velocity(self):
        proj = SpatialProjector("node-1", dimension=2)
        proj.project_state(
            "agent-1", "room1", WorldState(position=(0.0, 0.0), velocity=(1.0, 0.0))
        )

        pred = proj.predict_trajectory("agent-1", horizon=3)
        assert len(pred.trajectory) == 4  # current + 3 steps
        assert pred.trajectory[1].position == (1.0, 0.0)
        assert pred.trajectory[2].position == (2.0, 0.0)
        assert pred.trajectory[3].position == (3.0, 0.0)

    def test_predict_trajectory_no_velocity(self):
        proj = SpatialProjector("node-1", dimension=2)
        proj.project_state("agent-1", "room1", WorldState(position=(5.0, 5.0)))

        pred = proj.predict_trajectory("agent-1", horizon=3)
        assert len(pred.trajectory) == 4
        # Static: all positions same
        assert all(s.position == (5.0, 5.0) for s in pred.trajectory)

    def test_predict_trajectory_no_state(self):
        proj = SpatialProjector("node-1", dimension=2)
        with pytest.raises(ValueError):
            proj.predict_trajectory("nonexistent", horizon=3)

    def test_flux_hard_constraint_violation(self):
        proj = SpatialProjector("node-1", dimension=2)
        proj.add_flux_constraint(create_thermal_constraint(max_temp=50.0, hard=True))

        # Create a prediction with temperature > 50
        states = [
            WorldState(position=(0.0, 0.0), semantics={"temperature": 30.0}),
            WorldState(
                position=(1.0, 0.0), semantics={"temperature": 60.0}
            ),  # Violation!
        ]
        pred = Prediction(trajectory=states)

        with pytest.raises(ValueError, match="thermal_feasibility"):
            proj.apply_flux_gate(pred)

    def test_flux_soft_constraint(self):
        proj = SpatialProjector("node-1", dimension=2)
        proj.add_flux_constraint(create_thermal_constraint(max_temp=50.0, hard=False))

        states = [
            WorldState(
                position=(0.0, 0.0), semantics={"temperature": 30.0}, confidence=1.0
            ),
            WorldState(
                position=(1.0, 0.0), semantics={"temperature": 60.0}, confidence=1.0
            ),
        ]
        pred = Prediction(trajectory=states)
        result = proj.apply_flux_gate(pred)

        # Soft constraint: confidence reduced
        assert all(s.confidence < 1.0 for s in result.trajectory[1:])

    def test_flux_multiple_constraints(self):
        proj = SpatialProjector("node-1", dimension=2)
        proj.add_flux_constraint(create_thermal_constraint(max_temp=100.0, hard=True))
        proj.add_flux_constraint(
            create_uncertainty_constraint(max_uncertainty=0.5, hard=True)
        )

        states = [WorldState(position=(0.0, 0.0), semantics={"temperature": 50.0})]
        pred = Prediction(trajectory=states, uncertainty=[0.3])
        result = proj.apply_flux_gate(pred)
        assert result is not None

    def test_broadcast_prediction(self):
        proj = SpatialProjector("node-1", dimension=2)
        received = []
        proj.on_prediction(lambda p: received.append(p))

        states = [WorldState(position=(0.0, 0.0))]
        pred = Prediction(trajectory=states)
        meta = proj.broadcast_prediction(pred)

        assert meta["flux_passed"] is True
        assert len(received) == 1

    def test_get_proximal_agents(self):
        proj = SpatialProjector("node-1", dimension=2)
        proj.project_state("a1", "room1", WorldState(position=(0.0, 0.0)))
        proj.project_state("a2", "room1", WorldState(position=(3.0, 4.0)))
        proj.project_state("a3", "room1", WorldState(position=(100.0, 0.0)))

        proximal = proj.get_proximal_agents("a1", radius=6.0)
        assert proximal == ["a2"]

    def test_spatial_diversity_isolated(self):
        proj = SpatialProjector("node-1", dimension=2)
        proj.project_state("a1", "room1", WorldState(position=(0.0, 0.0)))
        # No other agents
        score = proj.get_spatial_diversity_score("a1")
        assert score == 1.0

    def test_spatial_diversity_clustered(self):
        proj = SpatialProjector("node-1", dimension=2)
        proj.project_state("a1", "room1", WorldState(position=(0.0, 0.0)))
        proj.project_state("a2", "room1", WorldState(position=(1.0, 0.0)))
        proj.project_state("a3", "room1", WorldState(position=(2.0, 0.0)))

        score = proj.get_spatial_diversity_score("a1")
        assert 0.0 < score < 1.0  # Clustered but not identical

    def test_snapshot_and_ingest(self):
        proj1 = SpatialProjector("node-1", dimension=2)
        proj1.project_state("a1", "room1", WorldState(position=(1.0, 2.0)))
        snap = proj1.snapshot()

        proj2 = SpatialProjector("node-2", dimension=2)
        count = proj2.ingest_snapshot(snap)
        assert count == 1
        assert proj2.get_agent_state("a1") is not None

    def test_room_constraint(self):
        proj = SpatialProjector("node-1", dimension=2)
        proj.add_flux_constraint(create_room_constraint(["room1", "room2"], hard=True))

        # Prediction staying in allowed room
        states = [WorldState(position=(0.0, 0.0), room_id="room1")]
        pred_ok = Prediction(trajectory=states)
        assert proj.apply_flux_gate(pred_ok) is not None

        # Prediction going to forbidden room
        states_bad = [
            WorldState(position=(0.0, 0.0), room_id="room1"),
            WorldState(position=(1.0, 0.0), room_id="room3"),
        ]
        pred_bad = Prediction(trajectory=states_bad)
        with pytest.raises(ValueError, match="room_boundary"):
            proj.apply_flux_gate(pred_bad)

    def test_prediction_history(self):
        proj = SpatialProjector("node-1", dimension=2)
        proj.project_state(
            "a1", "room1", WorldState(position=(0.0, 0.0), velocity=(1.0, 0.0))
        )

        p1 = proj.predict_trajectory("a1", horizon=2)
        p2 = proj.predict_trajectory("a1", horizon=2)

        history = proj._prediction_history.get("a1", [])
        assert len(history) == 2

    def test_semantic_broadcast_filter(self):
        proj = SpatialProjector("node-1", dimension=2)
        proj.project_state(
            "breeder-1",
            "ethos",
            WorldState(position=(0.0, 0.0), semantics={"role": "breeder"}),
        )
        proj.project_state(
            "solver-1",
            "pathos",
            WorldState(position=(10.0, 0.0), semantics={"role": "solver"}),
        )

        breeders = proj.query_semantic("role", "breeder")
        assert len(breeders) == 1
        assert breeders[0].agent_id == "breeder-1"


# ──────────────────────────── WorldModelBridge ────────────────────────────


class TestWorldModelBridge:
    def test_detect_swm_mock(self):
        bridge = WorldModelBridge()
        # In test environment, stable-worldmodel is not installed
        assert bridge.has_real_worldmodel is False

    def test_fallback_predict(self):
        bridge = WorldModelBridge()
        state = WorldState(position=(0.0, 0.0), velocity=(1.0, 0.0))
        pred = bridge.predict("agent-1", state, horizon=3)
        assert len(pred.trajectory) == 4
        assert pred.model_id == "fleet-default"

    def test_fallback_predict_static(self):
        bridge = WorldModelBridge()
        state = WorldState(position=(5.0, 5.0))  # No velocity
        pred = bridge.predict("agent-1", state, horizon=3)
        assert len(pred.trajectory) == 4
        assert all(s.position == (5.0, 5.0) for s in pred.trajectory)

    def test_predict_batch(self):
        bridge = WorldModelBridge()
        states = [
            ("a1", WorldState(position=(0.0, 0.0), velocity=(1.0, 0.0))),
            ("a2", WorldState(position=(10.0, 0.0), velocity=(0.0, 1.0))),
        ]
        preds = bridge.predict_batch(states, horizon=3)
        assert len(preds) == 2
        assert "a1" in preds
        assert "a2" in preds

    def test_solver_config(self):
        cfg = SolverConfig(name="MPPI", num_samples=500, horizon=20)
        assert cfg.name == "MPPI"
        assert cfg.num_samples == 500

    def test_environment_config(self):
        cfg = EnvironmentConfig(env_id="TwoRoom-v0", num_envs=4)
        assert cfg.env_id == "TwoRoom-v0"
        assert cfg.num_envs == 4

    def test_bridge_status(self):
        bridge = WorldModelBridge()
        status = bridge.get_status()
        assert status["has_swm"] is False
        assert status["mock_fallback"] is True
        assert status["solver_name"] == "CEM"

    def test_to_flux_constraints(self):
        bridge = WorldModelBridge()
        constraints = bridge.to_flux_constraints(thermal_budget=75.0)
        assert len(constraints) == 1
        assert constraints[0].name == "worldmodel_thermal"

    def test_evaluate_policy_mock(self):
        bridge = WorldModelBridge()
        metrics = bridge.evaluate_policy(lambda x: x, episodes=10)
        assert "success_rate" in metrics
        assert "mean_reward" in metrics

    def test_load_solver_mock(self):
        bridge = WorldModelBridge()
        # Should return False because stable-worldmodel not installed
        assert bridge.load_solver("CEM") is False

    def test_load_environment_mock(self):
        bridge = WorldModelBridge()
        assert bridge.load_environment("PushT-v1") is False


# ──────────────────────────── FluxConstraint Factory ────────────────────────────


class TestFluxConstraintFactories:
    def test_thermal_hard_pass(self):
        c = create_thermal_constraint(max_temp=50.0, hard=True)
        states = [WorldState(position=(0.0, 0.0), semantics={"temperature": 30.0})]
        pred = Prediction(trajectory=states)
        passed, _ = c.evaluate(pred)
        assert passed is True

    def test_thermal_hard_fail(self):
        c = create_thermal_constraint(max_temp=50.0, hard=True)
        states = [WorldState(position=(0.0, 0.0), semantics={"temperature": 60.0})]
        pred = Prediction(trajectory=states)
        passed, _ = c.evaluate(pred)
        assert passed is False

    def test_thermal_soft_penalty(self):
        c = create_thermal_constraint(max_temp=50.0, hard=False)
        states = [WorldState(position=(0.0, 0.0), semantics={"temperature": 60.0})]
        pred = Prediction(trajectory=states)
        passed, penalty = c.evaluate(pred)
        assert passed is True  # Soft: doesn't fail
        assert penalty > 0.0

    def test_uncertainty_hard(self):
        c = create_uncertainty_constraint(max_uncertainty=0.3, hard=True)
        pred = Prediction(trajectory=[WorldState(position=(0.0,))], uncertainty=[0.1])
        passed, _ = c.evaluate(pred)
        assert passed is True

        pred_bad = Prediction(
            trajectory=[WorldState(position=(0.0,))], uncertainty=[0.5]
        )
        passed, _ = c.evaluate(pred_bad)
        assert passed is False

    def test_room_constraint_pass(self):
        c = create_room_constraint(["room1", "room2"], hard=True)
        states = [WorldState(position=(0.0, 0.0), room_id="room1")]
        pred = Prediction(trajectory=states)
        passed, _ = c.evaluate(pred)
        assert passed is True

    def test_room_constraint_fail(self):
        c = create_room_constraint(["room1", "room2"], hard=True)
        states = [WorldState(position=(0.0, 0.0), room_id="room3")]
        pred = Prediction(trajectory=states)
        passed, _ = c.evaluate(pred)
        assert passed is False


# ──────────────────────────── Integration ────────────────────────────


class TestSpatialProjectorIntegration:
    def test_full_pipeline(self):
        """End-to-end: project → predict → flux gate → broadcast."""
        proj = SpatialProjector("node-test", dimension=2)
        proj.add_flux_constraint(create_thermal_constraint(max_temp=80.0, hard=True))

        # Agent projects state
        proj.project_state(
            "breeder-1",
            "ethos-thermal",
            WorldState(
                position=(0.0, 0.0),
                velocity=(1.0, 0.0),
                semantics={"temperature": 65.0},
            ),
        )

        # Predict trajectory
        pred = proj.predict_trajectory("breeder-1", horizon=5)
        assert len(pred.trajectory) == 6

        # Apply FLUX gate
        validated = proj.apply_flux_gate(pred)
        assert validated is not None

        # Broadcast
        received = []
        proj.on_prediction(lambda p: received.append(p))
        meta = proj.broadcast_prediction(validated)
        assert meta["flux_passed"] is True
        assert len(received) == 1

    def test_multi_agent_spatial_awareness(self):
        """Three agents in different rooms with spatial queries."""
        proj = SpatialProjector("node-test", dimension=3)

        # Agent in ethos room (thermal management)
        proj.project_state(
            "breeder-1",
            "ethos",
            WorldState(
                position=(0.0, 0.0, 0.0),
                semantics={"temperature": 65.0, "role": "breeder"},
            ),
        )

        # Agent in pathos room (human interaction)
        proj.project_state(
            "solver-1",
            "pathos",
            WorldState(
                position=(10.0, 0.0, 0.0),
                semantics={"sentiment": 0.8, "role": "solver"},
            ),
        )

        # Agent in logos room (code quality)
        proj.project_state(
            "auditor-1",
            "logos",
            WorldState(
                position=(20.0, 0.0, 0.0),
                semantics={"complexity": 12.5, "role": "auditor"},
            ),
        )

        # Query: who is near breeder-1?
        near = proj.query_neighbors("breeder-1", radius=15.0)
        assert len(near) == 1  # Only solver-1 within 15
        assert near[0].agent_id == "solver-1"

        # Query: all breeders
        breeders = proj.query_semantic("role", "breeder")
        assert len(breeders) == 1
        assert breeders[0].agent_id == "breeder-1"

        # Diversity: breeder-1 is isolated from auditors
        div = proj.get_spatial_diversity_score("breeder-1")
        assert 0.0 < div < 1.0

    def test_cross_node_sync(self):
        """Simulate two fleet nodes syncing spatial state."""
        node_alpha = SpatialProjector("node-alpha", dimension=2)
        node_beta = SpatialProjector("node-beta", dimension=2)

        # Alpha has an agent
        node_alpha.project_state(
            "agent-x", "ethos", WorldState(position=(5.0, 5.0), semantics={"load": 0.8})
        )

        # Beta ingests alpha's snapshot
        snap = node_alpha.snapshot()
        count = node_beta.ingest_snapshot(snap)
        assert count == 1

        # Beta can now query agent-x
        state = node_beta.get_agent_state("agent-x")
        assert state is not None
        assert state.position == (5.0, 5.0)

    def test_bridge_with_projector(self):
        """WorldModelBridge feeding predictions into projector."""
        proj = SpatialProjector("node-test", dimension=2)
        bridge = WorldModelBridge()

        # Bridge generates prediction
        current = WorldState(position=(0.0, 0.0), velocity=(2.0, 0.0))
        pred = bridge.predict("agent-1", current, horizon=4)

        # FLUX gate via projector
        proj.add_flux_constraint(
            create_uncertainty_constraint(max_uncertainty=0.6, hard=False)
        )
        validated = proj.apply_flux_gate(pred)
        assert validated is not None

    def test_trinity_room_positions(self):
        """Map trinity rooms to spatial coordinates."""
        proj = SpatialProjector("node-test", dimension=2)

        # Ethos at origin
        proj.project_state("agent-e", "ethos", WorldState(position=(0.0, 0.0)))
        # Pathos at (0, 10)
        proj.project_state("agent-p", "pathos", WorldState(position=(0.0, 10.0)))
        # Logos at (10, 0)
        proj.project_state("agent-l", "logos", WorldState(position=(10.0, 0.0)))

        # Distance from ethos to pathos = 10
        e = proj.get_agent_state("agent-e")
        p = proj.get_agent_state("agent-p")
        assert e.distance_to(p) == 10.0

        # Distance from ethos to logos = 10
        l = proj.get_agent_state("agent-l")
        assert e.distance_to(l) == 10.0

        # Distance from pathos to logos = sqrt(200) ≈ 14.14
        assert abs(p.distance_to(l) - 14.142) < 0.001
