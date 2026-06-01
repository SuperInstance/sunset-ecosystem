"""tests/test_fleet_consciousness_bridge.py — Test suite for FCI bridge.

Covers:
- FCI computation with default weights
- All consciousness levels (dormant → transcendent)
- Text/JSON/oneline rendering
- Raw fleet metrics computation
- Edge cases (zero division, clamping)
"""

import pytest
from fleet.fleet_consciousness_bridge import FleetConsciousnessIndex, ConsciousnessScore


class TestConsciousnessScore:
    def test_to_dict(self):
        s = ConsciousnessScore(
            fci=0.5,
            level="conscious",
            room_phi_score=0.3,
            attention_score=0.2,
            learning_score=0.5,
            meta_score=0.0,
            status="HEALTHY",
            recommendation="Test",
            details={"extra": 1},
        )
        d = s.to_dict()
        assert d["fci"] == 0.5
        assert d["level"] == "conscious"
        assert d["details"]["extra"] == 1


class TestFleetConsciousnessIndex:
    def test_compute_dormant(self):
        fci = FleetConsciousnessIndex()
        score = fci.compute(
            room_phi_score=0.0,
            attention_score=0.0,
            learning_score=0.0,
            meta_score=0.0,
        )
        assert score.fci == 0.0
        assert score.level == "dormant"
        assert score.status == "DEGRADED"

    def test_compute_emerging(self):
        fci = FleetConsciousnessIndex()
        score = fci.compute(
            room_phi_score=0.30,
            attention_score=0.20,
            learning_score=0.10,
            meta_score=0.05,
        )
        assert 0.15 <= score.fci < 0.30
        assert score.level == "emerging"

    def test_compute_aware(self):
        fci = FleetConsciousnessIndex()
        score = fci.compute(
            room_phi_score=0.50,
            attention_score=0.30,
            learning_score=0.50,
            meta_score=0.20,
        )
        assert 0.30 <= score.fci < 0.45
        assert score.level == "aware"
        assert score.status == "HEALTHY"

    def test_compute_conscious(self):
        fci = FleetConsciousnessIndex()
        score = fci.compute(
            room_phi_score=0.60,
            attention_score=0.40,
            learning_score=0.60,
            meta_score=0.30,
        )
        assert 0.45 <= score.fci < 0.60
        assert score.level == "conscious"

    def test_compute_self_aware(self):
        fci = FleetConsciousnessIndex()
        score = fci.compute(
            room_phi_score=0.80,
            attention_score=0.60,
            learning_score=0.80,
            meta_score=0.50,
        )
        assert 0.60 <= score.fci < 0.75
        assert score.level == "self-aware"

    def test_compute_transcendent(self):
        fci = FleetConsciousnessIndex()
        score = fci.compute(
            room_phi_score=1.0,
            attention_score=1.0,
            learning_score=1.0,
            meta_score=1.0,
        )
        assert score.fci == 1.0
        assert score.level == "transcendent"

    def test_clamping(self):
        fci = FleetConsciousnessIndex()
        score = fci.compute(
            room_phi_score=2.0,
            attention_score=2.0,
            learning_score=2.0,
            meta_score=2.0,
        )
        assert score.fci == 1.0

    def test_custom_weights(self):
        fci = FleetConsciousnessIndex(weights={
            "room_phi": 0.25,
            "attention": 0.25,
            "learning": 0.25,
            "meta": 0.25,
        })
        score = fci.compute(
            room_phi_score=0.4,
            attention_score=0.4,
            learning_score=0.4,
            meta_score=0.4,
        )
        assert score.fci == 0.4

    def test_render_text(self):
        fci = FleetConsciousnessIndex()
        score = fci.compute(
            room_phi_score=0.50,
            attention_score=0.30,
            learning_score=0.50,
            meta_score=0.20,
        )
        text = fci.render_text(score)
        assert "FLEET CONSCIOUSNESS DASHBOARD" in text
        assert "AWARE" in text
        assert "✓ HEALTHY" in text

    def test_render_json(self):
        fci = FleetConsciousnessIndex()
        score = fci.compute(
            room_phi_score=0.50,
            attention_score=0.30,
            learning_score=0.50,
            meta_score=0.20,
        )
        json_text = fci.render_json(score)
        import json
        data = json.loads(json_text)
        assert "fci" in data
        assert "level" in data

    def test_render_oneline(self):
        fci = FleetConsciousnessIndex()
        score = fci.compute(
            room_phi_score=0.50,
            attention_score=0.30,
            learning_score=0.50,
            meta_score=0.20,
        )
        line = fci.render_oneline(score)
        assert "FCI:" in line
        assert "aware" in line
        assert "HEALTHY" in line

    def test_from_fleet_metrics(self):
        score = FleetConsciousnessIndex.from_fleet_metrics(
            rooms=30,
            total_rooms_capacity=100,
            active_agents=5,
            total_agents=10,
            positive_learning_passes=8,
            total_learning_passes=10,
            meta_tile_depth_sum=20.0,
            total_tiles=50,
        )
        assert score.fci == pytest.approx(
            0.30 * 0.40 + 0.50 * 0.20 + 0.80 * 0.25 + (20 / 50 / 10) * 0.15,
            abs=0.01,
        )
        assert score.level == "aware"

    def test_from_fleet_metrics_zero_division(self):
        score = FleetConsciousnessIndex.from_fleet_metrics(
            rooms=0,
            total_rooms_capacity=0,
            active_agents=0,
            total_agents=0,
            positive_learning_passes=0,
            total_learning_passes=0,
            meta_tile_depth_sum=0.0,
            total_tiles=0,
        )
        assert score.fci == 0.0
        assert score.level == "dormant"
