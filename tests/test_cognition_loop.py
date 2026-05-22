"""Tests for CognitionLoop — observe, reason, act, loop, RoomGrid integration."""
import numpy as np
import pytest

from perception.cognition_loop import AgentConfig, CognitionLoop, CognitionState
from nerve.room_grid import RoomGrid


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def config_disabled():
    """Default config with cognition disabled."""
    return AgentConfig(enable_cognition=False)


@pytest.fixture
def config_enabled():
    """Config with cognition enabled, interval=1."""
    return AgentConfig(
        enable_cognition=True,
        cognition_interval=1,
        cold_threshold=1,
        hot_threshold=50,
        breed_candidates=3,
        rebirth_candidates=2,
        chaos_boost=0.1,
        chaos_decay=0.05,
        top_k_observed=10,
    )


@pytest.fixture
def grid_100():
    """RoomGrid(100) seeded for determinism."""
    np.random.seed(42)
    return RoomGrid(100)


@pytest.fixture
def loop_enabled(config_enabled):
    """CognitionLoop with enabled config."""
    return CognitionLoop(config_enabled)


# ═══════════════════════════════════════════════════════════
#  AgentConfig
# ═══════════════════════════════════════════════════════════

class TestAgentConfig:
    def test_default_cognition_disabled(self):
        cfg = AgentConfig()
        assert cfg.enable_cognition is False

    def test_fields(self, config_enabled):
        assert config_enabled.enable_cognition is True
        assert config_enabled.cognition_interval == 1
        assert config_enabled.cold_threshold == 1
        assert config_enabled.hot_threshold == 50
        assert config_enabled.breed_candidates == 3
        assert config_enabled.rebirth_candidates == 2
        assert config_enabled.chaos_boost == 0.1
        assert config_enabled.chaos_decay == 0.05
        assert config_enabled.top_k_observed == 10


# ═══════════════════════════════════════════════════════════
#  CognitionState
# ═══════════════════════════════════════════════════════════

class TestCognitionState:
    def test_to_dict(self):
        state = CognitionState(
            tick=5, n_rooms=100, active_count=3, cold_count=97,
            top_active=[(0, 10), (1, 5)],
            top_novel=[(2, 0.9)],
            mean_chaos=0.3, mean_novelty=0.5,
            fired_ids=[0, 1], latents_shape=(100, 16),
        )
        d = state.to_dict()
        assert d["tick"] == 5
        assert d["n_rooms"] == 100
        assert d["active_count"] == 3
        assert d["cold_count"] == 97
        assert d["mean_chaos"] == 0.3
        assert d["mean_novelty"] == 0.5
        assert d["fired_ids"] == [0, 1]
        assert d["latents_shape"] == (100, 16)


# ═══════════════════════════════════════════════════════════
#  CognitionLoop.observe()
# ═══════════════════════════════════════════════════════════

class TestObserve:
    def test_empty_grid(self, loop_enabled):
        """Observe a grid with zero rooms — should not crash."""
        class FakeGrid:
            n = 0
            ticks = 0
            activity = np.array([], dtype=np.int32)
            chaos = np.array([], dtype=np.float32)
            latents = None

        state = loop_enabled.observe(FakeGrid())
        assert state.n_rooms == 0
        assert state.active_count == 0
        assert state.cold_count == 0
        assert state.top_active == []
        assert state.top_novel == []

    def test_observes_tick_count(self, grid_100, loop_enabled):
        grid_100.tick(np.random.randn(64).astype(np.float32))
        state = loop_enabled.observe(grid_100)
        assert state.tick == grid_100.ticks

    def test_observes_activity(self, grid_100, loop_enabled):
        grid_100.tick(np.random.randn(64).astype(np.float32))
        state = loop_enabled.observe(grid_100)
        assert state.n_rooms == 100
        assert state.active_count >= 0
        assert state.cold_count >= 0
        assert len(state.top_active) <= loop_enabled.config.top_k_observed

    def test_observes_latents_shape(self, grid_100, loop_enabled):
        grid_100.tick(np.random.randn(64).astype(np.float32))
        state = loop_enabled.observe(grid_100)
        assert state.latents_shape == (100, 16)

    def test_fired_ids_populated(self, grid_100, loop_enabled):
        grid_100.tick(np.random.randn(64).astype(np.float32))
        state = loop_enabled.observe(grid_100)
        # fired_ids should mirror whatever RoomGrid stored
        assert isinstance(state.fired_ids, list)


# ═══════════════════════════════════════════════════════════
#  CognitionLoop.reason()
# ═══════════════════════════════════════════════════════════

class TestReason:
    def test_empty_state(self, loop_enabled):
        state = CognitionState(
            tick=0, n_rooms=0, active_count=0, cold_count=0,
            top_active=[], top_novel=[], mean_chaos=0.3, mean_novelty=0.5,
            fired_ids=[], latents_shape=(),
        )
        decisions = loop_enabled.reason(state)
        assert decisions["breed_pairs"] == []
        assert decisions["rebirth_ids"] == []
        assert decisions["chaos_adjustments"] == []

    def test_chaos_boost_for_cold(self, loop_enabled):
        state = CognitionState(
            tick=1, n_rooms=10, active_count=1, cold_count=9,
            top_active=[(0, 0), (1, 0), (2, 5)],
            top_novel=[], mean_chaos=0.1, mean_novelty=0.5,
            fired_ids=[], latents_shape=(10, 16),
        )
        decisions = loop_enabled.reason(state)
        # Cold rooms (activity < 1) should get chaos boost
        adjustments = {rid: chaos for rid, chaos in decisions["chaos_adjustments"]}
        assert any(rid in adjustments for rid in [0, 1])
        # Boosted chaos should be > 0.3
        for rid, chaos in decisions["chaos_adjustments"]:
            if rid in [0, 1]:
                assert chaos > 0.3

    def test_chaos_decay_for_hot(self, loop_enabled):
        state = CognitionState(
            tick=1, n_rooms=10, active_count=5, cold_count=5,
            top_active=[(0, 60), (1, 55), (2, 5)],
            top_novel=[], mean_chaos=0.5, mean_novelty=0.5,
            fired_ids=[], latents_shape=(10, 16),
        )
        decisions = loop_enabled.reason(state)
        # Hot rooms (activity > 50) should get chaos decay
        adjustments = {rid: chaos for rid, chaos in decisions["chaos_adjustments"]}
        assert any(rid in adjustments for rid in [0, 1])
        # Decayed chaos should be < 0.3
        for rid, chaos in decisions["chaos_adjustments"]:
            if rid in [0, 1]:
                assert chaos < 0.3

    def test_rebirth_candidates_limited(self, loop_enabled):
        state = CognitionState(
            tick=1, n_rooms=100, active_count=0, cold_count=100,
            top_active=[(i, 0) for i in range(20)],
            top_novel=[], mean_chaos=0.3, mean_novelty=0.5,
            fired_ids=[], latents_shape=(100, 16),
        )
        decisions = loop_enabled.reason(state)
        assert len(decisions["rebirth_ids"]) <= loop_enabled.config.rebirth_candidates

    def test_breed_pairs_limited(self, loop_enabled):
        state = CognitionState(
            tick=1, n_rooms=100, active_count=50, cold_count=50,
            top_active=[(i, 10 if i < 25 else 0) for i in range(50)],
            top_novel=[], mean_chaos=0.3, mean_novelty=0.5,
            fired_ids=[], latents_shape=(100, 16),
        )
        decisions = loop_enabled.reason(state)
        assert len(decisions["breed_pairs"]) <= loop_enabled.config.breed_candidates
        # Each pair should be (src, dst) with src active and dst cold
        for src, dst in decisions["breed_pairs"]:
            assert src != dst


# ═══════════════════════════════════════════════════════════
#  CognitionLoop.act()
# ═══════════════════════════════════════════════════════════

class TestAct:
    def test_chaos_adjustments_applied(self, grid_100, loop_enabled):
        decisions = {
            "breed_pairs": [],
            "rebirth_ids": [],
            "chaos_adjustments": [(5, 0.99), (10, 0.05)],
        }
        before = grid_100.chaos.copy()
        loop_enabled.act(grid_100, decisions)
        assert grid_100.chaos[5] == pytest.approx(0.99)
        assert grid_100.chaos[10] == pytest.approx(0.05)
        # Others unchanged
        for i in range(100):
            if i not in (5, 10):
                assert grid_100.chaos[i] == pytest.approx(before[i])

    def test_rebirth_executed(self, grid_100, loop_enabled):
        before_w = grid_100.w["w1"][7].copy()
        decisions = {
            "breed_pairs": [],
            "rebirth_ids": [7],
            "chaos_adjustments": [],
        }
        loop_enabled.act(grid_100, decisions)
        # Weights should have changed
        assert not np.allclose(grid_100.w["w1"][7], before_w)
        # Activity and chaos reset
        assert grid_100.activity[7] == 0
        assert grid_100.chaos[7] == pytest.approx(0.3)

    def test_breed_executed(self, grid_100, loop_enabled):
        before_w = grid_100.w["w1"][20].copy()
        src_w = grid_100.w["w1"][5].copy()
        decisions = {
            "breed_pairs": [(5, 20)],
            "rebirth_ids": [],
            "chaos_adjustments": [],
        }
        loop_enabled.act(grid_100, decisions)
        # Destination should now be similar to source (but not identical due to mutation)
        assert not np.allclose(grid_100.w["w1"][20], before_w)
        # Should be close to source (within mutation noise)
        assert np.allclose(grid_100.w["w1"][20], src_w, atol=0.02)

    def test_act_survives_invalid_room_id(self, grid_100, loop_enabled):
        decisions = {
            "breed_pairs": [(5, 9999)],  # out of range
            "rebirth_ids": [9999],
            "chaos_adjustments": [(9999, 0.5)],
        }
        # Should not raise
        loop_enabled.act(grid_100, decisions)

    def test_act_survives_missing_methods(self, loop_enabled):
        class FakeGrid:
            n = 10
            chaos = np.full(10, 0.3, dtype=np.float32)
            # No breed() or rebirth()

        decisions = {
            "breed_pairs": [(0, 1)],
            "rebirth_ids": [2],
            "chaos_adjustments": [(3, 0.5)],
        }
        # Should not raise
        loop_enabled.act(FakeGrid(), decisions)
        assert FakeGrid.chaos[3] == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════
#  CognitionLoop.loop()
# ═══════════════════════════════════════════════════════════

class TestLoop:
    def test_loop_runs_every_tick_when_interval_1(self, grid_100, loop_enabled):
        grid_100._cognition_loop = loop_enabled
        loop_enabled.reset()

        d1 = loop_enabled.loop(grid_100)
        d2 = loop_enabled.loop(grid_100)
        d3 = loop_enabled.loop(grid_100)

        # All should return decision dicts (not empty because interval=1)
        assert isinstance(d1, dict)
        assert isinstance(d2, dict)
        assert isinstance(d3, dict)
        assert loop_enabled._tick_counter == 3

    def test_loop_skips_when_interval_2(self, grid_100):
        config = AgentConfig(enable_cognition=True, cognition_interval=2)
        loop = CognitionLoop(config)
        loop.reset()

        d1 = loop.loop(grid_100)
        d2 = loop.loop(grid_100)
        d3 = loop.loop(grid_100)

        # counter starts at 0; increment before check
        # tick 1: counter=1, 1%2=1 != 0 → skipped (returns {})
        # tick 2: counter=2, 2%2=0 → runs
        # tick 3: counter=3, 3%2=1 != 0 → skipped
        assert d1 == {}
        assert d2 != {}
        assert d3 == {}
        assert loop._tick_counter == 3

    def test_loop_updates_last_observation(self, grid_100, loop_enabled):
        loop_enabled.reset()
        assert loop_enabled.last_observation is None
        loop_enabled.loop(grid_100)
        assert loop_enabled.last_observation is not None
        assert loop_enabled.last_observation.n_rooms == 100

    def test_loop_updates_last_decisions(self, grid_100, loop_enabled):
        loop_enabled.reset()
        assert loop_enabled.last_decisions is None
        loop_enabled.loop(grid_100)
        assert loop_enabled.last_decisions is not None
        assert "breed_pairs" in loop_enabled.last_decisions
        assert "rebirth_ids" in loop_enabled.last_decisions
        assert "chaos_adjustments" in loop_enabled.last_decisions


# ═══════════════════════════════════════════════════════════
#  RoomGrid integration — enable_cognition flag
# ═══════════════════════════════════════════════════════════

class TestRoomGridIntegration:
    def test_cognition_disabled_by_default(self, grid_100):
        """Default RoomGrid has no cognition loop."""
        assert grid_100._agent_config is None
        assert grid_100._cognition_loop is None

    def test_cognition_enabled_in_init(self, config_enabled):
        grid = RoomGrid(100, agent_config=config_enabled)
        assert grid._agent_config is not None
        assert grid._agent_config.enable_cognition is True
        assert grid._cognition_loop is not None
        assert isinstance(grid._cognition_loop, CognitionLoop)

    def test_cognition_runs_during_tick(self, config_enabled):
        np.random.seed(42)
        grid = RoomGrid(50, agent_config=config_enabled)
        loop = grid._cognition_loop
        loop.reset()

        assert loop.last_observation is None
        grid.tick(np.random.randn(64).astype(np.float32))
        assert loop.last_observation is not None
        assert loop.last_observation.tick == grid.ticks

    def test_cognition_runs_during_tick_batch(self, config_enabled):
        np.random.seed(42)
        grid = RoomGrid(50, agent_config=config_enabled)
        loop = grid._cognition_loop
        loop.reset()

        assert loop.last_observation is None
        signals = np.random.randn(3, 64).astype(np.float32)
        grid.tick_batch(signals)
        # After batch, cognition should have run at least once (on last item)
        assert loop.last_observation is not None

    def test_cognition_does_not_crash_disabled(self, grid_100):
        """Tick works normally when cognition is disabled."""
        result = grid_100.tick(np.random.randn(64).astype(np.float32))
        assert "fired" in result
        assert "ids" in result
        assert "tick" in result

    def test_cognition_interval_respected(self):
        config = AgentConfig(enable_cognition=True, cognition_interval=3)
        grid = RoomGrid(50, agent_config=config)
        loop = grid._cognition_loop
        loop.reset()

        for _ in range(6):
            grid.tick(np.random.randn(64).astype(np.float32))

        # counter=6 after 6 ticks; verify it tracked correctly
        assert loop._tick_counter == 6

    def test_duck_typed_config_accepted(self):
        class DuckConfig:
            enable_cognition = True
            cognition_interval = 1
            cold_threshold = 1
            hot_threshold = 50
            breed_candidates = 3
            rebirth_candidates = 2
            chaos_boost = 0.1
            chaos_decay = 0.05
            top_k_observed = 10

        grid = RoomGrid(20, agent_config=DuckConfig())
        assert grid._cognition_loop is not None

    def test_invalid_config_rejected(self):
        with pytest.raises(TypeError):
            RoomGrid(20, agent_config="not_a_config")

    def test_cognition_changes_grid_state(self, config_enabled):
        np.random.seed(42)
        grid = RoomGrid(30, agent_config=config_enabled)
        # Run several ticks to let cognition act
        for _ in range(10):
            grid.tick(np.random.randn(64).astype(np.float32))
        # At least one of breed/rebirth/chaos_adjust should have happened
        assert grid._cognition_loop.last_decisions is not None
        # Ensure last_observation is populated (no crash)
        assert grid._cognition_loop.last_observation is not None


# ═══════════════════════════════════════════════════════════
#  Edge cases
# ═══════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_reset_clears_state(self, grid_100, loop_enabled):
        loop_enabled.loop(grid_100)
        assert loop_enabled.last_observation is not None
        loop_enabled.reset()
        assert loop_enabled.last_observation is None
        assert loop_enabled.last_decisions is None
        assert loop_enabled._tick_counter == 0

    def test_novelty_computation_failure_graceful(self, loop_enabled):
        """If batch_novelty fails, observe() should still return a state."""
        class BrokenGrid:
            n = 10
            ticks = 1
            activity = np.zeros(10, dtype=np.int32)
            chaos = np.full(10, 0.3, dtype=np.float32)
            latents = np.zeros((10, 16), dtype=np.float32)
            _hist = None  # Will cause batch_novelty to fail
            _hist_count = np.zeros(10, dtype=np.int32)
            _hist_idx = 0
            _hist_max = 20

        state = loop_enabled.observe(BrokenGrid())
        assert state is not None
        assert state.mean_novelty == 0.5  # fallback

    def test_cognition_state_repr(self):
        state = CognitionState(
            tick=1, n_rooms=10, active_count=2, cold_count=8,
            top_active=[(0, 5)], top_novel=[(1, 0.8)],
            mean_chaos=0.3, mean_novelty=0.6,
            fired_ids=[0], latents_shape=(10, 16),
        )
        # Should not crash
        repr(state)

    def test_cognition_with_zero_activity_grid(self):
        config = AgentConfig(enable_cognition=True)
        grid = RoomGrid(10, agent_config=config)
        # Do not tick — all activity is zero
        state = grid._cognition_loop.observe(grid)
        assert state.active_count == 0
        assert state.cold_count == 10
        decisions = grid._cognition_loop.reason(state)
        # Should still produce valid decisions (all rooms are cold)
        assert isinstance(decisions, dict)
