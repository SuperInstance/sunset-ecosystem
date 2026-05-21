"""Tests for JEPA nerve fibers and chaos rooms."""

import pytest
import torch

from nerve.jepa import (
    MinimalJEPA,
    JEPAFiber,
    ChaosRoom,
    ChaosMessage,
    JEPASwarm,
)


class TestMinimalJEPA:
    def test_forward_shape(self):
        model = MinimalJEPA(input_dim=64, latent_dim=16)
        x = torch.randn(4, 64)
        z, z_pred = model(x)
        assert z.shape == (4, 16)
        assert z_pred.shape == (4, 16)

    def test_encode(self):
        model = MinimalJEPA(input_dim=64, latent_dim=16)
        x = torch.randn(2, 64)
        z = model.encode(x)
        assert z.shape == (2, 16)

    def test_params(self):
        model = MinimalJEPA(input_dim=64, hidden_dim=32, latent_dim=16)
        assert sum(p.numel() for p in model.parameters()) == 3424

    def test_latent_distance(self):
        model = MinimalJEPA()
        a = torch.randn(1, 16)
        b = torch.randn(1, 16)
        d = model.latent_distance(a, b)
        assert 0.0 <= d <= 2.0


class TestJEPAFiber:
    def test_perceive(self):
        f = JEPAFiber(fiber_id="test-1", seed=42)
        signal = torch.randn(1, 64)
        result = f.perceive(signal)
        assert result["fiber_id"] == "test-1"
        assert result["latent"].shape == (1, 16)
        assert "novelty" in result
        assert "chaos_triggered" in result

    def test_different_seeds_different_latents(self):
        f1 = JEPAFiber(fiber_id="a", seed=42)
        f2 = JEPAFiber(fiber_id="b", seed=99)
        signal = torch.randn(1, 64)
        r1 = f1.perceive(signal)
        r2 = f2.perceive(signal)
        # Different seeds should produce different latents
        dist = (r1["latent"] - r2["latent"]).norm().item()
        assert dist > 0.001, "Different seeds should produce different latents"

    def test_novelty_increases(self):
        f = JEPAFiber(fiber_id="test", seed=42, chaos_rate=0.0)
        # Same signal repeated — novelty should decrease
        signal = torch.ones(1, 64)
        novelties = []
        for _ in range(3):
            r = f.perceive(signal)
            novelties.append(r["novelty"])
        # First should be higher than last (pattern becomes familiar)
        assert novelties[-1] <= novelties[0] or abs(novelties[-1] - novelties[0]) < 0.5

    def test_chaos_rate(self):
        f = JEPAFiber(fiber_id="t", seed=42, chaos_rate=0.5)
        signal = torch.randn(1, 64)
        triggered = 0
        for _ in range(20):
            r = f.perceive(signal)
            if r["chaos_triggered"]:
                triggered += 1
        # With 0.5 rate over 20 trials, should fire some
        assert triggered > 0


class TestChaosRoom:
    def test_receive_fires_on_novelty(self):
        room = ChaosRoom(room_id="r1", problem_statement="test prob")
        msg = ChaosMessage(
            source_fiber="f1",
            latent=torch.randn(1, 16),
            novelty=0.8,  # high novelty = fire
            chaos_path="path-1",
        )
        assert room.receive(msg) is True

    def test_connection_strengthening(self):
        room = ChaosRoom(room_id="r1", problem_statement="test")
        for _ in range(5):
            msg = ChaosMessage(
                source_fiber="f1",
                latent=torch.randn(1, 16),
                novelty=0.9,
                chaos_path="path-1",
            )
            room.receive(msg)
        assert room._connections.get("f1", 0.0) > 0.1

    def test_chaos_decays(self):
        room = ChaosRoom(room_id="r1", problem_statement="test", chaos_decay=0.95)
        initial = room._chaos_prob
        # Fire a lot — chaos should decrease
        for _ in range(20):
            msg = ChaosMessage(
                source_fiber="f1",
                latent=torch.randn(1, 16),
                novelty=0.9,
                chaos_path="path-1",
            )
            room.receive(msg)
        assert room._chaos_prob <= initial


class TestJEPASwarm:
    def test_tick(self):
        swarm = JEPASwarm(n_fibers=6, n_rooms=3, input_dim=64, latent_dim=16)
        signal = torch.randn(1, 64)
        results = swarm.tick(signal)
        # Should be a dict of room_id → list of fired messages
        assert isinstance(results, dict)

    def test_multiple_ticks(self):
        swarm = JEPASwarm(n_fibers=6, n_rooms=3)
        for _ in range(5):
            swarm.tick(torch.randn(1, 64))
        assert swarm._signal_count == 5

    def test_distill_candidates(self):
        swarm = JEPASwarm(n_fibers=12, n_rooms=4)
        for _ in range(20):
            swarm.tick(torch.randn(1, 64))
        # With 20 ticks, some rooms should have fired enough
        candidates = swarm.distill_candidates(min_fires=1)
        assert isinstance(candidates, list)

    def test_repr(self):
        swarm = JEPASwarm(n_fibers=6, n_rooms=2)
        assert "JEPASwarm" in repr(swarm)

    def test_stats(self):
        swarm = JEPASwarm(n_fibers=6, n_rooms=2)
        swarm.tick(torch.randn(1, 64))
        s = swarm.stats
        assert s["fibers"] == 6
        assert s["rooms"] == 2
        assert s["signals_processed"] == 1
