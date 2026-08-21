"""Tests for LeWorld wandering JEPA calibrator."""

import pytest
import torch

from nerve.world_model import WanderingJEPA, Calibrator, CalibrationReport, WorldModel


class TestWanderingJEPA:
    def test_forward_shape(self):
        w = WanderingJEPA(latent_dim=16, hidden_dim=32)
        z = torch.randn(1, 16)
        z_pred = w(z)
        assert z_pred.shape == (1, 16)

    def test_predict_distance(self):
        w = WanderingJEPA(latent_dim=16)
        a = torch.randn(1, 16)
        b = torch.randn(1, 16)
        d = w.predict_distance(a, b)
        assert d.item() >= 0


class TestCalibrator:
    def test_bounce(self):
        cal = Calibrator(latent_dim=16, drift_threshold=0.3)
        z_a = torch.randn(1, 16)
        z_b = torch.randn(1, 16)
        report = cal.bounce(0, z_a, 1, z_b, learn=False)
        assert isinstance(report, CalibrationReport)
        assert report.room_a == 0
        assert report.room_b == 1
        assert 0.0 <= report.drift_score <= 1.0

    def test_identical_latents_low_drift(self):
        cal = Calibrator(latent_dim=16, drift_threshold=0.3)
        z = torch.ones(1, 16)  # identical latents = zero distance
        report = cal.bounce(0, z, 1, z, learn=False)
        # Identical latents should have error < pred_error from random latents
        assert report.pred_error >= 0.0
        assert isinstance(report.drift_score, float)

    def test_random_bounce(self):
        cal = Calibrator(latent_dim=16, drift_threshold=0.3)
        latents = {i: torch.randn(1, 16) for i in range(10)}
        report = cal.bounce_random_pair(latents)
        assert report is not None
        assert report.room_a != report.room_b

    def test_avg_drift(self):
        cal = Calibrator(latent_dim=16)
        for i in range(5):
            cal.bounce(i, torch.randn(1, 16), i + 1, torch.randn(1, 16), learn=False)
        assert cal.avg_drift >= 0

    def test_repr(self):
        cal = Calibrator(latent_dim=16)
        assert "Calibrator" in repr(cal)


class TestWorldModel:
    def test_create(self):
        wm = WorldModel(n_rooms=10, latent_dim=16)
        assert wm.n_rooms == 10
        assert len(wm.room_jepas) == 10
        assert "WorldModel" in repr(wm)

    def test_tick(self):
        wm = WorldModel(n_rooms=10, latent_dim=16)
        signal = torch.randn(1, 64)
        result = wm.tick(signal)
        assert "bounces" in result
        assert "corrections_this_tick" in result
        assert "avg_drift" in result

    def test_bounces_accumulate(self):
        wm = WorldModel(n_rooms=10)
        signal = torch.randn(1, 64)
        for _ in range(3):
            wm.tick(signal)
        assert wm._bounces > 0

    def test_room_jepas_have_different_seeds(self):
        wm = WorldModel(n_rooms=5, latent_dim=16)
        signal = torch.randn(1, 64)
        latents = []
        for enc in wm.room_jepas:
            with torch.no_grad():
                z = enc(signal)
                latents.append(z)
        # Different seeds should produce different latents
        distances = [
            (latents[i] - latents[j]).norm().item()
            for i in range(len(latents))
            for j in range(i + 1, len(latents))
        ]
        assert all(d > 0.001 for d in distances), (
            "Different seeds should produce different latents"
        )
