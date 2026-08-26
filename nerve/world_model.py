"""LeWorld — A LWM-style wandering JEPA that calibrates room latents.

One small mobile JEPA (same 3.4K params) bounces between rooms.
It sees room A → predicts room B → measures prediction error.
If error is large, room A's latent has drifted from consensus.
The calibrator corrects room A, keeping all room-JEPAs synchronized.

This replaces the need for a global loss function or central optimizer.
"""

from __future__ import annotations

__all__ = [
    "WanderingJEPA",
    "Calibrator",
    "CalibrationReport",
    "WorldModel",
]

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn


class WanderingJEPA(nn.Module):
    """A small mobile JEPA that predicts one room's latent from another's.

    Travels between rooms: observes room A's response to a signal,
    predicts what room B's response will be.

    Args:
        latent_dim: The shared latent dimension across rooms.
        hidden_dim: Hidden layer size for the predictor.
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_a: torch.Tensor) -> torch.Tensor:
        """Predict room B's latent from room A's latent."""
        return self.predictor(z_a)

    def predict_distance(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """How far is prediction from reality?"""
        z_pred = self.forward(z_a)
        return (z_pred - z_b).norm(p=2, dim=-1).mean()


@dataclass
class CalibrationReport:
    """Result of one calibration bounce.

    Attributes:
        room_a: Source room.
        room_b: Target room.
        pred_error: Prediction error (latent distance).
        drift_score: How much room A's latent has drifted (0-1).
        corrected: Whether room A was recalibrated.
    """

    room_a: int
    room_b: int
    pred_error: float
    drift_score: float
    corrected: bool = False


class Calibrator:
    """Bounces a wandering JEPA between rooms to detect and fix latent drift.

    Args:
        n_rooms: Total rooms in the grid.
        latent_dim: JEPA latent dimension.
        drift_threshold: Prediction error above this triggers correction.
    """

    def __init__(
        self,
        n_rooms: int = 250,
        latent_dim: int = 16,
        drift_threshold: float = 0.3,
    ) -> None:
        self.wanderer = WanderingJEPA(latent_dim=latent_dim)
        self.n_rooms = n_rooms
        self.latent_dim = latent_dim
        self.drift_threshold = drift_threshold
        self._history: list[CalibrationReport] = []
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"Calibrator(wanderer={self.wanderer}, "
            f"history={len(self._history)}, "
            f"threshold={self.drift_threshold})"
        )

    def bounce(
        self,
        room_a: int,
        z_a: torch.Tensor,
        room_b: int,
        z_b: torch.Tensor,
        learn: bool = True,
    ) -> CalibrationReport:
        """One calibration bounce: A→B.

        Args:
            room_a: Source room index.
            z_a: Source room's latent.
            room_b: Target room index.
            z_b: Target room's latent (ground truth).
            learn: Whether to update the wanderer's weights.

        Returns:
            CalibrationReport with drift analysis.
        """
        with torch.no_grad():
            z_pred = self.wanderer(z_a)
            error = (z_pred - z_b).norm(p=2, dim=-1).mean().item()
            drift = min(1.0, error / self.drift_threshold)
            corrected = drift > self.drift_threshold

        if learn and corrected:
            # Fine-tune wanderer on this pair
            self._learn_step(z_a, z_b)

        report = CalibrationReport(
            room_a=room_a,
            room_b=room_b,
            pred_error=error,
            drift_score=drift,
            corrected=corrected,
        )

        with self._lock:
            self._history.append(report)

        return report

    def _learn_step(self, z_a: torch.Tensor, z_b: torch.Tensor) -> None:
        """One gradient step for the wanderer, no optimizer object."""
        # Manual gradient step avoids torch.optim.SGD constructor
        # which triggers beartype crashes on onnx import
        with torch.enable_grad():
            self.wanderer.zero_grad()
            z_pred = self.wanderer(z_a)
            loss = (z_pred - z_b).pow(2).mean()
            loss.backward()
            with torch.no_grad():
                for p in self.wanderer.parameters():
                    if p.grad is not None:
                        p.data.add_(p.grad, alpha=-0.001)

    def bounce_random_pair(
        self,
        room_latents: dict[int, torch.Tensor],
    ) -> Optional[CalibrationReport]:
        """Pick a random room pair and bounce."""
        if len(room_latents) < 2:
            return None
        room_ids = sorted(room_latents.keys())
        a, b = random.sample(room_ids, 2)
        return self.bounce(a, room_latents[a], b, room_latents[b])

    @property
    def avg_drift(self) -> float:
        """Average drift across all calibration bounces."""
        if not self._history:
            return 0.0
        return sum(r.drift_score for r in self._history) / len(self._history)

    @property
    def correction_rate(self) -> float:
        """Fraction of bounces that triggered correction."""
        if not self._history:
            return 0.0
        return sum(1 for r in self._history if r.corrected) / len(self._history)


class WorldModel:
    """The complete world model: room grid + wandering calibrator.

    Rooms perceive signals through their JEPAs.
    The wanderer bounces between rooms, detecting drift and correcting.
    Over time, all rooms converge to a shared latent space WITHOUT
    a central optimizer.

    Args:
        n_rooms: Number of rooms.
        latent_dim: JEPA latent dimension.
    """

    def __init__(self, n_rooms: int = 100, latent_dim: int = 16) -> None:
        self.n_rooms = n_rooms
        self.latent_dim = latent_dim
        self.room_jepas: list[nn.Module] = []

        # Each room has its own tiny JEPA encoder
        for i in range(n_rooms):
            torch.manual_seed(42 + i * 7)
            enc = nn.Sequential(
                nn.Linear(latent_dim * 4, latent_dim * 2),
                nn.ReLU(),
                nn.Linear(latent_dim * 2, latent_dim),
            )
            self.room_jepas.append(enc)

        self.calibrator = Calibrator(n_rooms=n_rooms, latent_dim=latent_dim)
        self._bounces: int = 0

    def __repr__(self) -> str:
        return (
            f"WorldModel(rooms={self.n_rooms}, "
            f"latent={self.latent_dim}d, "
            f"bounces={self._bounces})"
        )

    def tick(self, signal: torch.Tensor) -> dict[str, Any]:
        """One tick: rooms process signal, wanderer bounces.

        Returns:
            Dict with corrections and drift stats.
        """
        # Each room encodes the signal
        room_latents: dict[int, torch.Tensor] = {}
        for i, enc in enumerate(self.room_jepas):
            with torch.no_grad():
                z = enc(signal)
                room_latents[i] = z

        # Wanderer bounces between random pairs
        corrections = 0
        for _ in range(min(10, self.n_rooms)):
            report = self.calibrator.bounce_random_pair(room_latents)
            if report and report.corrected:
                corrections += 1
            self._bounces += 1

        return {
            "bounces": self._bounces,
            "corrections_this_tick": corrections,
            "avg_drift": self.calibrator.avg_drift,
            "correction_rate": self.calibrator.correction_rate,
        }
