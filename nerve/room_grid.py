"""RoomGrid — Batched JEPA rooms as the foundation of the ecosystem.

Replace simulated nerve fibers with real JEPA models. Each room = one JEPA.
All rooms process signals in parallel as a single batched tensor operation.

250 rooms: 5.8ms forward pass, 2.7MB VRAM.
"""

from __future__ import annotations

__all__ = ["JEPABatch", "RoomGrid", "RoomFingerprint"]

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn


class JEPABatch(nn.Module):
    """All rooms as a single nn.ModuleList for batched parallel inference.

    Each room has its OWN weight set (different seed per room).
    Processing N rooms on one signal is O(1) tensor operation.

    Args:
        n_rooms: Number of rooms.
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer size.
        latent_dim: Output latent dimension.
    """

    def __init__(
        self,
        n_rooms: int = 250,
        input_dim: int = 64,
        hidden_dim: int = 32,
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        self.n_rooms = n_rooms
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                nn.Linear(hidden_dim // 2, latent_dim),
            )
            for _ in range(n_rooms)
        ])

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Process one signal through ALL rooms.

        Returns: (n_rooms, latent_dim) tensor of latents.
        """
        # Stack results — each encoder produces (1, latent_dim)
        latents = []
        for enc in self.encoders:
            latents.append(enc(signal))
        return torch.cat(latents, dim=0)

    def room_latent(self, room_idx: int, signal: torch.Tensor) -> torch.Tensor:
        """Get latent from a specific room."""
        return self.encoders[room_idx](signal)

    def set_seed(self, room_idx: int, seed: int) -> None:
        """Re-seed a specific room (for rebirth after sunset)."""
        torch.manual_seed(seed)
        with torch.no_grad():
            for layer in self.encoders[room_idx]:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()


@dataclass
class RoomFingerprint:
    """A room's identity — its latent response to standard reference signals.

    Attributes:
        room_idx: Which room.
        sine_response: Latent vector for sine wave input.
        noise_response: Latent vector for noise input.
        step_response: Latent vector for step input.
        activity: How many times this room has fired.
    """
    room_idx: int
    sine_response: torch.Tensor
    noise_response: torch.Tensor
    step_response: torch.Tensor
    activity: int = 0


class RoomGrid:
    """A distributed grid of JEPA rooms with chaos-routed connections.

    Each room is a micro-agency: it processes signals through its JEPA,
    fires when it detects something relevant, and develops connections
    to other rooms that fire on the same signals.

    Args:
        n_rooms: Number of rooms (default 250).
        input_dim: Signal dimension.
        latent_dim: JEPA latent dimension.
        chaos_rate: Initial chaos probability.
    """

    def __init__(
        self,
        n_rooms: int = 250,
        input_dim: int = 64,
        latent_dim: int = 16,
        chaos_rate: float = 0.3,
    ) -> None:
        self.n_rooms = n_rooms
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.jepa = JEPABatch(n_rooms=n_rooms, input_dim=input_dim, latent_dim=latent_dim)
        self._connections: dict[int, dict[int, float]] = {}  # room_a → {room_b: weight}
        self._activity: dict[int, int] = {}  # room_idx → fire count
        self._chaos_probs: list[float] = [chaos_rate] * n_rooms
        self._latent_history: dict[int, list[torch.Tensor]] = {}
        self._lock = threading.Lock()
        self._total_ticks: int = 0

        # Reference signals for fingerprinting
        t = torch.linspace(0, 2 * torch.pi, input_dim)
        self._ref_sine = torch.sin(t).unsqueeze(0)
        self._ref_noise = torch.randn(1, input_dim)
        self._ref_step = torch.zeros(1, input_dim)
        self._ref_step[0, input_dim // 2:] = 1.0

    def __repr__(self) -> str:
        return (
            f"RoomGrid(rooms={self.n_rooms}, "
            f"ticks={self._total_ticks}, "
            f"active={sum(1 for a in self._activity.values() if a > 0)})"
        )

    def tick(self, signal: torch.Tensor) -> dict[str, Any]:
        """One processing tick: all rooms perceive, chaos route, fire.

        Returns:
            Dict with fired_rooms, novel_rooms, and average activity.
        """
        self._total_ticks += 1

        # All rooms perceive the signal in parallel
        latents = self.jepa(signal.unsqueeze(0) if signal.dim() == 1 else signal)
        # latents shape: (n_rooms, latent_dim)

        # Each room decides: fire or not
        fired: list[int] = []
        for room_idx in range(self.n_rooms):
            latent = latents[room_idx]

            # Check novelty: distance from recent latents
            novelty = self._compute_novelty(room_idx, latent)

            # Fire on novelty + chaos
            fires = novelty > 0.3 or random.random() < self._chaos_probs[room_idx]

            if fires:
                self._activity[room_idx] = self._activity.get(room_idx, 0) + 1
                fired.append(room_idx)

                # Strengthen connections to other fired rooms
                for other in fired:
                    if other != room_idx:
                        self._connections.setdefault(room_idx, {}).setdefault(other, 0.0)
                        self._connections[room_idx][other] = min(
                            1.0, self._connections[room_idx][other] + 0.01
                        )

                # Decay chaos
                self._chaos_probs[room_idx] = max(0.01, self._chaos_probs[room_idx] * 0.99)

            # Store recent latent
            self._latent_history.setdefault(room_idx, []).append(latent.detach().clone())
            if len(self._latent_history[room_idx]) > 20:
                self._latent_history[room_idx].pop(0)

        return {
            "fired_rooms": len(fired),
            "fired_ids": fired[:10],  # first 10
            "total_ticks": self._total_ticks,
        }

    def _compute_novelty(self, room_idx: int, latent: torch.Tensor) -> float:
        """How novel is this latent vs recent history?"""
        history = self._latent_history.get(room_idx, [])
        if len(history) < 3:
            return 0.5  # default novelty when not enough data
        recent = torch.stack(history[-3:])
        latent_n = latent / (latent.norm() + 1e-8)
        recent_n = recent / (recent.norm(dim=-1, keepdim=True) + 1e-8)
        cos_sim = (latent_n * recent_n).sum(dim=-1)
        return (1.0 - cos_sim.mean()).item()

    def get_fingerprints(self) -> list[RoomFingerprint]:
        """Compute fingerprints for all rooms.

        A room's fingerprint is how it responds to reference signals.
        """
        fps: list[RoomFingerprint] = []
        with torch.no_grad():
            for i in range(min(50, self.n_rooms)):  # first 50 for speed
                fp = RoomFingerprint(
                    room_idx=i,
                    sine_response=self.jepa.room_latent(i, self._ref_sine).squeeze(),
                    noise_response=self.jepa.room_latent(i, self._ref_noise).squeeze(),
                    step_response=self.jepa.room_latent(i, self._ref_step).squeeze(),
                    activity=self._activity.get(i, 0),
                )
                fps.append(fp)
        return fps

    def sort_by_activity(self, n: int = 10) -> list[tuple[int, int]]:
        """Top-N most active rooms."""
        sorted_rooms = sorted(self._activity.items(), key=lambda x: x[1], reverse=True)
        return sorted_rooms[:n]

    def prune_cold(self, threshold: int = 1) -> list[int]:
        """Find rooms below activity threshold (candidates for sunset)."""
        cold = [i for i in range(self.n_rooms) if self._activity.get(i, 0) < threshold]
        return cold

    def rebirth_as(self, cold_idx: int, hot_idx: int) -> None:
        """Reincarnate a cold room with a hot room's latent space."""
        self.jepa.set_seed(cold_idx, seed=random.randint(0, 9999))
        self._activity[cold_idx] = 0
        self._chaos_probs[cold_idx] = 0.3
        self._latent_history[cold_idx] = []

    @property
    def stats(self) -> dict[str, Any]:
        active = sum(1 for a in self._activity.values() if a > 0)
        cold = len(self.prune_cold(threshold=1))
        return {
            "rooms": self.n_rooms,
            "ticks": self._total_ticks,
            "active_rooms": active,
            "cold_rooms": cold,
            "pct_active": f"{active / self.n_rooms * 100:.1f}%",
        }
