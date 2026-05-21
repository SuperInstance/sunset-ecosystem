"""JEPA Nerve — A V-JEPA-style sensory fiber with chaos-engineered routing.

Each fiber trains a tiny JEPA (3.4K params) to encode signals into latent
representations. Chaos-engineered routing means routes diversify through
stochastic exploration — agents don't converge to the same pattern.

The key insight: JEPA predicts one view from another. Two fibers processing
the same signal will learn DIFFERENT latent spaces. The chaos engine
compares these differences and routes novelty to the rooms that need it.
"""

from __future__ import annotations

__all__ = [
    "MinimalJEPA",
    "JEPAFiber",
    "ChaosRoom",
    "ChaosMessage",
    "JEPASwarm",
]

import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn


# ── Minimal JEPA Model ───────────────────────────────────────

class MinimalJEPA(nn.Module):
    """Tiny JEPA-style encoder: input → latent → predicted latent.

    Two views of the same signal should have close latent representations.
    Different fibers learn different latent spaces for the same signal domain.

    Args:
        input_dim: Size of input signal features.
        hidden_dim: Hidden layer size.
        latent_dim: Output latent dimension.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and predict. Returns (latent, predicted_latent)."""
        z = self.encoder(x)
        z_pred = self.predictor(z)
        return z, z_pred

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode only (no prediction)."""
        return self.encoder(x)

    def latent_distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine distance between two latent vectors."""
        a_n = a / (a.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        b_n = b / (b.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        cos_sim = (a_n * b_n).sum(dim=-1)
        return (1.0 - cos_sim).mean().item()


# ── JEPA Nerve Fiber ─────────────────────────────────────────

class JEPAFiber:
    """A nerve fiber backed by a tiny JEPA model.

    Each fiber has its own JEPA with a DIFFERENT random initialization
    (different seed per fiber). This means they learn different latent
    spaces for the same signals — diversity by construction.

    Args:
        fiber_id: Unique identifier.
        seed: Random seed (different per fiber for diversity).
        input_dim: JEPA input dimension.
        latent_dim: JEPA latent dimension.
        chaos_rate: How often to randomly reroute (0-1).
        device: 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        fiber_id: str,
        seed: int = 42,
        input_dim: int = 64,
        latent_dim: int = 16,
        chaos_rate: float = 0.1,
        device: str = "",
    ) -> None:
        torch.manual_seed(seed)
        self.fiber_id = fiber_id
        self.chaos_rate = chaos_rate
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.jepa = MinimalJEPA(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        self._signals_processed: int = 0
        self._latent_history: list[torch.Tensor] = []
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"JEPAFiber(id={self.fiber_id!r}, "
            f"processed={self._signals_processed}, "
            f"chaos={self.chaos_rate:.2f})"
        )

    def perceive(self, signal: torch.Tensor) -> dict[str, Any]:
        """Process a signal through the JEPA.

        Returns:
            Dict with latent, latent_pred, novelty_score, and chaos_flag.
        """
        with self._lock:
            self._signals_processed += 1
            signal = signal.to(self.device)

            with torch.no_grad():
                latent, latent_pred = self.jepa(signal)

            # Compute novelty: distance from recent latent history
            novelty = 0.5
            if self._latent_history:
                recent = torch.stack(self._latent_history[-5:])
                distances = torch.stack([
                    self._jepa_latent_dist(latent, h) for h in recent
                ])
                novelty = distances.mean().item()

            self._latent_history.append(latent)
            if len(self._latent_history) > 100:
                self._latent_history.pop(0)

            # Chaos probability
            chaos_flag = random.random() < self.chaos_rate

            return {
                "fiber_id": self.fiber_id,
                "latent": latent.cpu(),
                "latent_pred": latent_pred.cpu(),
                "novelty": novelty,
                "chaos_triggered": chaos_flag,
                "signals_processed": self._signals_processed,
            }

    def _jepa_latent_dist(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute distance between a latent and a history vector."""
        z_n = z / (z.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        h_n = h / (h.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        cos_sim = (z_n * h_n).sum(dim=-1)
        return 1.0 - cos_sim.mean()


@dataclass
class ChaosMessage:
    """A message passed between chaos rooms.

    Attributes:
        source_fiber: Which fiber sent this.
        latent: The JEPA latent vector.
        novelty: Novelty score (0-1).
        chaos_path: Which chaos path this traveled.
        hop_count: How many rooms this passed through.
    """
    source_fiber: str
    latent: torch.Tensor
    novelty: float
    chaos_path: str
    hop_count: int = 0


class ChaosRoom:
    """A room that receives JEPA signals through chaos-routed connections.

    Each room is a "micro-problem solver": it takes signals from fibers,
    compares them with its own latent space, and decides whether to fire
    or remain silent.

    Args:
        room_id: Unique identifier.
        problem_statement: The tiny problem this room solves.
        latent_dim: JEPA latent dimension.
        chaos_decay: How fast chaos decays (0-1).
    """

    def __init__(
        self,
        room_id: str,
        problem_statement: str,
        latent_dim: int = 16,
        chaos_decay: float = 0.95,
    ) -> None:
        self.room_id = room_id
        self.problem_statement = problem_statement
        self.latent_dim = latent_dim
        self.chaos_decay = chaos_decay

        # Each room maintains its OWN latent space (learned from signals it finds useful)
        self._latent_memory: list[dict[str, Any]] = []
        self._connections: dict[str, float] = {}  # fiber_id → weight
        self._chaos_prob: float = 0.3  # start high, decay as connections strengthen
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"ChaosRoom(id={self.room_id!r}, "
            f"memories={len(self._latent_memory)}, "
            f"connections={len(self._connections)}, "
            f"chaos={self._chaos_prob:.2f})"
        )

    def receive(self, msg: ChaosMessage) -> bool:
        """Receive a chaos message. Returns True if the room FIRES.

        A room fires when:
        - The signal is novel enough (novelty > threshold)
        - OR chaos says to fire anyway
        """
        with self._lock:
            novelty = msg.novelty
            chaos_fire = random.random() < self._chaos_prob
            fires = novelty > 0.3 or chaos_fire

            self._latent_memory.append({
                "fiber": msg.source_fiber,
                "latent": msg.latent,
                "novelty": novelty,
                "chaos": chaos_fire,
            })

            # Strengthen connection to source fiber
            key = msg.source_fiber
            self._connections[key] = min(1.0, self._connections.get(key, 0.0) + 0.1 * novelty)

            # Decay chaos as connections strengthen
            avg_strength = sum(self._connections.values()) / max(1, len(self._connections))
            self._chaos_prob = max(0.01, 0.3 * (1.0 - avg_strength))

            # Trim memory
            if len(self._latent_memory) > 100:
                self._latent_memory = self._latent_memory[-100:]

        return fires


class JEPASwarm:
    """A swarm of JEPA fibers routing through chaos rooms.

    Each fiber perceives a signal, produces a latent.
    Chaos rooms compare latents and decide to fire.
    Fired rooms become the "distillation source" for the next generation.

    Args:
        n_fibers: Number of JEPA fibers.
        n_rooms: Number of chaos rooms.
        input_dim: Signal dimension.
        latent_dim: JEPA latent dimension.
    """

    def __init__(
        self,
        n_fibers: int = 12,
        n_rooms: int = 4,
        input_dim: int = 64,
        latent_dim: int = 16,
    ) -> None:
        self.fibers: list[JEPAFiber] = [
            JEPAFiber(fiber_id=f"jf-{i:02d}", seed=42 + i * 7, input_dim=input_dim, latent_dim=latent_dim)
            for i in range(n_fibers)
        ]
        self.rooms: list[ChaosRoom] = [
            ChaosRoom(room_id=f"cr-{i:02d}", problem_statement=problem)
            for i, problem in enumerate([
                "Find patterns in latent space",
                "Detect novelty vs noise",
                "Route high-novelty to distiller",
                "Bridge between fiber types",
            ][:n_rooms])
        ]
        self._signal_count: int = 0
        self._locks: dict[str, threading.Lock] = {}

        # Chaos routing matrix: fiber_id → {(room_id, chaos_path): probability}
        self._routing_matrix: dict[str, list[tuple[str, str, float]]] = {}
        for f in self.fibers:
            self._routing_matrix[f.fiber_id] = []
            for r in self.rooms:
                # Each fiber-room pair has a UNIQUE chaos path
                path = f"chaos-{random.randint(1000, 9999)}"
                prob = random.random()  # diverse initial probabilities
                self._routing_matrix[f.fiber_id].append((r.room_id, path, prob))

    def __repr__(self) -> str:
        return (
            f"JEPASwarm(fibers={len(self.fibers)}, "
            f"rooms={len(self.rooms)}, "
            f"signals={self._signal_count})"
        )

    def tick(self, signal: torch.Tensor) -> dict[str, Any]:
        """Run one tick: all fibers perceive, route through chaos rooms.

        Returns dict with results from each room.
        """
        self._signal_count += 1
        results: dict[str, Any] = {}

        # Each fiber perceives
        perceptions = {}
        for f in self.fibers:
            p = f.perceive(signal)
            perceptions[f.fiber_id] = p

        # Route through chaos rooms (chaos-probabilistic)
        for f in self.fibers:
            p = perceptions[f.fiber_id]
            for room_id, path, prob in self._routing_matrix[f.fiber_id]:
                # Apply chaos: prob is dynamically weighted by novelty
                effective_prob = prob * (1.0 + p["novelty"])
                if random.random() < effective_prob or p["chaos_triggered"]:
                    msg = ChaosMessage(
                        source_fiber=f.fiber_id,
                        latent=p["latent"],
                        novelty=p["novelty"],
                        chaos_path=path,
                    )
                    # Find the room and send
                    for r in self.rooms:
                        if r.room_id == room_id:
                            fired = r.receive(msg)
                            if fired:
                                results.setdefault(room_id, [])
                                results[room_id].append({
                                    "fiber": f.fiber_id,
                                    "novelty": p["novelty"],
                                    "chaos": p["chaos_triggered"],
                                })
                            break

        return results

    def distill_candidates(self, min_fires: int = 3) -> list[str]:
        """Which rooms are ready to distill?

        A room with > min_fires recent fires is a candidate.
        """
        candidates = []
        for r in self.rooms:
            recent = [m for m in r._latent_memory[-10:] if m["novelty"] > 0.3]
            if len(recent) >= min_fires:
                candidates.append(r.room_id)
        return candidates

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "fibers": len(self.fibers),
            "rooms": len(self.rooms),
            "signals_processed": self._signal_count,
            "distill_candidates": self.distill_candidates(),
        }
