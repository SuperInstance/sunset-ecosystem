#!/usr/bin/env python3
"""chaos_distill.py — JEPA fibers route through chaos rooms to find distillation sources.

Each room solves a micro-problem. Rooms that fire most become "teachers"
for the next generation — their latent space seeds the next room's JEPA.

The question: can rooms that never fire be pruned (thermal budget)?
Can we detect which room solves WHICH micro-problem from its firing pattern?
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nerve.jepa import JEPASwarm, ChaosRoom, ChaosMessage


# ── Micro-problems each room solves ──────────────────────
MICRO_PROBLEMS = [
    "Which dimension of the latent space correlates with signal intensity?",
    "Which fiber provides the most diverse perspective on this signal?",
    "Is this signal novel or just noise?",
    "Which combination of fibers gives the best reconstruction?",
    "Should this signal be forwarded or filtered?",
    "Is the chaos probability helping or hurting exploration?",
]


# ── Signal types to test ────────────────────────────────
def make_signal(kind: str) -> torch.Tensor:
    """Generate different signal types."""
    if kind == "sine":
        t = torch.linspace(0, 2 * torch.pi, 64)
        return torch.sin(t).unsqueeze(0)
    elif kind == "noise":
        return torch.randn(1, 64)
    elif kind == "step":
        x = torch.zeros(1, 64)
        x[0, 32:] = 1.0
        return x
    elif kind == "trend":
        return torch.linspace(0, 1, 64).unsqueeze(0)
    elif kind == "sawtooth":
        t = torch.linspace(0, 4 * torch.pi, 64)
        return (t % (2 * torch.pi) / (2 * torch.pi)).unsqueeze(0)
    else:
        return torch.randn(1, 64)


# ── Run experiment ──────────────────────────────────────
def run_chaos_experiment() -> None:
    print("=" * 60)
    print("JEPA CHAOS DISTILLATION EXPERIMENT")
    print("=" * 60)
    print()

    # Create swarm with rooms for each micro-problem
    n_rooms = len(MICRO_PROBLEMS)
    swarm = JEPASwarm(n_fibers=12, n_rooms=n_rooms, input_dim=64, latent_dim=16)

    # Override room problem statements
    for i, room in enumerate(swarm.rooms):
        room.problem_statement = MICRO_PROBLEMS[i]

    print(f"Running swarm: {len(swarm.fibers)} fibers → {len(swarm.rooms)} rooms")
    print(f"Signal types: sine, noise, step, trend, sawtooth")
    print()

    # ── Phase 1: Explore all signal types ─────────────
    print("Phase 1: Exploration (100 signals, mixed types)")
    print("-" * 50)
    signal_types = ["sine", "noise", "step", "trend", "sawtooth"]
    room_fire_counts = {r.room_id: 0 for r in swarm.rooms}
    chaos_fire_counts = {r.room_id: 0 for r in swarm.rooms}

    for _ in range(100):
        sig_type = random.choice(signal_types)
        signal = make_signal(sig_type)
        results = swarm.tick(signal)

        for room_id, fires in results.items():
            room_fire_counts[room_id] = room_fire_counts.get(room_id, 0) + len(fires)
            for f in fires:
                if f.get("chaos"):
                    chaos_fire_counts[room_id] = chaos_fire_counts.get(room_id, 0) + 1

    print(f"Total ticks: {swarm._signal_count}")
    print(f"Room firing counts:")
    for r in swarm.rooms:
        total = room_fire_counts.get(r.room_id, 0)
        chaos = chaos_fire_counts.get(r.room_id, 0)
        pct = (chaos / total * 100) if total > 0 else 0
        print(
            f"  {r.room_id}: {total:3d} fires ({pct:.0f}% chaos) — {r.problem_statement[:40]}..."
        )
    print()

    # ── Phase 2: Find distillation candidates ─────────
    print("Phase 2: Distillation Candidates")
    print("-" * 50)
    candidates = swarm.distill_candidates(min_fires=10)
    print(f"Room candidates (fires > 10): {candidates}")

    for room_id in candidates:
        room = next(r for r in swarm.rooms if r.room_id == room_id)
        print(f"  {room_id} — {room.problem_statement}")
        print(f"  Connections: {len(room._connections)} fibers")
        print(f"  Current chaos: {room._chaos_prob:.3f}")
    print()

    # ── Phase 3: Thermal pruning + rebirth ────────────
    print("Phase 3: Thermal Pruning")
    print("-" * 50)
    cold_rooms = [r for r in swarm.rooms if room_fire_counts.get(r.room_id, 0) < 5]
    hot_rooms = [r for r in swarm.rooms if room_fire_counts.get(r.room_id, 0) >= 5]

    print(f"Hot rooms (keep): {len(hot_rooms)}")
    print(f"Cold rooms (prune): {len(cold_rooms)}")

    for cold in cold_rooms:
        print(f"  Pruning {cold.room_id} ({cold.problem_statement[:40]}...)")
        # In real system: sacrifice parent, spawn child with hot room's latent

    for hot in hot_rooms:
        print(f"  Keeping {hot.room_id} ({hot.problem_statement[:40]}...)")
        print(
            f"    Top fibers: {sorted(hot._connections.items(), key=lambda x: x[1], reverse=True)[:3]}"
        )
    print()

    # ── Summary ────────────────────────────────────────
    print("Summary")
    print("-" * 50)
    s = swarm.stats
    print(f"Signals processed: {s['signals_processed']}")
    print(f"Distill candidates: {s['distill_candidates']}")

    # What each room learned (its connection profile = its fingerprint)
    print(f"\nRoom latent fingerprints (connection weights):")
    for r in sorted(
        swarm.rooms, key=lambda r: room_fire_counts.get(r.room_id, 0), reverse=True
    ):
        top = sorted(r._connections.items(), key=lambda x: x[1], reverse=True)[:3]
        weights = ", ".join(f"{f}:{w:.2f}" for f, w in top)
        print(f"  {r.room_id}: [{weights}]")


if __name__ == "__main__":
    run_chaos_experiment()
