"""Demo: Full breeding cycle across 20 generations on a 50-room grid.

Shows lifecycle state transitions in real-time:
    EGG → INCUBATE → COMPETE → SURVIVE → BREED → SUNSET

Prints thermal pressure, population count, and diversity score.
Ends with a summary of breeding outcomes.

Run: python scripts/demo_breeding_cycle.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path so imports resolve when run from scripts/
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import tempfile
import time
import types

import numpy as np

# ── Mock turbovec ──
_mock_turbovec = types.ModuleType("turbovec")


class _MockIdMapIndex:
    def __init__(self, dim: int, bit_width: int = 4) -> None:
        self.dim = dim
        self.bit_width = bit_width
        self._vectors: dict[int, np.ndarray] = {}

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        for vec, aid in zip(vectors, ids):
            self._vectors[int(aid)] = vec.copy()

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        allowlist: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._vectors:
            return (
                np.zeros((1, k), dtype=np.float32),
                np.zeros((1, k), dtype=np.uint64),
            )
        q = query[0]
        candidates = list(self._vectors.items())
        if allowlist is not None:
            allowed = set(int(a) for a in allowlist)
            candidates = [(aid, v) for aid, v in candidates if aid in allowed]

        qn = q / (np.linalg.norm(q) + 1e-8)
        sims: list[tuple[int, float]] = []
        for aid, vec in candidates:
            vn = vec / (np.linalg.norm(vec) + 1e-8)
            sims.append((aid, float(np.dot(qn, vn))))
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:k]
        while len(top) < k:
            top.append((0, 0.0))
        scores = np.array([[s for _, s in top]], dtype=np.float32)
        ids_arr = np.array([[aid for aid, _ in top]], dtype=np.uint64)
        return scores, ids_arr

    def remove(self, agent_id: int) -> bool:
        return self._vectors.pop(agent_id, None) is not None

    def contains(self, agent_id: int) -> bool:
        return agent_id in self._vectors

    def write(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str) -> "_MockIdMapIndex":
        return cls(dim=256)


_mock_turbovec.IdMapIndex = _MockIdMapIndex  # type: ignore[attr-defined]
sys.modules["turbovec"] = _mock_turbovec

from nerve.room_grid import RoomGrid
from swarm.breeder_daemon_v2 import (
    BreederDaemonV2,
    DiversityConfig,
    LifecycleState,
    ThermalConfig,
)
from swarm.thermal import DeviceType, ThermalBudget
from swarm.vector_table import AgentVector, FluxVectorTable
from swarm.worker_pool import WorkerPool


def run_demo():
    # ── Setup ──────────────────────────────────────────
    n_rooms = 50
    n_generations = 20
    tick_interval = 0.1  # seconds between generations for readability

    grid = RoomGrid(n=n_rooms)
    thermal = ThermalBudget({DeviceType.GPU: 30, DeviceType.CPU: 10})
    pool = WorkerPool(grid, thermal, max_workers=30)

    # Pre-populated vector table for diversity tracking
    vt = FluxVectorTable(dim=256, bit_width=4)
    rng = np.random.RandomState(42)
    for i in range(50):
        vec = (rng.randn(256).astype(np.float32) * (1.0 + np.random.random())).tolist()
        vt.add(
            AgentVector(
                agent_id=i,
                vector=vec,
                fitness=0.5 + np.random.random() * 0.4,
                generation=0,
                capability_mask=0xFFFF,
                thermal_pressure=0.1,
            )
        )

    fd, wal_path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)

    daemon = BreederDaemonV2(
        grid=grid,
        thermal=thermal,
        vector_table=vt,
        diversity=DiversityConfig(),
        thermal_cfg=ThermalConfig(max_agents=30, hysteresis_ticks=2),
        wal_path=wal_path,
        tick_interval=tick_interval,
    )
    daemon.start()

    # ── Seed population ───────────────────────────────
    print("🥚  Breeding Cycle Demo — 50 rooms, 20 generations")
    print("=" * 60)
    print(f"{'Gen':>4} | {'Pop':>4} | {'Thermal':>7} | {'Diversity':>9} | {'Events'}")
    print("-" * 60)

    seed_rooms = list(range(8))
    for rid in seed_rooms:
        grid.activity[rid] = 5

    agents = []
    for rid in seed_rooms:
        aid = pool.spawn_worker(
            config={
                "room_id": rid,
                "tick_interval": 0.05,
                "max_ticks": 1000,
                "generation": 0,
            }
        )
        agents.append(aid)

    # Let seed mature
    time.sleep(0.6)

    # ── Metrics tracking ────────────────────────────────
    total_bred = 0
    total_sunset = 0
    fitness_values: list[float] = []
    generation_sizes = []

    # ── Generation loop ───────────────────────────────
    for gen in range(1, n_generations + 1):
        active = pool.list_active()
        available = [
            aid for aid, info in active.items()
            if info["lifecycle"] in ("SURVIVE", "BREED", "COMPETE")
        ]

        events = []
        bred_this_gen = 0
        sunset_this_gen = 0

        if len(available) >= 2:
            a, b = available[:2]
            daemon.queue_breed(parent_a=a, parent_b=b, priority=10)
            transitions = daemon.step()

            for tr in transitions:
                if tr.to_state == LifecycleState.INCUBATE:
                    total_bred += 1
                    bred_this_gen += 1
                    child_id = tr.agent_id

                    # Find child's room
                    room_id = None
                    for rid, allocated_aid in daemon._room_allocations.items():
                        if allocated_aid == child_id:
                            room_id = rid
                            break

                    if room_id is not None:
                        try:
                            # Daemon already allocated thermal; release so pool can manage
                            thermal.release(f"agent_{child_id}")
                        except Exception:
                            pass
                        try:
                            pool.spawn_worker(
                                agent_id=child_id,
                                config={
                                    "room_id": room_id,
                                    "tick_interval": 0.05,
                                    "max_ticks": 1000,
                                    "generation": gen,
                                    "parent_a": tr.parent_a,
                                    "parent_b": tr.parent_b,
                                }
                            )
                        except RuntimeError as e:
                            events.append(f"spawn-fail:{e}")

                if tr.to_state == LifecycleState.SUNSET:
                    total_sunset += 1
                    sunset_this_gen += 1

        # Collect fitness from vector table for active agents
        pop_size = len(pool.list_active())
        gen_fitness = []
        for aid in pool.list_active():
            meta = vt._meta.get(aid)
            if meta:
                gen_fitness.append(meta.fitness)

        if gen_fitness:
            fitness_values.extend(gen_fitness)

        diversity = daemon.diversity_score
        thermal_pct = (thermal.total_current / thermal.total_max) * 100

        gen_events = []
        if bred_this_gen:
            gen_events.append(f"+{bred_this_gen}")
        if sunset_this_gen:
            gen_events.append(f"-{sunset_this_gen}")
        if not gen_events:
            gen_events.append("stable")

        print(
            f"{gen:>4} | {pop_size:>4} | {thermal_pct:>6.1f}% | {diversity:>9.4f} | {' '.join(gen_events)}"
        )

        generation_sizes.append(pop_size)
        time.sleep(tick_interval)

    # ── Cleanup ─────────────────────────────────────────
    daemon.stop()
    final_active = pool.list_active()

    # Let remaining workers finish or kill them
    for aid in list(final_active.keys()):
        pool.kill_worker(aid)

    pool.kill_all()
    os.unlink(wal_path)

    # ── Summary ─────────────────────────────────────────
    avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0.0
    max_pop = max(generation_sizes) if generation_sizes else 0
    min_pop = min(generation_sizes) if generation_sizes else 0

    print("=" * 60)
    print("📊 SUMMARY")
    print(f"  Total bred:    {total_bred}")
    print(f"  Total sunset:  {total_sunset}")
    print(f"  Avg fitness:   {avg_fitness:.3f}")
    print(f"  Max population: {max_pop}")
    print(f"  Min population: {min_pop}")
    print(f"  Final diversity: {daemon.diversity_score:.4f}")
    print("=" * 60)
    print("✅ Demo complete.")


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(0)
