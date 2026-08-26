"""Hardware-Conditional NAS — Aging evolution over RoomGrid configs, per hardware target.

Experiment 2 from RESEARCH_SELF_IMPROVEMENT.md.
"""

from __future__ import annotations

__all__ = [
    "HardwareConditionalNAS",
    "oracle1_profile",
    "jetson_profile",
    "laptop_profile",
    "pareto_dominates",
]

import json
import math
import random
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ── Import RoomGrid (repo has no root __init__.py; use path hack if needed) ──
try:
    from nerve.room_grid import RoomGrid, batch_novelty
except ImportError:
    import sys
    from pathlib import Path

    _NERVE = Path(__file__).parent.parent / "nerve"
    sys.path.insert(0, str(_NERVE.parent))
    from nerve.room_grid import RoomGrid, batch_novelty


# ── Hardware Profiles ──────────────────────────────────────────────────────────

oracle1_profile = {
    "device": "Alibaba Cloud",
    "ram_gb": 32,
    "cpu_cores": 8,
    "gpu": "none",
}

jetson_profile = {
    "device": "Jetson Orin",
    "ram_gb": 8,
    "cpu_cores": 8,
    "gpu": "CUDA",
}

laptop_profile = {
    "device": "RTX 4050 Laptop",
    "ram_gb": 16,
    "cpu_cores": 8,
    "gpu": "CUDA",
}


# ── Search Space ───────────────────────────────────────────────────────────────

_SEARCH_SPACE = {
    "n_rooms": [100, 250, 500, 750, 1000],
    "d_latent": [32, 64, 128],
    "h_history": [8, 16, 32],
    "l_signal": [8, 16, 32, 64],
    "chaos_decay": [0.90, 0.95, 0.99],
    "route_density": [0.01, 0.05, 0.10, 0.20],
}

_SEARCH_SPACE_SIZE = math.prod(len(v) for v in _SEARCH_SPACE.values())


# ── Config utilities ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable config key for hashing / dedup."""

    n_rooms: int
    d_latent: int
    h_history: int
    l_signal: int
    chaos_decay: float
    route_density: float

    def to_dict(self) -> dict:
        return {
            "n_rooms": self.n_rooms,
            "d_latent": self.d_latent,
            "h_history": self.h_history,
            "l_signal": self.l_signal,
            "chaos_decay": self.chaos_decay,
            "route_density": self.route_density,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(**{k: d[k] for k in _SEARCH_SPACE})

    def __repr__(self) -> str:
        return (
            f"Config(n={self.n_rooms}, d={self.d_latent}, h={self.h_history}, "
            f"l={self.l_signal}, decay={self.chaos_decay}, density={self.route_density})"
        )


def _random_config(rng: random.Random) -> Config:
    """Sample uniformly from the discrete search space."""
    return Config(
        n_rooms=rng.choice(_SEARCH_SPACE["n_rooms"]),
        d_latent=rng.choice(_SEARCH_SPACE["d_latent"]),
        h_history=rng.choice(_SEARCH_SPACE["h_history"]),
        l_signal=rng.choice(_SEARCH_SPACE["l_signal"]),
        chaos_decay=rng.choice(_SEARCH_SPACE["chaos_decay"]),
        route_density=rng.choice(_SEARCH_SPACE["route_density"]),
    )


def _mutate_config(
    parent: Config, rng: random.Random, p_mutate: float = 0.10
) -> Config:
    """Independent parameter crossover + adjacent-value mutation."""
    fields = [
        "n_rooms",
        "d_latent",
        "h_history",
        "l_signal",
        "chaos_decay",
        "route_density",
    ]
    vals: dict = {}
    for f in fields:
        choices = _SEARCH_SPACE[f]
        # 50% chance to keep parent value, else sample from full space (crossover-like)
        if rng.random() < 0.5:
            base = getattr(parent, f)
        else:
            base = rng.choice(choices)
        # Mutation: move to adjacent discrete value with probability p_mutate
        if rng.random() < p_mutate:
            idx = choices.index(base)
            delta = rng.choice([-1, 1])
            new_idx = max(0, min(len(choices) - 1, idx + delta))
            base = choices[new_idx]
        vals[f] = base
    return Config(**vals)


def _feasible_for_hardware(config: Config, profile: dict) -> bool:
    """Reject configs that exceed hardware limits."""
    ram_gb = profile.get("ram_gb", 16)
    # Rough memory model: each room has ~ (d*h + h*l + l*l) * 4 bytes weights
    # Plus history buffer: hist_max * n * l * 4 bytes
    # Plus overhead ~ 20%
    w1 = config.n_rooms * config.d_latent * config.h_history * 4
    w2 = config.n_rooms * config.h_history * config.l_signal * 4
    w3 = config.n_rooms * config.l_signal * config.l_signal * 4
    hist = 20 * config.n_rooms * config.l_signal * 4
    total_mb = (w1 + w2 + w3 + hist) / (1024 * 1024) * 1.2
    return total_mb < ram_gb * 1024 * 0.75  # keep 25% headroom


# ── Pareto helpers ────────────────────────────────────────────────────────────


def pareto_dominates(
    a: dict, b: dict, objectives: list[str], maximize: set[str]
) -> bool:
    """Return True if a Pareto-dominates b.

    objectives: list of metric names.
    maximize: set of metric names to maximize (the rest are minimized).
    """
    better_on_any = False
    for obj in objectives:
        av, bv = a[obj], b[obj]
        if obj in maximize:
            if av < bv:
                return False
            if av > bv:
                better_on_any = True
        else:
            if av > bv:
                return False
            if av < bv:
                better_on_any = True
    return better_on_any


def compute_pareto_frontier(
    points: list[dict],
    objectives: list[str] = None,
    maximize: set[str] = None,
) -> list[dict]:
    """Return the non-dominated subset."""
    if objectives is None:
        objectives = ["ticks_per_second", "diversity", "stability", "memory_mb"]
    if maximize is None:
        maximize = {"ticks_per_second", "diversity", "stability"}
    frontier = []
    for p in points:
        dominated = False
        for q in frontier:
            if pareto_dominates(q, p, objectives, maximize):
                dominated = True
                break
        if not dominated:
            # Remove any points that p dominates
            frontier = [
                q for q in frontier if not pareto_dominates(p, q, objectives, maximize)
            ]
            frontier.append(p)
    return frontier


# ── Core NAS class ───────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    """Result of evaluating one config."""

    config: Config
    ticks_per_second: float
    memory_mb: float
    diversity: float
    stability: float
    age: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = self.config.to_dict()
        d.update(
            {
                "ticks_per_second": self.ticks_per_second,
                "memory_mb": self.memory_mb,
                "diversity": self.diversity,
                "stability": self.stability,
                "age": self.age,
            }
        )
        return d


class HardwareConditionalNAS:
    """Aging evolution search over RoomGrid configs, per hardware target."""

    SEARCH_SPACE = _SEARCH_SPACE
    SEARCH_SPACE_SIZE = _SEARCH_SPACE_SIZE

    def __init__(self, hardware_profile: dict, max_evals: int = 100, seed: int = 42):
        """hardware_profile: {'device': 'RTX4050', 'ram_gb': 16, 'cpu_cores': 8, 'gpu': 'CUDA'}"""
        self.hardware = hardware_profile
        self.max_evals = max_evals
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.evaluated: list[EvalResult] = []
        self._eval_count = 0

    @property
    def eval_count(self) -> int:
        return self._eval_count

    def evaluate(self, config: Config | dict) -> EvalResult:
        """Run a short RoomGrid simulation (100 ticks) and measure metrics.

        Returns:
            EvalResult with ticks_per_second, memory_mb, diversity, stability.
        """
        if isinstance(config, dict):
            config = Config.from_dict(config)

        if not _feasible_for_hardware(config, self.hardware):
            # Return a penalty result so infeasible configs are filtered out
            return EvalResult(
                config=config,
                ticks_per_second=0.0,
                memory_mb=999999.0,
                diversity=0.0,
                stability=0.0,
            )

        # Seed for reproducibility per config
        seed = hash(config) % (2**31)
        np.random.seed(seed)

        grid = RoomGrid(
            n=config.n_rooms,
            d=config.d_latent,
            h=config.h_history,
            l=config.l_signal,
            chaos=0.3,
        )

        # Warm-up
        warmup_signal = np.random.randn(config.d_latent).astype(np.float32)
        for _ in range(5):
            grid.tick(warmup_signal)

        # Measurement
        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0] / (1024 * 1024)

        chaos_history = []
        tick_times = []
        for _ in range(100):
            signal = np.random.randn(config.d_latent).astype(np.float32)
            t0 = time.perf_counter()
            grid.tick(signal)
            t1 = time.perf_counter()
            tick_times.append(t1 - t0)
            chaos_history.append(grid.chaos.copy())

        end_mem = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
        tracemalloc.stop()

        ticks_per_second = 1.0 / (sum(tick_times) / len(tick_times))
        memory_mb = end_mem - start_mem

        # Diversity: mean novelty score across rooms on final tick
        # Re-run one more tick to get fresh latents + novelty
        signal = np.random.randn(config.d_latent).astype(np.float32)
        grid.tick(signal)
        latents = grid.latents
        diversity = float(np.std(latents))  # proxy: variance in latent activations
        # Also compute actual novelty if history is warm
        if grid._hist_count.min() >= 3:
            nv = batch_novelty(
                latents,
                grid._hist,
                grid._hist_count,
                grid._hist_idx,
                grid._hist_max,
            )
            diversity = float(nv.mean())

        # Stability: inverse of mean chaos variance across rooms
        chaos_stack = np.stack(chaos_history, axis=0)  # (100, n_rooms)
        chaos_var = chaos_stack.var(axis=0).mean()
        stability = float(1.0 / (1.0 + chaos_var))

        self._eval_count += 1
        return EvalResult(
            config=config,
            ticks_per_second=ticks_per_second,
            memory_mb=max(0.0, memory_mb),
            diversity=diversity,
            stability=stability,
        )

    def aging_evolution(
        self,
        population_size: int = 20,
        generations: int = 10,
        tournament_k: int = 5,
        progress_cb: Callable | None = None,
    ) -> list[dict]:
        """Aging evolution: older configs have higher selection pressure.

        Returns Pareto frontier of (ticks/s, diversity, memory, stability) configs.
        """
        population: list[EvalResult] = []

        # ── Initialize ──
        while len(population) < population_size:
            cfg = _random_config(self.rng)
            if not _feasible_for_hardware(cfg, self.hardware):
                continue
            # Skip duplicates
            if any(r.config == cfg for r in population):
                continue
            result = self.evaluate(cfg)
            population.append(result)
            if progress_cb:
                progress_cb("init", len(population), population_size, result)
            if self._eval_count >= self.max_evals:
                break

        # ── Evolve ──
        for gen in range(generations):
            if self._eval_count >= self.max_evals:
                break

            # Tournament selection: prefer older (aging pressure)
            # Age = how many generations this config has survived
            candidates = self.rng.sample(population, min(tournament_k, len(population)))
            # Older = higher age = higher selection pressure (Real et al. 2019)
            parent = max(candidates, key=lambda r: r.age)

            child_cfg = _mutate_config(parent.config, self.rng)
            # Avoid duplicates in population
            attempts = 0
            while any(r.config == child_cfg for r in population) and attempts < 10:
                child_cfg = _mutate_config(parent.config, self.rng)
                attempts += 1

            child = self.evaluate(child_cfg)

            # Aging: remove oldest (highest age), add child with age=0
            # If child is infeasible, remove it immediately
            if child.memory_mb >= 999000:
                # Infeasible — don't add, just age everyone
                for r in population:
                    r.age += 1
                if progress_cb:
                    progress_cb("gen_infeasible", gen, generations, child)
                continue

            oldest = max(population, key=lambda r: r.age)
            population.remove(oldest)
            population.append(child)

            # Age surviving members
            for r in population:
                r.age += 1

            if progress_cb:
                progress_cb("gen", gen, generations, child)

        # ── Extract Pareto frontier ──
        points = [r.to_dict() for r in population]
        frontier = compute_pareto_frontier(points)
        return frontier

    def best_for_hardware(self, top_k: int = 5) -> list[dict]:
        """Run aging evolution and return top-k configs sorted by composite score."""
        frontier = self.aging_evolution()
        # Composite score: ticks/s * diversity * stability / memory (lower memory is better)
        scored = []
        for p in frontier:
            score = (
                p.get("ticks_per_second", 0)
                * p.get("diversity", 0)
                * p.get("stability", 0)
            ) / (1.0 + p.get("memory_mb", 1))
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]


# ── CLI helpers ────────────────────────────────────────────────────────────────


def run_nas_for_profile(profile: dict, profile_name: str, **kwargs) -> list[dict]:
    """High-level wrapper: run NAS for a named hardware profile."""
    nas = HardwareConditionalNAS(profile, **kwargs)
    print(f"\n🔧 Profile: {profile_name} ({profile['device']})")
    print(f"   Search space: {nas.SEARCH_SPACE_SIZE} configs")
    print(f"   Max evals: {nas.max_evals}")

    def cb(stage, a, b, result):
        if stage == "init":
            print(
                f"   Init {a}/{b}: {result.config} → tps={result.ticks_per_second:.1f}"
            )
        elif stage == "gen":
            if a % 2 == 0 or a == b - 1:
                print(
                    f"   Gen {a}: {result.config} → tps={result.ticks_per_second:.1f}, div={result.diversity:.3f}"
                )

    frontier = nas.aging_evolution(
        progress_cb=cb,
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ("max_evals", "seed", "hardware_profile")
        },
    )
    print(f"   Pareto frontier: {len(frontier)} configs ({nas.eval_count} evals)")
    return frontier


if __name__ == "__main__":
    import sys

    # Simple smoke-test
    nas = HardwareConditionalNAS(jetson_profile, max_evals=10)
    result = nas.evaluate(
        Config(
            n_rooms=100,
            d_latent=32,
            h_history=8,
            l_signal=8,
            chaos_decay=0.95,
            route_density=0.05,
        )
    )
    print("Smoke test result:", result)
    frontier = nas.aging_evolution(population_size=5, generations=3)
    print(f"Frontier ({len(frontier)}):")
    for p in frontier:
        print(" ", p)
