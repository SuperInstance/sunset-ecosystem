"""Bounded Evolutionary Parameter Engine with Rollback.

Implements Pattern 4 from the SuperInstance audit: tunable parameters
with explicit bounds, mutation rates, fitness scores, and full
rollback capability. Supports three evolution modes (aggressive,
normal, elite) and eight mutation types.

Reference: flux-evolve — bounded evolutionary parameter engine.
"""
from __future__ import annotations

import copy
import enum
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EvolutionMode(enum.Enum):
    AGGRESSIVE = "aggressive"
    NORMAL = "normal"
    ELITE = "elite"


class MutationType(enum.Enum):
    PARAM_ADJUST = "ParamAdjust"
    THRESHOLD_SHIFT = "ThresholdShift"
    WEIGHT_REBALANCE = "WeightRebalance"
    CROSSOVER = "Crossover"
    SWAP = "Swap"
    INVERT = "Invert"
    SCALE = "Scale"
    RESET = "Reset"


@dataclass
class BoundedParameter:
    value: float
    min: float
    max: float
    mutation_rate: float = 0.1
    fitness_score: float = 0.0
    name: str = ""

    def clamp(self) -> float:
        self.value = max(self.min, min(self.max, self.value))
        return self.value

    @property
    def range(self) -> float:
        return self.max - self.min

    def copy(self) -> BoundedParameter:
        return BoundedParameter(
            value=self.value,
            min=self.min,
            max=self.max,
            mutation_rate=self.mutation_rate,
            fitness_score=self.fitness_score,
            name=self.name,
        )


@dataclass
class GenerationSnapshot:
    generation: int
    parameters: dict[str, BoundedParameter]
    mode: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "parameters": {
                name: {
                    "value": p.value,
                    "min": p.min,
                    "max": p.max,
                    "mutation_rate": p.mutation_rate,
                    "fitness_score": p.fitness_score,
                    "name": p.name,
                }
                for name, p in self.parameters.items()
            },
            "mode": self.mode,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationSnapshot:
        params = {
            name: BoundedParameter(
                value=p["value"],
                min=p["min"],
                max=p["max"],
                mutation_rate=p.get("mutation_rate", 0.1),
                fitness_score=p.get("fitness_score", 0.0),
                name=p.get("name", name),
            )
            for name, p in data["parameters"].items()
        }
        return cls(
            generation=data["generation"],
            parameters=params,
            mode=data["mode"],
            timestamp=data.get("timestamp", 0.0),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GenerationSnapshot):
            return NotImplemented
        return (
            self.generation == other.generation
            and self.mode == other.mode
            and {
                k: (v.value, v.min, v.max, v.mutation_rate, v.fitness_score, v.name)
                for k, v in self.parameters.items()
            }
            == {
                k: (v.value, v.min, v.max, v.mutation_rate, v.fitness_score, v.name)
                for k, v in other.parameters.items()
            }
        )


class EvolutionEngine:
    """Evolution engine for bounded behavioral parameters.

    Maintains a set of :class:`BoundedParameter` values, mutates them
    according to a selectable strategy, accumulates fitness evidence via
    :meth:`score`, and can :meth:`rollback` to any previous generation.
    """

    MODE_MAGNITUDE: dict[EvolutionMode, float] = {
        EvolutionMode.AGGRESSIVE: 3.0,
        EvolutionMode.NORMAL: 1.0,
        EvolutionMode.ELITE: 0.5,
    }

    # Mutations that never touch another parameter — safe for elite mode.
    _SAFE_MUTATIONS: list[MutationType] = [
        MutationType.PARAM_ADJUST,
        MutationType.THRESHOLD_SHIFT,
        MutationType.WEIGHT_REBALANCE,
        MutationType.INVERT,
        MutationType.SCALE,
        MutationType.RESET,
    ]

    # All mutation types.
    _ALL_MUTATIONS: list[MutationType] = list(MutationType)

    def __init__(
        self,
        parameters: list[BoundedParameter],
        mode: EvolutionMode = EvolutionMode.NORMAL,
        auto_mode: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.parameters: dict[str, BoundedParameter] = {}
        self._initial_values: dict[str, float] = {}
        for p in parameters:
            name = p.name or f"param_{len(self.parameters)}"
            p.name = name
            self.parameters[name] = p
            self._initial_values[name] = p.value
        self.mode = mode
        self.auto_mode = auto_mode
        self.generation = 0
        self._score_count = 0
        self._score_sum = 0.0
        self.snapshots: dict[int, GenerationSnapshot] = {}
        self._rng = random.Random(seed)

    # ── fitness & mode ─────────────────────────────────────────

    def average_fitness(self) -> float:
        if not self.parameters:
            return 0.0
        return sum(p.fitness_score for p in self.parameters.values()) / len(
            self.parameters
        )

    def _magnitude(self) -> float:
        return self.MODE_MAGNITUDE.get(self.mode, 1.0)

    def _select_mode(self) -> None:
        if not self.auto_mode or not self.parameters:
            return
        avg = self.average_fitness()
        if avg < 0.3:
            self.mode = EvolutionMode.AGGRESSIVE
        elif avg > 0.7:
            self.mode = EvolutionMode.ELITE
        else:
            self.mode = EvolutionMode.NORMAL

    # ── scoring ────────────────────────────────────────────────

    def score(self, behavior: dict[str, float], outcome: float) -> None:
        """Accumulate fitness evidence.

        ``behavior`` maps parameter names to the values they had during
        the observed behavior. Every listed parameter has ``outcome``
        added to its cumulative :attr:`BoundedParameter.fitness_score`.
        If ``behavior`` is empty, all parameters are scored.
        """
        targets = list(behavior.keys()) if behavior else list(self.parameters.keys())
        for name in targets:
            if name in self.parameters:
                self.parameters[name].fitness_score += outcome
        self._score_count += 1
        self._score_sum += outcome

    # ── snapshot & rollback ────────────────────────────────────

    def snapshot(self) -> GenerationSnapshot:
        snap = GenerationSnapshot(
            generation=self.generation,
            parameters={name: p.copy() for name, p in self.parameters.items()},
            mode=self.mode.value,
        )
        self.snapshots[self.generation] = snap
        return snap

    def rollback(self, target_generation: int) -> None:
        if target_generation not in self.snapshots:
            raise ValueError(f"No snapshot for generation {target_generation}")
        snap = self.snapshots[target_generation]
        self.parameters = {name: p.copy() for name, p in snap.parameters.items()}
        self.mode = EvolutionMode(snap.mode)
        self.generation = target_generation
        # Prune future snapshots
        self.snapshots = {
            g: s for g, s in self.snapshots.items() if g <= target_generation
        }

    # ── evolution ──────────────────────────────────────────────

    def evolve(self) -> None:
        """Run one generation of mutation.

        1. Auto-select mode based on average fitness (if enabled).
        2. Snapshot current state.
        3. Apply mutations according to the active strategy.
        4. Increment generation counter.
        """
        self._select_mode()
        self.snapshot()
        self.generation += 1

        if not self.parameters:
            return

        mode = self.mode
        magnitude = self._magnitude()

        if mode == EvolutionMode.ELITE:
            # Only mutate worst performers
            sorted_params = sorted(
                self.parameters.items(), key=lambda kv: kv[1].fitness_score
            )
            n = max(1, int(len(sorted_params) * 0.3))
            targets = [name for name, _ in sorted_params[:n]]
            self._mutate_single_params(targets, magnitude, safe_only=True)
        elif mode == EvolutionMode.AGGRESSIVE:
            targets = list(self.parameters.keys())
            self._mutate_with_pairs(targets, magnitude)
        else:
            targets = list(self.parameters.keys())
            self._mutate_single_params(targets, magnitude, safe_only=False)

    def _mutate_single_params(
        self, targets: list[str], magnitude: float, safe_only: bool = False
    ) -> None:
        pool = self._SAFE_MUTATIONS if safe_only else self._ALL_MUTATIONS
        for name in targets:
            mtype = self._rng.choice(pool)
            self._apply_single_mutation(name, mtype, magnitude)

    def _mutate_with_pairs(self, targets: list[str], magnitude: float) -> None:
        shuffled = targets[:]
        self._rng.shuffle(shuffled)

        # Apply one crossover pair
        if len(shuffled) >= 2:
            self._apply_pair_mutation(
                MutationType.CROSSOVER, shuffled[0], shuffled[1]
            )
        # Apply one swap pair
        if len(shuffled) >= 4:
            self._apply_pair_mutation(
                MutationType.SWAP, shuffled[2], shuffled[3]
            )
        # Remaining get random single mutations
        remaining: list[str] = []
        if len(shuffled) >= 4:
            remaining = shuffled[4:]
        elif len(shuffled) >= 2:
            remaining = shuffled[2:]
        else:
            remaining = shuffled
        for name in remaining:
            mtype = self._rng.choice(self._ALL_MUTATIONS)
            self._apply_single_mutation(name, mtype, magnitude)

    def _apply_single_mutation(
        self, name: str, mutation_type: MutationType, magnitude: float
    ) -> None:
        p = self.parameters[name]
        rng = self._rng
        mr = p.mutation_rate * magnitude
        r = p.range

        if mutation_type == MutationType.PARAM_ADJUST:
            delta = rng.uniform(-1.0, 1.0) * mr * r
            p.value += delta

        elif mutation_type == MutationType.THRESHOLD_SHIFT:
            direction = 1.0 if rng.random() < 0.5 else -1.0
            delta = direction * mr * r
            p.value += delta

        elif mutation_type == MutationType.WEIGHT_REBALANCE:
            mid = (p.min + p.max) / 2.0
            p.value += (mid - p.value) * mr

        elif mutation_type == MutationType.INVERT:
            p.value = p.min + p.max - p.value

        elif mutation_type == MutationType.SCALE:
            factor = 1.0 + rng.uniform(-1.0, 1.0) * mr
            p.value *= factor

        elif mutation_type == MutationType.RESET:
            p.value = self._initial_values.get(name, (p.min + p.max) / 2.0)

        elif mutation_type == MutationType.CROSSOVER:
            partners = [n for n in self.parameters if n != name]
            if partners:
                partner = self.parameters[rng.choice(partners)]
                alpha = rng.random()
                p.value = p.value * (1.0 - alpha) + partner.value * alpha
            else:
                p.value += rng.uniform(-1.0, 1.0) * mr * r

        elif mutation_type == MutationType.SWAP:
            partners = [n for n in self.parameters if n != name]
            if partners:
                partner_name = rng.choice(partners)
                partner = self.parameters[partner_name]
                p.value, partner.value = partner.value, p.value
                partner.clamp()
            else:
                p.value += rng.uniform(-1.0, 1.0) * mr * r

        p.clamp()

    def _apply_pair_mutation(
        self, mutation_type: MutationType, name_a: str, name_b: str
    ) -> None:
        a = self.parameters[name_a]
        b = self.parameters[name_b]

        if mutation_type == MutationType.CROSSOVER:
            avg = (a.value + b.value) / 2.0
            a.value = avg
            b.value = avg
        elif mutation_type == MutationType.SWAP:
            a.value, b.value = b.value, a.value
        else:
            # Fallback — should not happen for pair calls
            for name in (name_a, name_b):
                self._apply_single_mutation(
                    name, MutationType.PARAM_ADJUST, self._magnitude()
                )
            return

        a.clamp()
        b.clamp()

    # ── representation ─────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"EvolutionEngine(gen={self.generation}, mode={self.mode.value}, "
            f"params={len(self.parameters)}, fitness={self.average_fitness():.3f})"
        )
