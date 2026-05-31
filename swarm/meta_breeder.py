"""Cocapn Fleet — MetaBreeder
Adaptive SDA loop that selects the optimal breeder for the current fitness landscape.
It "breeds the breeders" by observing landscape characteristics and switching
strategies when the current breeder stalls.

SDA Loop:  SENSE landscape → DECIDE breeder → ACT run generation → ADAPT if stall
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque

from swarm.breeding_kernel import (
    BreedingKernel,
    BreedingPreset,
    BreedingEvent,
    Genome,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Landscape type classification
# ═══════════════════════════════════════════════════════════════════════════════

class LandscapeType(Enum):
    """Classification of fitness landscape characteristics."""

    SMOOTH = auto()       # Low ruggedness, few local optima
    RUGGED = auto()       # High ruggedness, many local optima
    MULTIMODAL = auto()   # Multiple distinct optima
    UNKNOWN = auto()      # Not enough data to classify


# ═══════════════════════════════════════════════════════════════════════════════
# LandscapeAnalyzer — analyzes fitness history to classify landscape
# ═══════════════════════════════════════════════════════════════════════════════

class LandscapeAnalyzer:
    """Analyzes fitness and diversity history to classify the current landscape."""

    def __init__(
        self,
        window_size: int = 10,
        ruggedness_threshold: float = 0.3,
        smoothness_threshold: float = 0.05,
        modality_threshold: float = 0.15,
    ):
        self.window_size = window_size
        self.ruggedness_threshold = ruggedness_threshold
        self.smoothness_threshold = smoothness_threshold
        self.modality_threshold = modality_threshold

    def analyze(self, fitness_history: List[float], diversity_history: List[float]) -> LandscapeType:
        """Classify landscape based on fitness and diversity history."""
        if len(fitness_history) < 3:
            return LandscapeType.UNKNOWN

        # Use sliding window
        window = fitness_history[-self.window_size:]

        ruggedness = self._compute_ruggedness(window)
        smoothness = self._compute_smoothness(window)
        modality = self._compute_modality(window)

        if modality > self.modality_threshold and diversity_history and diversity_history[-1] > 0.1:
            return LandscapeType.MULTIMODAL
        if ruggedness > self.ruggedness_threshold:
            return LandscapeType.RUGGED
        if smoothness < self.smoothness_threshold:
            return LandscapeType.SMOOTH

        # Default: if we have enough data but no strong signal, classify by dominant trait
        if ruggedness > smoothness * 2:
            return LandscapeType.RUGGED
        if modality > ruggedness:
            return LandscapeType.MULTIMODAL
        return LandscapeType.SMOOTH

    @staticmethod
    def _compute_ruggedness(fitness_window: List[float]) -> float:
        """Measure landscape ruggedness as normalized fitness fluctuation.

        Defined as the mean absolute first-difference divided by the range.
        """
        if len(fitness_window) < 2:
            return 0.0
        diffs = [abs(fitness_window[i] - fitness_window[i - 1]) for i in range(1, len(fitness_window))]
        mean_diff = sum(diffs) / len(diffs)
        value_range = max(fitness_window) - min(fitness_window)
        if value_range == 0:
            return 0.0
        return mean_diff / value_range

    @staticmethod
    def _compute_smoothness(fitness_window: List[float]) -> float:
        """Measure landscape smoothness via second-derivative magnitude.

        Lower values = smoother landscape.
        """
        if len(fitness_window) < 3:
            return float("inf")
        second_diffs = []
        for i in range(2, len(fitness_window)):
            d1 = fitness_window[i - 1] - fitness_window[i - 2]
            d2 = fitness_window[i] - fitness_window[i - 1]
            second_diffs.append(abs(d2 - d1))
        return sum(second_diffs) / len(second_diffs) if second_diffs else float("inf")

    @staticmethod
    def _compute_modality(fitness_window: List[float]) -> float:
        """Detect multiple local optima by counting direction changes.

        Returns the proportion of points that are local extrema.
        """
        if len(fitness_window) < 3:
            return 0.0
        extrema_count = 0
        for i in range(1, len(fitness_window) - 1):
            left = fitness_window[i - 1]
            center = fitness_window[i]
            right = fitness_window[i + 1]
            if (center > left and center > right) or (center < left and center < right):
                extrema_count += 1
        return extrema_count / (len(fitness_window) - 2)

    def __repr__(self) -> str:
        return f"LandscapeAnalyzer(window={self.window_size})"


# ═══════════════════════════════════════════════════════════════════════════════
# MetaBreedingEvent — events with reasoning
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetaBreedingEvent:
    """Event emitted by the MetaBreeder with selection reasoning."""

    event_type: str
    generation: int
    selected_breeder: Optional[str]
    landscape: Optional[LandscapeType]
    reasoning: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"MetaBreedingEvent({self.event_type!r}, gen={self.generation}, "
            f"breeder={self.selected_breeder!r}, landscape={self.landscape})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# BreederPortfolio — manages a collection of breeders with performance history
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BreederRecord:
    """Performance record for a single breeder instance."""

    breeder: BreedingKernel
    preset: BreedingPreset
    qd_scores: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    generations_active: int = 0
    stall_count: int = 0

    def add_qd(self, score: float) -> None:
        self.qd_scores.append(score)

    @property
    def avg_qd(self) -> float:
        if not self.qd_scores:
            return 0.0
        return sum(self.qd_scores) / len(self.qd_scores)

    @property
    def recent_qd_trend(self) -> float:
        """Return the slope of QD-score over recent history (positive = improving)."""
        if len(self.qd_scores) < 3:
            return 0.0
        n = len(self.qd_scores)
        x_mean = (n - 1) / 2
        y_mean = self.avg_qd
        numerator = sum((i - x_mean) * (qd - y_mean) for i, qd in enumerate(self.qd_scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def __repr__(self) -> str:
        return f"BreederRecord({self.breeder.name!r}, preset={self.preset.name}, avg_qd={self.avg_qd:.3f})"


class BreederPortfolio:
    """Manages a collection of breeders with performance history."""

    def __init__(self, evaluator: Any, pop_size: int = 100, gene_dim: int = 5):
        self.evaluator = evaluator
        self.pop_size = pop_size
        self.gene_dim = gene_dim
        self.breeders: Dict[str, BreederRecord] = {}
        self._evaluator_wrapper = evaluator

    def add_preset(self, preset: BreedingPreset, name: Optional[str] = None) -> BreedingKernel:
        """Add a breeder from a preset with a fresh random population."""
        if name is None:
            name = f"{preset.name.lower()}_{len(self.breeders)}"
        population = self._seed_population()
        breeder = BreedingKernel.from_preset(
            preset=preset,
            evaluator=self._evaluator_wrapper,
            population=population,
            pop_size=self.pop_size,
            name=name,
        )
        record = BreederRecord(breeder=breeder, preset=preset)
        self.breeders[name] = record
        return breeder

    def add_breeder(self, breeder: BreedingKernel, preset: BreedingPreset) -> None:
        """Add an existing breeder to the portfolio."""
        record = BreederRecord(breeder=breeder, preset=preset)
        self.breeders[breeder.name] = record

    def update_qd(self, breeder_name: str, qd_score: float) -> None:
        """Record a QD-score observation for a breeder."""
        if breeder_name in self.breeders:
            self.breeders[breeder_name].add_qd(qd_score)

    def get_best_breeder(self, landscape: LandscapeType) -> Tuple[str, BreederRecord]:
        """Select the breeder best suited for the given landscape type.

        Uses a scoring function that blends recent QD performance and landscape match.
        """
        if not self.breeders:
            raise ValueError("Portfolio is empty")

        scores: Dict[str, float] = {}
        for name, record in self.breeders.items():
            base_score = record.avg_qd
            trend_bonus = max(0.0, record.recent_qd_trend * 10)  # Reward improving breeders
            landscape_bonus = self._landscape_match_bonus(record.preset, landscape)
            # Penalize breeders that have been active too long without improvement
            stall_penalty = record.stall_count * 0.5
            scores[name] = base_score + trend_bonus + landscape_bonus - stall_penalty

        best_name = max(scores, key=scores.get)
        return best_name, self.breeders[best_name]

    def _landscape_match_bonus(self, preset: BreedingPreset, landscape: LandscapeType) -> float:
        """Return a bonus for matching preset to landscape."""
        match = {
            BreedingPreset.EXPLOITATION: LandscapeType.SMOOTH,
            BreedingPreset.EXPLORATION: LandscapeType.RUGGED,
            BreedingPreset.DIVERSITY: LandscapeType.MULTIMODAL,
            BreedingPreset.BALANCED: LandscapeType.UNKNOWN,
        }
        if match.get(preset) == landscape:
            return 2.0
        return 0.0

    def _seed_population(self) -> List[Genome]:
        """Generate a random initial population."""
        return [Genome(genes=[random.uniform(-1, 1) for _ in range(self.gene_dim)]) for _ in range(self.pop_size)]

    def __len__(self) -> int:
        return len(self.breeders)

    def __repr__(self) -> str:
        return f"BreederPortfolio(breeders={list(self.breeders.keys())})"


# ═══════════════════════════════════════════════════════════════════════════════
# StallDetector — detects when a breeder has stalled
# ═══════════════════════════════════════════════════════════════════════════════

class StallDetector:
    """Detects fitness plateaus and diversity collapse."""

    def __init__(
        self,
        fitness_window: int = 5,
        fitness_tolerance: float = 1e-4,
        diversity_threshold: float = 0.01,
        min_generations: int = 3,
    ):
        self.fitness_window = fitness_window
        self.fitness_tolerance = fitness_tolerance
        self.diversity_threshold = diversity_threshold
        self.min_generations = min_generations

    def is_stalled(
        self,
        fitness_history: List[float],
        diversity_history: List[float],
        generations_active: int,
    ) -> Tuple[bool, str]:
        """Return (stalled, reason) tuple."""
        if generations_active < self.min_generations:
            return False, "too_early"

        # Fitness plateau detection
        if len(fitness_history) >= self.fitness_window:
            recent = fitness_history[-self.fitness_window:]
            if max(recent) - min(recent) < self.fitness_tolerance:
                return True, "fitness_plateau"

        # Diversity collapse detection
        if diversity_history and diversity_history[-1] < self.diversity_threshold:
            return True, "diversity_collapse"

        # Monotonic decline detection
        if len(fitness_history) >= self.fitness_window:
            recent = fitness_history[-self.fitness_window:]
            if all(recent[i] <= recent[i - 1] for i in range(1, len(recent))):
                return True, "monotonic_decline"

        return False, "active"

    def __repr__(self) -> str:
        return f"StallDetector(window={self.fitness_window}, tol={self.fitness_tolerance})"


# ═══════════════════════════════════════════════════════════════════════════════
# MetaBreeder — the SDA loop controller
# ═══════════════════════════════════════════════════════════════════════════════

class MetaBreeder:
    """Adaptive meta-breeder that selects the optimal breeder for the landscape.

    SDA Loop:
        SENSE  → Analyze landscape from current breeder's history
        DECIDE → Select best breeder from portfolio for landscape
        ACT    → Run one generation with the selected breeder
        ADAPT  → If stall detected, switch breeder (warm-start from archive)
    """

    def __init__(
        self,
        portfolio: BreederPortfolio,
        analyzer: Optional[LandscapeAnalyzer] = None,
        stall_detector: Optional[StallDetector] = None,
        evaluator: Any = None,
        pop_size: int = 100,
        gene_dim: int = 5,
        max_stall_switches: int = 3,
        warm_start_ratio: float = 0.5,
    ):
        self.portfolio = portfolio
        self.analyzer = analyzer or LandscapeAnalyzer()
        self.stall_detector = stall_detector or StallDetector()
        self.evaluator = evaluator
        self.pop_size = pop_size
        self.gene_dim = gene_dim
        self.max_stall_switches = max_stall_switches
        self.warm_start_ratio = warm_start_ratio

        self.current_breeder_name: Optional[str] = None
        self.current_breeder_record: Optional[BreederRecord] = None
        self.generation: int = 0
        self.events: List[MetaBreedingEvent] = []
        self.stall_switch_count: int = 0

        # Initialize with the first breeder if portfolio has one
        if portfolio.breeders:
            self._activate_breeder(next(iter(portfolio.breeders)))

    def _activate_breeder(self, name: str, warm_start_population: Optional[List[Genome]] = None) -> MetaBreedingEvent:
        """Activate a breeder by name, optionally warm-starting from a previous population."""
        if name not in self.portfolio.breeders:
            raise ValueError(f"Unknown breeder: {name}")

        self.current_breeder_name = name
        self.current_breeder_record = self.portfolio.breeders[name]

        if warm_start_population is not None:
            # Warm-start: blend archive/population from previous breeder into new one
            if len(warm_start_population) == 0:
                # Empty warm-start: create all-random population
                self.current_breeder_record.breeder.population = [
                    Genome(genes=[random.uniform(-1, 1) for _ in range(self.gene_dim)])
                    for _ in range(self.pop_size)
                ]
            else:
                n_warm = int(self.pop_size * self.warm_start_ratio)
                n_random = self.pop_size - n_warm
                warm = self._select_diverse_subset(warm_start_population, n_warm)
                random_pop = [Genome(genes=[random.uniform(-1, 1) for _ in range(self.gene_dim)]) for _ in range(n_random)]
                self.current_breeder_record.breeder.population = warm + random_pop

        self.current_breeder_record.generations_active = 0
        self.current_breeder_record.stall_count = 0

        event = MetaBreedingEvent(
            event_type="breeder_activated",
            generation=self.generation,
            selected_breeder=name,
            landscape=None,
            reasoning=f"Activated breeder {name} from portfolio.",
            payload={"warm_start": warm_start_population is not None},
        )
        self.events.append(event)
        return event

    def _select_diverse_subset(self, population: List[Genome], n: int) -> List[Genome]:
        """Greedy selection of a diverse subset from a population."""
        if len(population) <= n:
            return [g.copy() for g in population]

        # Start with the best individual
        sorted_pop = sorted(population, key=lambda g: (g.fitness if g.fitness is not None else -math.inf), reverse=True)
        selected = [sorted_pop[0].copy()]
        remaining = sorted_pop[1:]

        while len(selected) < n and remaining:
            # Pick the individual furthest from the already selected set
            def min_distance(g):
                return min(
                    math.sqrt(sum((a - b) ** 2 for a, b in zip(g.genes, s.genes)))
                    for s in selected
                )

            best = max(remaining, key=min_distance)
            selected.append(best.copy())
            remaining.remove(best)

        return selected

    def _sense(self) -> LandscapeType:
        """SENSE phase: analyze the current landscape."""
        if self.current_breeder_record is None:
            return LandscapeType.UNKNOWN
        breeder = self.current_breeder_record.breeder
        return self.analyzer.analyze(breeder.fitness_history, breeder.diversity_history)

    def _decide(self, landscape: LandscapeType) -> Tuple[str, str]:
        """DECIDE phase: select the best breeder for the landscape."""
        if len(self.portfolio.breeders) == 1:
            name = next(iter(self.portfolio.breeders))
            return name, f"Only one breeder available ({name}), selecting it."

        best_name, _ = self.portfolio.get_best_breeder(landscape)
        return best_name, f"Selected {best_name} for landscape {landscape.name} based on QD-score and landscape match."

    def _act(self) -> BreedingEvent:
        """ACT phase: run one generation with the current breeder."""
        if self.current_breeder_record is None:
            raise RuntimeError("No breeder is currently active")
        event = self.current_breeder_record.breeder.step()
        self.current_breeder_record.generations_active += 1
        self.generation += 1
        return event

    def _adapt(self, landscape: LandscapeType) -> Optional[MetaBreedingEvent]:
        """ADAPT phase: detect stall and switch breeders if needed.

        Returns a MetaBreedingEvent if a switch occurred, otherwise None.
        """
        if self.current_breeder_record is None:
            return None

        breeder = self.current_breeder_record.breeder
        stalled, reason = self.stall_detector.is_stalled(
            breeder.fitness_history,
            breeder.diversity_history,
            self.current_breeder_record.generations_active,
        )

        if not stalled:
            return None

        self.current_breeder_record.stall_count += 1

        if self.stall_switch_count >= self.max_stall_switches:
            event = MetaBreedingEvent(
                event_type="stall_limit_reached",
                generation=self.generation,
                selected_breeder=self.current_breeder_name,
                landscape=landscape,
                reasoning=f"Stall detected ({reason}) but max stall switches ({self.max_stall_switches}) reached. Continuing with current breeder.",
                payload={"stall_reason": reason, "stall_count": self.current_breeder_record.stall_count},
            )
            self.events.append(event)
            return event

        # Select a different breeder (warm-start from current archive + population)
        warm_pop = breeder.population + breeder.archive
        next_name, select_reason = self._decide(landscape)

        if next_name == self.current_breeder_name:
            # Force switch to a different breeder if possible
            other_names = [n for n in self.portfolio.breeders if n != self.current_breeder_name]
            if other_names:
                next_name = random.choice(other_names)
                select_reason = f"Forced switch to {next_name} because current breeder stalled ({reason})."

        self.stall_switch_count += 1
        event = self._activate_breeder(next_name, warm_start_population=warm_pop)
        event.event_type = "breeder_switched"
        event.landscape = landscape
        event.reasoning = f"Stall detected: {reason}. {select_reason}"
        event.payload.update({"stall_reason": reason, "switch_count": self.stall_switch_count})
        return event

    def step(self) -> List[Any]:
        """Run one full SDA cycle and return emitted events.

        Returns a list of BreedingEvent and/or MetaBreedingEvent objects.
        """
        events: List[Any] = []

        # SENSE
        landscape = self._sense()

        # DECIDE (if no breeder active, or if we should re-evaluate)
        if self.current_breeder_name is None:
            next_name, reasoning = self._decide(landscape)
            self._activate_breeder(next_name)
            events.append(self.events[-1])

        # ACT
        breeding_event = self._act()
        events.append(breeding_event)

        # Update portfolio QD tracking
        if self.current_breeder_record is not None:
            qd = self.current_breeder_record.breeder.qd_score
            self.portfolio.update_qd(self.current_breeder_name, qd)

        # ADAPT
        adapt_event = self._adapt(landscape)
        if adapt_event is not None:
            events.append(adapt_event)

        return events

    def run(self, generations: int = 10) -> List[Any]:
        """Run multiple generations."""
        all_events = []
        for _ in range(generations):
            all_events.extend(self.step())
        return all_events

    def __repr__(self) -> str:
        return f"MetaBreeder(gen={self.generation}, active={self.current_breeder_name}, portfolio={len(self.portfolio)})"
