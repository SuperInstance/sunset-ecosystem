"""DreamingLoop — Speculative breeding during idle cycles.

When the fleet is idle (no active tasks), it runs "what if" experiments
using the diversity archive as a memory of past experiments. Results are
stored as "dreams" for reuse when real tasks arrive.

Architecture
------------
    SENSE  →  DECIDE  →  ACT  →  STORE
      ↓         ↓        ↓        ↓
   idle?    dream     run     archive
   check    hypothesis breeder  dream

Key classes
-----------
- ``Dream`` — stored speculative result
- ``DreamArchive`` — CRDT-like storage with search/retrieval
- ``DreamingLoop`` — SDA loop orchestrator
- ``DreamMatcher`` — matches incoming tasks to stored dreams
- ``HypothesisGenerator`` — generates "what if" scenarios

Usage
-----
    archive = DreamArchive(max_size=1000)
    loop = DreamingLoop(
        archive=archive,
        kernel=BreedingKernel.from_preset(BreedingPreset.TOURNAMENT),
        idle_threshold_ms=5000,
    )
    loop.tick()  # runs a dream step if idle
"""
from __future__ import annotations

__all__ = [
    "Dream",
    "DreamArchive",
    "DreamingLoop",
    "DreamMatcher",
    "HypothesisGenerator",
    "IdleDetector",
]

import hashlib
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Dream ─────────────────────────────────────────────────────────

@dataclass
class Dream:
    """A stored speculative breeding result.

    Attributes:
        dream_id: Unique identifier (UUID4).
        tags: Semantic tags describing the hypothesis (e.g., ["hamiltonian", "pythagorean"]).
        hypothesis: Human-readable description of the "what if" scenario.
        breeder_preset: Which BreedingPreset was used.
        population_size: Size of the speculative population.
        generations: How many generations were run.
        best_fitness: Best fitness achieved in the dream.
        mean_fitness: Mean fitness across final population.
        diversity: Diversity metric (std of fitness) of final population.
        created_at: Unix timestamp when the dream was created.
        accessed_at: Unix timestamp of last retrieval.
        access_count: How many times this dream was retrieved.
        raw_result: Optional raw population / genome data.
        metadata: Extra key-value pairs.
    """

    dream_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: List[str] = field(default_factory=list)
    hypothesis: str = ""
    breeder_preset: str = "tournament"
    population_size: int = 0
    generations: int = 0
    best_fitness: float = 0.0
    mean_fitness: float = 0.0
    diversity: float = 0.0
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    raw_result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Mark this dream as accessed."""
        self.accessed_at = time.time()
        self.access_count += 1

    def score_relevance(self, task_tags: List[str]) -> float:
        """Return Jaccard-like overlap between dream tags and task tags."""
        if not self.tags or not task_tags:
            return 0.0
        s_dream = set(self.tags)
        s_task = set(task_tags)
        inter = len(s_dream & s_task)
        union = len(s_dream | s_task)
        return inter / union if union else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dream_id": self.dream_id,
            "tags": self.tags,
            "hypothesis": self.hypothesis,
            "breeder_preset": self.breeder_preset,
            "population_size": self.population_size,
            "generations": self.generations,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "diversity": self.diversity,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }


# ── IdleDetector ──────────────────────────────────────────────

class IdleDetector(ABC):
    """Abstract strategy for detecting fleet idleness."""

    @abstractmethod
    def is_idle(self) -> bool:
        """Return True if the fleet is currently idle."""
        ...


class TimeSinceLastTaskIdleDetector(IdleDetector):
    """Idle when no task has been seen for ``threshold_ms``."""

    def __init__(self, threshold_ms: float = 5000.0):
        self.threshold_ms = threshold_ms
        self._last_task_time = time.time()

    def on_task(self) -> None:
        """Call this when a real task arrives."""
        self._last_task_time = time.time()

    def is_idle(self) -> bool:
        elapsed = (time.time() - self._last_task_time) * 1000
        return elapsed >= self.threshold_ms


class AlwaysBusyIdleDetector(IdleDetector):
    """Never idle — useful for testing or disabling dreams."""

    def is_idle(self) -> bool:
        return False


class AlwaysIdleIdleDetector(IdleDetector):
    """Always idle — useful for testing or forcing dream mode."""

    def is_idle(self) -> bool:
        return True


# ── DreamArchive ───────────────────────────────────────────────

class DreamArchive:
    """CRDT-like storage for dreams with search, retrieval, and pruning.

    Behaves like an append-only set with automatic eviction when
    ``max_size`` is exceeded. Supports tag-based search and LRU-like
    pruning based on ``access_count`` and ``accessed_at``.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._dreams: Dict[str, Dream] = {}
        self._tag_index: Dict[str, set] = {}  # tag -> dream_ids

    def __len__(self) -> int:
        return len(self._dreams)

    def add(self, dream: Dream) -> None:
        """Store a dream; evict oldest if at capacity."""
        if len(self._dreams) >= self.max_size:
            self._prune_one()
        self._dreams[dream.dream_id] = dream
        for tag in dream.tags:
            self._tag_index.setdefault(tag, set()).add(dream.dream_id)

    def get(self, dream_id: str) -> Optional[Dream]:
        """Retrieve a dream by ID and touch it."""
        dream = self._dreams.get(dream_id)
        if dream:
            dream.touch()
        return dream

    def find_by_tags(self, tags: List[str]) -> List[Dream]:
        """Return all dreams matching any of the provided tags."""
        if not tags:
            return []
        ids = set()
        for tag in tags:
            ids |= self._tag_index.get(tag, set())
        dreams = [self._dreams[did] for did in ids]
        for d in dreams:
            d.touch()
        return dreams

    def search(self, query_tags: List[str], top_k: int = 5) -> List[Tuple[Dream, float]]:
        """Return top-k dreams sorted by tag relevance (Jaccard overlap)."""
        scored = []
        for dream in self._dreams.values():
            score = dream.score_relevance(query_tags)
            if score > 0:
                scored.append((dream, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def list_all(self) -> List[Dream]:
        """Return all dreams (no ordering guarantee)."""
        return list(self._dreams.values())

    def prune_old(self, max_age_seconds: float) -> int:
        """Remove dreams older than ``max_age_seconds``; return count removed."""
        cutoff = time.time() - max_age_seconds
        to_remove = [
            did for did, d in self._dreams.items()
            if d.created_at < cutoff
        ]
        for did in to_remove:
            self._remove(did)
        return len(to_remove)

    def _prune_one(self) -> None:
        """Evict the single least-valuable dream.

        Heuristic: lowest (access_count * recency_factor)."""
        if not self._dreams:
            return
        now = time.time()
        worst_id = None
        worst_score = float("inf")
        for did, d in self._dreams.items():
            age = now - d.accessed_at
            recency = 1.0 / (1.0 + age)
            score = d.access_count * recency
            if score < worst_score:
                worst_score = score
                worst_id = did
        if worst_id:
            self._remove(worst_id)

    def _remove(self, dream_id: str) -> None:
        dream = self._dreams.pop(dream_id, None)
        if dream:
            for tag in dream.tags:
                self._tag_index.get(tag, set()).discard(dream_id)

    def merge(self, other: "DreamArchive") -> "DreamArchive":
        """CRDT-like merge: union of both archives, capped at max_size.

        Returns a *new* archive containing the merged state.
        """
        merged = DreamArchive(max_size=max(self.max_size, other.max_size))
        all_dreams = list(self._dreams.values()) + list(other._dreams.values())
        # Sort by access_count descending, then accessed_at descending
        all_dreams.sort(key=lambda d: (d.access_count, d.accessed_at), reverse=True)
        for dream in all_dreams[:merged.max_size]:
            merged.add(dream)
        return merged

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_size": self.max_size,
            "count": len(self._dreams),
            "dreams": [d.to_dict() for d in self._dreams.values()],
        }


# ── HypothesisGenerator ─────────────────────────────────────────

class HypothesisGenerator:
    """Generates "what if" breeding scenarios from a diversity archive.

    The diversity archive is treated as a memory of past experiments.
    Each hypothesis combines two past experiment tags or proposes a
    novel crossover of breeder presets.
    """

    PRESETS = [
        "tournament",
        "pythagorean",
        "spectral",
        "hamiltonian",
        "bounded",
        "causal",
        "information_theoretic",
        "adversarial",
        "nca",
        "gnn",
        "meta_learning",
        "swarm_intelligence",
        "differential",
        "spatial",
        "ensemble",
        "sim_real",
        "bft_qd",
        "constraint",
    ]

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate(
        self,
        memory_tags: List[List[str]],
        n_hypotheses: int = 1,
    ) -> List[Dict[str, Any]]:
        """Return a list of hypothesis dicts.

        Each hypothesis contains:
        - tags: list of tags
        - hypothesis: human-readable description
        - breeder_preset: suggested preset
        - population_size: suggested small population
        - generations: suggested short generations
        """
        if not memory_tags:
            return self._generate_novel(n_hypotheses)

        flat_tags = [t for sub in memory_tags for t in sub]
        if not flat_tags:
            return self._generate_novel(n_hypotheses)

        results = []
        for _ in range(n_hypotheses):
            tag_pool = list(set(flat_tags))
            self.rng.shuffle(tag_pool)
            n_tags = self.rng.randint(1, min(3, len(tag_pool)))
            tags = sorted(tag_pool[:n_tags])

            preset = self.rng.choice(self.PRESETS)
            pop_size = self.rng.randint(10, 40)
            gens = self.rng.randint(5, 25)

            desc = self._describe(tags, preset, pop_size, gens)
            results.append({
                "tags": tags,
                "hypothesis": desc,
                "breeder_preset": preset,
                "population_size": pop_size,
                "generations": gens,
            })
        return results

    def _generate_novel(self, n: int) -> List[Dict[str, Any]]:
        """Generate purely random hypotheses when no memory exists."""
        results = []
        for _ in range(n):
            preset = self.rng.choice(self.PRESETS)
            tags = [preset, "speculative"]
            pop_size = self.rng.randint(10, 40)
            gens = self.rng.randint(5, 25)
            results.append({
                "tags": tags,
                "hypothesis": f"What if we run {preset} with a fresh population?",
                "breeder_preset": preset,
                "population_size": pop_size,
                "generations": gens,
            })
        return results

    def _describe(self, tags: List[str], preset: str, pop_size: int, gens: int) -> str:
        if len(tags) >= 2:
            return (
                f"What if we breed {preset} combining "
                f"{' + '.join(tags)} with population={pop_size} for {gens} generations?"
            )
        return f"What if we breed {preset} on {tags[0]} with population={pop_size} for {gens} generations?"


# ── DreamMatcher ────────────────────────────────────────────────

class DreamMatcher:
    """Matches incoming real-world tasks to stored dreams by tag similarity.

    Provides a ``match`` method that returns the best dream and a
    confidence score. If no dream exceeds ``min_score``, returns None.
    """

    def __init__(self, archive: DreamArchive, min_score: float = 0.25):
        self.archive = archive
        self.min_score = min_score

    def match(self, task_tags: List[str]) -> Optional[Tuple[Dream, float]]:
        """Return the best matching dream and its score, or None."""
        candidates = self.archive.search(task_tags, top_k=10)
        if not candidates:
            return None
        best_dream, best_score = candidates[0]
        if best_score < self.min_score:
            return None
        best_dream.touch()
        return best_dream, best_score

    def match_all(self, task_tags: List[str]) -> List[Tuple[Dream, float]]:
        """Return all matches above ``min_score``, sorted by score descending."""
        candidates = self.archive.search(task_tags, top_k=len(self.archive))
        return [(d, s) for d, s in candidates if s >= self.min_score]


# ── DreamingLoop ────────────────────────────────────────────────

class _BreedingKernelLike(Protocol):
    """Protocol for anything that looks like a BreedingKernel."""

    def run(
        self, population: List[Any], generations: int = 100, mutation_rate: float = 0.1
    ) -> Iterator[Any]:
        ...


class DreamingLoop:
    """SDA loop: SENSE idle → DECIDE dream → ACT execute → STORE result.

    Args:
        archive: DreamArchive for storing / retrieving dreams.
        kernel: BreedingKernel (or compatible) for executing dreams.
        idle_detector: Strategy for detecting idleness.
        hypothesis_generator: Generates speculative hypotheses.
        idle_threshold_ms: Deprecated; use ``idle_detector`` instead.
        max_concurrent_dreams: Maximum number of dreams to store per tick.
        dream_population_size: Population size for speculative runs.
        dream_generations: Generations per speculative run.
    """

    def __init__(
        self,
        archive: DreamArchive,
        kernel: Optional[_BreedingKernelLike] = None,
        idle_detector: Optional[IdleDetector] = None,
        hypothesis_generator: Optional[HypothesisGenerator] = None,
        idle_threshold_ms: Optional[float] = None,
        max_concurrent_dreams: int = 1,
        dream_population_size: int = 20,
        dream_generations: int = 10,
    ):
        self.archive = archive
        self.kernel = kernel
        self.idle_detector = idle_detector or TimeSinceLastTaskIdleDetector(
            threshold_ms=idle_threshold_ms or 5000.0
        )
        self.generator = hypothesis_generator or HypothesisGenerator()
        self.max_concurrent_dreams = max_concurrent_dreams
        self.dream_population_size = dream_population_size
        self.dream_generations = dream_generations

        self._dreams_this_tick: List[Dream] = []
        self._total_dreams_run: int = 0
        self._total_ticks: int = 0
        self._is_busy: bool = False

    # -- SENSE ---------------------------------------------------

    def sense_idle(self) -> bool:
        """Return True if the fleet is currently idle."""
        return self.idle_detector.is_idle() and not self._is_busy

    # -- DECIDE --------------------------------------------------

    def decide_dream(self) -> Optional[Dream]:
        """Generate a speculative dream hypothesis.

        Returns a Dream object with no result yet (raw_result=None).
        """
        # Build memory from existing archive tags
        memory_tags = [d.tags for d in self.archive.list_all()]
        hypotheses = self.generator.generate(memory_tags, n_hypotheses=1)
        if not hypotheses:
            return None
        h = hypotheses[0]
        return Dream(
            tags=h["tags"],
            hypothesis=h["hypothesis"],
            breeder_preset=h["breeder_preset"],
            population_size=h["population_size"],
            generations=h["generations"],
        )

    # -- ACT -----------------------------------------------------

    def act_execute(self, dream: Dream) -> Dream:
        """Execute the dream using the breeding kernel.

        If no kernel is available, returns the dream with synthetic
        metrics for testing purposes.
        """
        self._is_busy = True
        try:
            if self.kernel is None:
                # Synthetic fallback — deterministic from tags for tests
                dream.best_fitness = self._synthetic_fitness(dream)
                dream.mean_fitness = dream.best_fitness * 0.8
                dream.diversity = dream.best_fitness * 0.1
                dream.raw_result = {"synthetic": True}
                return dream

            # Build a small random population
            population = self._make_population(dream.population_size)
            best = -float("inf")
            mean = 0.0
            div = 0.0
            count = 0
            for event in self.kernel.run(population, generations=dream.generations):
                best = max(best, event.best_fitness)
                mean = event.mean_fitness
                div = event.diversity
                count += 1
            dream.best_fitness = best if best != -float("inf") else 0.0
            dream.mean_fitness = mean
            dream.diversity = div
            dream.raw_result = {"events": count}
            return dream
        finally:
            self._is_busy = False

    def _make_population(self, size: int) -> List[Any]:
        """Build a simple float-vector population for generic kernels."""
        return [np.random.randn(8).astype(np.float32).tolist() for _ in range(size)]

    def _synthetic_fitness(self, dream: Dream) -> float:
        """Deterministic synthetic fitness from tag hash for reproducible tests."""
        tag_str = "".join(sorted(dream.tags))
        h = hashlib.sha256(tag_str.encode()).hexdigest()
        val = int(h[:8], 16) / 0xFFFFFFFF
        return round(val, 6)

    # -- STORE ---------------------------------------------------

    def store_result(self, dream: Dream) -> None:
        """Persist a completed dream into the archive."""
        self.archive.add(dream)
        self._dreams_this_tick.append(dream)
        self._total_dreams_run += 1

    # -- Full SDA loop --------------------------------------------

    def tick(self) -> Optional[Dream]:
        """Run one full SDA cycle if idle.

        Returns the completed Dream, or None if not idle or no dream generated.
        """
        self._total_ticks += 1
        self._dreams_this_tick = []
        if not self.sense_idle():
            return None
        dream = self.decide_dream()
        if dream is None:
            return None
        completed = self.act_execute(dream)
        self.store_result(completed)
        return completed

    def run(self, max_dreams: int = 1) -> List[Dream]:
        """Run up to ``max_dreams`` dreams sequentially.

        Returns the list of completed dreams.
        """
        results: List[Dream] = []
        for _ in range(max_dreams):
            d = self.tick()
            if d is None:
                break
            results.append(d)
        return results

    # -- Task matching --------------------------------------------

    def find_dream_for_task(self, task_tags: List[str], matcher: Optional[DreamMatcher] = None) -> Optional[Tuple[Dream, float]]:
        """Check if a stored dream matches an incoming real task."""
        m = matcher or DreamMatcher(self.archive)
        return m.match(task_tags)

    # -- Stats ----------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "total_ticks": self._total_ticks,
            "total_dreams_run": self._total_dreams_run,
            "archive_size": len(self.archive),
            "max_concurrent_dreams": self.max_concurrent_dreams,
            "dream_population_size": self.dream_population_size,
            "dream_generations": self.dream_generations,
            "is_busy": self._is_busy,
        }
