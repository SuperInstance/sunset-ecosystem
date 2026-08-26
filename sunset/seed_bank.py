"""Seed bank — manages onboarding docs for next generation selection."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from sunset.sunset_documents import Onboarding


@dataclass
class SeedEntry:
    """A single onboarding document stored in the seed bank."""

    onboarding: Onboarding
    relevance: float = 0.5
    novelty: float = 0.5
    times_selected: int = 0

    def __repr__(self) -> str:
        return (
            f"SeedEntry(agent={self.onboarding.agent_id!r}, "
            f"variant={self.onboarding.variant!r}, "
            f"weight={self.weight:.4f})"
        )

    @property
    def weight(self) -> float:
        """Selection weight: relevance * novelty, decayed by usage."""
        decay = 1.0 / (1.0 + self.times_selected)
        return self.relevance * self.novelty * decay


class SeedBank:
    """Manages onboarding documents for spawning new agents.

    Stores onboardings tagged by parent, generation, and variant.
    Selects onboardings weighted by relevance + novelty.
    Supports cross-breeding and mutation.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, SeedEntry] = {}
        self._by_parent: Dict[str, List[str]] = {}
        self._by_generation: Dict[int, List[str]] = {}

    def __repr__(self) -> str:
        return f"SeedBank(entries={len(self._entries)})"

    def store(
        self, onboarding: Onboarding, relevance: float = 0.5, novelty: float = 0.5
    ) -> str:
        """Store an onboarding document. Returns the entry key."""
        key = f"{onboarding.agent_id}:{onboarding.variant}:{id(onboarding)}"
        entry = SeedEntry(onboarding=onboarding, relevance=relevance, novelty=novelty)
        self._entries[key] = entry

        parent = onboarding.parent_id or onboarding.agent_id
        self._by_parent.setdefault(parent, []).append(key)
        self._by_generation.setdefault(onboarding.generation, []).append(key)
        return key

    def select(self, n: int = 1, generation: Optional[int] = None) -> List[Onboarding]:
        """Select n onboardings weighted by relevance + novelty."""
        candidates = self._filtered_entries(generation)
        if not candidates:
            return []

        keys, weights = zip(*((k, e.weight) for k, e in candidates))
        total = sum(weights)
        if total == 0:
            return [
                candidates[i][1].onboarding
                for i in random.sample(range(len(candidates)), min(n, len(candidates)))
            ]

        probs = [w / total for w in weights]
        chosen_keys = set()
        result: List[Onboarding] = []

        for _ in range(min(n, len(candidates))):
            available = [(k, p) for k, p in zip(keys, probs) if k not in chosen_keys]
            if not available:
                break
            ks, ps = zip(*available)
            ps_norm = [p / sum(ps) for p in ps]
            pick = random.choices(list(ks), weights=ps_norm, k=1)[0]
            chosen_keys.add(pick)
            self._entries[pick].times_selected += 1
            result.append(self._entries[pick].onboarding)

        return result

    def cross_breed(self, parent_ids: List[str], n: int = 1) -> List[List[Onboarding]]:
        """Create n child bundles, each getting onboardings from multiple parents."""
        bundles: List[List[Onboarding]] = []
        for _ in range(n):
            bundle: List[Onboarding] = []
            for pid in parent_ids:
                keys = self._by_parent.get(pid, [])
                if keys:
                    pick = random.choice(keys)
                    bundle.append(self._entries[pick].onboarding)
            bundles.append(bundle)
        return bundles

    def mutate(self, fallback: Optional[Onboarding] = None) -> Optional[Onboarding]:
        """Occasionally pick a wild onboarding — maximum novelty."""
        if not self._entries:
            return fallback
        entries = sorted(self._entries.values(), key=lambda e: e.novelty, reverse=True)
        pick = entries[0]
        pick.times_selected += 1
        return pick.onboarding

    def _filtered_entries(
        self, generation: Optional[int] = None
    ) -> List[Tuple[str, SeedEntry]]:
        if generation is None:
            return list(self._entries.items())
        keys = set(self._by_generation.get(generation, []))
        return [(k, v) for k, v in self._entries.items() if k in keys]


__all__ = ["SeedBank", "SeedEntry"]
