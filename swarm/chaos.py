"""Chaos Injection — Stochastic exploration for Hebbian routing.

Prevents routes from settling into local optima by occasionally
swapping or re-routing connections. The chaos rate decays as the
system's adaptation score increases — less exploration needed once
patterns are well-established.
"""

from __future__ import annotations

__all__ = ["ChaosProbability", "ChaosEvent", "inject_chaos"]

import random
import time
from dataclasses import dataclass, field


@dataclass
class ChaosProbability:
    """Decaying chaos probability for route exploration.

    Starts high (lots of exploration), decays as adaptation increases.
    Never goes below minimum — some chaos always remains.

    Attributes:
        initial: Starting chaos probability.
        decay: Multiplicative decay factor per adaptation step.
        minimum: Floor for chaos probability.
        current: Current chaos probability.
    """
    initial: float = 0.3
    decay: float = 0.95
    minimum: float = 0.01
    current: float = 0.3

    def __repr__(self) -> str:
        return f"ChaosProbability(current={self.current:.4f}, min={self.minimum})"

    def update(self, adaptation_score: float) -> float:
        """Decay chaos based on adaptation score (0.0-1.0).

        Higher adaptation → more decay → less chaos.
        The decay is proportional to how adapted the system is.

        Args:
            adaptation_score: System adaptation score (0.0 = fresh, 1.0 = fully adapted).

        Returns:
            Updated chaos probability.
        """
        # Scale decay by adaptation: more adapted = faster decay
        effective_decay = self.decay ** (1.0 + adaptation_score)
        self.current = max(self.minimum, self.current * effective_decay)
        return self.current

    def reset(self) -> None:
        """Reset chaos to initial level (e.g., after a major route restructuring)."""
        self.current = self.initial


@dataclass
class ChaosEvent:
    """Record of a single chaos intervention.

    Attributes:
        original_route: The route that was originally planned.
        new_route: The route chaos substituted.
        reason: Why the chaos fired (swap, reroute, random).
        timestamp: When this event occurred.
    """
    original_route: str
    new_route: str
    reason: str
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"ChaosEvent({self.original_route}→{self.new_route}, "
            f"reason={self.reason!r})"
        )


def inject_chaos(
    routes: dict[str, list[str]],
    chaos_prob: ChaosProbability,
    adaptation_score: float = 0.0,
) -> tuple[dict[str, list[str]], list[ChaosEvent]]:
    """Inject chaos into a routing map by randomly swapping destinations.

    For each source, each destination has a probability (chaos_prob.current)
    of being swapped with a destination from another source. This prevents
    the Hebbian system from getting stuck in local optima.

    The chaos rate is updated based on adaptation_score before injection.

    Args:
        routes: Mapping of source_id → list of destination_ids.
        chaos_prob: ChaosProbability instance controlling injection rate.
        adaptation_score: Current system adaptation (0.0-1.0).

    Returns:
        Tuple of (modified_routes, list of ChaosEvents recording what changed).
    """
    # Update chaos probability based on adaptation
    prob = chaos_prob.update(adaptation_score)

    # Deep copy to avoid mutating input
    new_routes: dict[str, list[str]] = {
        src: list(dsts) for src, dsts in routes.items()
    }
    events: list[ChaosEvent] = []

    # Collect all destinations for potential swaps
    all_destinations: list[str] = []
    for dsts in routes.values():
        all_destinations.extend(dsts)

    if not all_destinations:
        return new_routes, events

    sources = list(routes.keys())

    for src in sources:
        destinations = new_routes[src]
        for i, dst in enumerate(destinations):
            if random.random() >= prob:
                continue

            original = dst

            # Decide chaos action: swap with another source's dest, or reroute
            roll = random.random()

            if roll < 0.5 and len(sources) > 1:
                # Swap: pick a destination from a different source
                other_srcs = [s for s in sources if s != src]
                if other_srcs:
                    other_src = random.choice(other_srcs)
                    other_dsts = new_routes[other_src]
                    if other_dsts:
                        j = random.randrange(len(other_dsts))
                        # Swap the two destinations
                        new_routes[src][i] = other_dsts[j]
                        new_routes[other_src][j] = original
                        events.append(ChaosEvent(
                            original_route=f"{src}→{original}",
                            new_route=f"{src}→{other_dsts[j]}",
                            reason="swap",
                        ))
                        events.append(ChaosEvent(
                            original_route=f"{other_src}→{other_dsts[j]}",
                            new_route=f"{other_src}→{original}",
                            reason="swap",
                        ))
            else:
                # Reroute: pick a random destination from the pool
                candidates = [d for d in all_destinations if d != dst]
                if candidates:
                    new_dst = random.choice(candidates)
                    new_routes[src][i] = new_dst
                    events.append(ChaosEvent(
                        original_route=f"{src}→{original}",
                        new_route=f"{src}→{new_dst}",
                        reason="reroute",
                    ))

    return new_routes, events
