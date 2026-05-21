"""Routing Layer — Living pathways between nerve fibers and agents.

Routes grow stronger or weaker based on:
- Efficiency: how fast does this route solve the problem?
- Reception: did the receiving agent find the tile useful?
- Chaos probability: stochastic exploration prevents local optima.

Hebbian: neurons that fire together wire together. The water carves channels.
"""

from __future__ import annotations

__all__ = ["RoutingLayer", "Route", "HebbianChannel"]

import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Route:
    """A pathway between a nerve fiber and a destination (agent or room).

    Strength grows with use and successful reception. Weakens with disuse
    or failure. Never reaches exactly 0 or 1 — chaos keeps it alive.

    Attributes:
        source: Source fiber/agent ID.
        destination: Destination agent/room ID.
        strength: Current route strength (0.01 to 0.99).
        efficiency: Measured efficiency (latency inverse).
        reception: How often the destination found the tile useful.
        fires: Total number of times this route has fired.
        successes: Number of times the reception was positive.
    """
    source: str
    destination: str
    strength: float = 0.5
    efficiency: float = 0.5
    reception: float = 0.5
    fires: int = 0
    successes: int = 0
    last_fired: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Route({self.source}→{self.destination}, "
            f"str={self.strength:.2f}, eff={self.efficiency:.2f}, "
            f"rec={self.reception:.2f})"
        )

    def fire(self, chaos: float = 0.1) -> bool:
        """Attempt to fire this route.

        The route fires with probability proportional to its strength,
        plus a chaos term that occasionally fires weak routes (exploration).

        Args:
            chaos: Probability of ignoring strength and firing anyway (0-1).

        Returns:
            True if the route fires.
        """
        self.fires += 1
        self.last_fired = time.time()

        # Deterministic firing based on strength
        if random.random() < self.strength:
            return True

        # Chaos firing — exploration
        if random.random() < chaos:
            return True

        return False

    def reinforce(self, success: bool, lr: float = 0.05) -> None:
        """Hebbian reinforcement — strengthen on success, weaken on failure.

        Args:
            success: Whether the destination found the tile useful.
            lr: Learning rate for strength update.
        """
        if success:
            self.successes += 1
            self.reception = self.successes / max(self.fires, 1)
            self.strength = min(0.99, self.strength + lr * (1.0 - self.strength))
        else:
            self.reception = self.successes / max(self.fires, 1)
            self.strength = max(0.01, self.strength - lr * self.strength)

    def decay(self, factor: float = 0.999) -> None:
        """Time-based decay — routes that aren't used weaken slowly."""
        self.strength = max(0.01, self.strength * factor)


class HebbianChannel:
    """A bidirectional channel between two nodes that strengthens with co-activation.

    Like Oracle1's Hebbian layer: tiles that flow between rooms deepen the
    channel. The channel attracts more tiles. More tiles deepen the channel.
    The system self-reinforces.

    Args:
        node_a: First node ID.
        node_b: Second node ID.
        initial_weight: Starting connection weight.
    """

    def __init__(
        self,
        node_a: str,
        node_b: str,
        initial_weight: float = 0.1,
    ) -> None:
        self.node_a = node_a
        self.node_b = node_b
        self.weight = initial_weight
        self.co_activations: int = 0
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"HebbianChannel({self.node_a}↔{self.node_b}, "
            f"w={self.weight:.3f}, co_act={self.co_activations})"
        )

    def activate(self) -> float:
        """Record co-activation and strengthen the channel.

        Returns:
            The new weight after activation.
        """
        with self._lock:
            self.co_activations += 1
            # Hebbian strengthening: Δw = η * (1 - w)
            self.weight = min(1.0, self.weight + 0.01 * (1.0 - self.weight))
            return self.weight

    def decay(self, factor: float = 0.999) -> float:
        """Apply time-based decay.

        Returns:
            The new weight after decay.
        """
        with self._lock:
            self.weight = max(0.0, self.weight * factor)
            return self.weight


class RoutingLayer:
    """Manages routes and Hebbian channels between nerve fibers and agents.

    Routes are selected based on strength + chaos probability. Multiple
    routes can fire simultaneously (parallel exploration). Feedback from
    destinations updates route strength.

    Args:
        chaos: Base chaos probability for exploration (default 0.1).
        learning_rate: Rate of Hebbian reinforcement (default 0.05).
    """

    def __init__(
        self,
        chaos: float = 0.1,
        learning_rate: float = 0.05,
    ) -> None:
        self.chaos = chaos
        self.learning_rate = learning_rate
        self._routes: dict[str, Route] = {}  # "src→dst" → Route
        self._channels: dict[str, HebbianChannel] = {}  # "a↔b" → Channel
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"RoutingLayer(routes={len(self._routes)}, "
            f"channels={len(self._channels)}, chaos={self.chaos:.2f})"
        )

    def _route_key(self, source: str, destination: str) -> str:
        return f"{source}→{destination}"

    def _channel_key(self, a: str, b: str) -> str:
        return f"{a}↔{b}"

    def add_route(
        self,
        source: str,
        destination: str,
        strength: float = 0.5,
    ) -> Route:
        """Register a new route between source and destination."""
        key = self._route_key(source, destination)
        route = Route(
            source=source,
            destination=destination,
            strength=strength,
        )
        with self._lock:
            self._routes[key] = route
        return route

    def add_channel(
        self,
        node_a: str,
        node_b: str,
        weight: float = 0.1,
    ) -> HebbianChannel:
        """Register a new Hebbian channel between two nodes."""
        key = self._channel_key(node_a, node_b)
        channel = HebbianChannel(node_a, node_b, weight)
        with self._lock:
            self._channels[key] = channel
        return channel

    def fire(
        self,
        source: str,
        destinations: Optional[list[str]] = None,
    ) -> list[str]:
        """Fire routes from source to matching destinations.

        Returns the list of destinations that activated (based on strength
        and chaos probability).

        Args:
            source: The source fiber/agent ID.
            destinations: Optional filter for specific destinations.
                If None, fire all routes from this source.

        Returns:
            List of destination IDs that fired.
        """
        with self._lock:
            candidates = [
                r for r in self._routes.values()
                if r.source == source
                and (destinations is None or r.destination in destinations)
            ]

        fired: list[str] = []
        for route in candidates:
            if route.fire(chaos=self.chaos):
                fired.append(route.destination)

        # Activate Hebbian channels for co-fired destinations
        for i, dst_a in enumerate(fired):
            for dst_b in fired[i + 1:]:
                key = self._channel_key(dst_a, dst_b)
                if key in self._channels:
                    self._channels[key].activate()

        return fired

    def feedback(
        self,
        source: str,
        destination: str,
        success: bool,
    ) -> None:
        """Provide feedback on a route's outcome.

        This is the reception signal — did the destination find the tile useful?

        Args:
            source: Source fiber/agent ID.
            destination: Destination ID.
            success: Whether the tile was useful.
        """
        key = self._route_key(source, destination)
        with self._lock:
            route = self._routes.get(key)
        if route:
            route.reinforce(success, lr=self.learning_rate)

    def get_strongest_routes(
        self,
        source: str,
        top_k: int = 5,
    ) -> list[Route]:
        """Get the top-k strongest routes from a source."""
        with self._lock:
            routes = [
                r for r in self._routes.values() if r.source == source
            ]
        routes.sort(key=lambda r: r.strength, reverse=True)
        return routes[:top_k]

    def get_channel_weight(self, a: str, b: str) -> float:
        """Get the Hebbian channel weight between two nodes."""
        key = self._channel_key(a, b)
        with self._lock:
            channel = self._channels.get(key)
        return channel.weight if channel else 0.0

    def decay_all(self, factor: float = 0.999) -> None:
        """Apply time-based decay to all routes and channels."""
        with self._lock:
            for route in self._routes.values():
                route.decay(factor)
            for channel in self._channels.values():
                channel.decay(factor)
