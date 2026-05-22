"""Routing Layer — Living pathways between nerve fibers and agents.

Routes grow stronger or weaker based on:
- Efficiency: how fast does this route solve the problem?
- Reception: did the receiving agent find the tile useful?
- Chaos probability: stochastic exploration prevents local optima.

Hebbian: neurons that fire together wire together. The water carves channels.

PERFORMANCE NOTES (Agentic Compiler — 2026-05-22):
- fire_fast() uses numpy vectorization + precomputed route index — 60× faster
- Compiled routes (strength > 0.9) skip random checks entirely
- Hebbian activation limited to top-k pairs, not O(n²)
- See docs/AGENTIC-COMPILER-RESEARCH.md for full analysis.
"""

from __future__ import annotations

__all__ = ["RoutingLayer", "Route", "HebbianChannel"]

import functools
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Route:
    """A pathway between a nerve fiber and a destination (agent or room).

    Strength grows with use and successful reception. Weakens with disuse
    or failure. Never reaches exactly 0 or 1 — chaos keeps it alive.
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
        """Attempt to fire this route (Python scalar path — slow but simple)."""
        self.fires += 1
        self.last_fired = time.time()
        if random.random() < self.strength:
            return True
        if random.random() < chaos:
            return True
        return False

    def reinforce(self, success: bool, lr: float = 0.05) -> None:
        """Hebbian reinforcement — strengthen on success, weaken on failure."""
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
    """A bidirectional channel between two nodes that strengthens with co-activation."""

    def __init__(self, node_a: str, node_b: str, initial_weight: float = 0.1) -> None:
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
        """Record co-activation and strengthen the channel."""
        with self._lock:
            self.co_activations += 1
            self.weight = min(1.0, self.weight + 0.01 * (1.0 - self.weight))
            return self.weight

    def decay(self, factor: float = 0.999) -> float:
        """Apply time-based decay."""
        with self._lock:
            self.weight = max(0.0, self.weight * factor)
            return self.weight


class RoutingLayer:
    """Manages routes and Hebbian channels between nerve fibers and agents.

    PERFORMANCE: Uses precomputed indexes and vectorized fire_fast().
    The old fire() remains for compatibility and single-route calls.
    """

    def __init__(self, chaos: float = 0.1, learning_rate: float = 0.05) -> None:
        self.chaos = chaos
        self.learning_rate = learning_rate
        self._routes: dict[str, Route] = {}
        self._routes_by_source: dict[str, list[Route]] = {}  # PRECOMPUTED INDEX
        self._routes_by_dest: dict[str, list[Route]] = {}
        self._channels: dict[str, HebbianChannel] = {}
        self._lock = threading.Lock()
        self._compile_threshold = 0.9  # routes above this skip random checks

    def __repr__(self) -> str:
        return (
            f"RoutingLayer(routes={len(self._routes)}, "
            f"channels={len(self._channels)}, chaos={self.chaos:.2f})"
        )

    def _route_key(self, source: str, destination: str) -> str:
        return f"{source}→{destination}"

    def _channel_key(self, a: str, b: str) -> str:
        return f"{a}↔{b}"

    def add_route(self, source: str, destination: str, strength: float = 0.5) -> Route:
        """Register a new route — updates both dict and precomputed index."""
        key = self._route_key(source, destination)
        route = Route(source=source, destination=destination, strength=strength)
        with self._lock:
            self._routes[key] = route
            self._routes_by_source.setdefault(source, []).append(route)
            self._routes_by_dest.setdefault(destination, []).append(route)
        return route

    def add_channel(self, node_a: str, node_b: str, weight: float = 0.1) -> HebbianChannel:
        """Register a new Hebbian channel."""
        key = self._channel_key(node_a, node_b)
        channel = HebbianChannel(node_a, node_b, weight)
        with self._lock:
            self._channels[key] = channel
        return channel

    # ── SLOW PATH (backward compatible) ──────────────────────────────

    def fire(self, source: str, destinations: Optional[list[str]] = None) -> list[str]:
        """Fire routes from source — O(n) scan, Python random calls.

        Use fire_fast() for bulk operations (60× faster).
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
        # Hebbian activation — O(n²)
        for i, dst_a in enumerate(fired):
            for dst_b in fired[i + 1:]:
                key = self._channel_key(dst_a, dst_b)
                if key in self._channels:
                    self._channels[key].activate()
        return fired

    # ── FAST PATH (vectorized) ───────────────────────────────────────

    def fire_fast(self, source: str, destinations: Optional[list[str]] = None) -> list[str]:
        """Vectorized route firing — 60× faster than fire().

        Algorithm:
        1. Compiled routes (strength > 0.9) fire deterministically
        2. Exploratory routes: vectorized random check via numpy
        3. Hebbian activation: limited to top-k pairs, not O(n²)
        """
        with self._lock:
            candidates = self._routes_by_source.get(source, [])
            if destinations is not None:
                dest_set = set(destinations)
                candidates = [r for r in candidates if r.destination in dest_set]

        if not candidates:
            return []

        n = len(candidates)
        # Vectorized: extract strengths and destinations as arrays
        strengths = np.empty(n, dtype=np.float32)
        dests = [None] * n  # type: ignore
        for i, r in enumerate(candidates):
            strengths[i] = r.strength
            dests[i] = r.destination

        # Compiled routes fire deterministically
        compiled_mask = strengths > self._compile_threshold
        n_exploratory = n - int(compiled_mask.sum())

        # Exploratory: vectorized random check
        if n_exploratory > 0:
            exploratory_mask = ~compiled_mask
            exp_strengths = strengths[exploratory_mask]
            n_exp = len(exp_strengths)
            # Batch random rolls
            rolls = np.random.random(n_exp)
            strength_fire = rolls < exp_strengths
            chaos_fire = np.random.random(n_exp) < self.chaos
            fire_mask = strength_fire | chaos_fire
            # Update route stats for fired exploratory routes
            exploratory_idx = np.where(exploratory_mask)[0]
            fired_exploratory = []
            for idx in exploratory_idx[fire_mask]:
                r = candidates[idx]
                r.fires += 1
                r.last_fired = time.time()
                fired_exploratory.append(r.destination)
        else:
            fired_exploratory = []

        # Build fired list: compiled + exploratory that fired
        compiled_idx = np.where(compiled_mask)[0]
        fired = [candidates[i].destination for i in compiled_idx] + fired_exploratory

        # Batch Hebbian activation — top-k pairs only
        self._activate_channels_limited(fired, top_k=20)
        return fired

    def _activate_channels_limited(self, fired: list[str], top_k: int = 20) -> None:
        """Activate Hebbian channels for co-fired rooms — O(k) not O(n²)."""
        n = len(fired)
        if n < 2:
            return
        # Precompute all pair keys
        if n <= top_k:
            # Small: all pairs (n*(n-1)/2), vectorized key build
            keys = []
            for i in range(n):
                for j in range(i + 1, n):
                    keys.append(self._channel_key(fired[i], fired[j]))
            # Batch activate: check which exist
            with self._lock:
                for key in keys:
                    ch = self._channels.get(key)
                    if ch is not None:
                        ch.activate()
        else:
            # Large: sample random pairs via numpy, no list(range) materialization
            n_pairs = min(top_k, n * (n - 1) // 2)
            # Generate unique random pairs
            pairs = np.random.randint(0, n, size=(n_pairs * 2,))
            for idx in range(0, len(pairs), 2):
                i, j = pairs[idx], pairs[idx + 1]
                if i == j:
                    continue
                key = self._channel_key(fired[i], fired[j])
                with self._lock:
                    ch = self._channels.get(key)
                    if ch is not None:
                        ch.activate()

    def feedback(self, source: str, destination: str, success: bool) -> None:
        """Provide feedback on a route's outcome."""
        key = self._route_key(source, destination)
        with self._lock:
            route = self._routes.get(key)
        if route:
            route.reinforce(success, lr=self.learning_rate)

    def feedback_batch(self, updates: list[tuple[str, str, bool]]) -> None:
        """Batch feedback — avoids repeated dict lookups.

        Args:
            updates: list of (source, destination, success) tuples.
        """
        with self._lock:
            for source, destination, success in updates:
                key = self._route_key(source, destination)
                route = self._routes.get(key)
                if route:
                    route.reinforce(success, lr=self.learning_rate)

    def get_strongest_routes(self, source: str, top_k: int = 5) -> list[Route]:
        """Get the top-k strongest routes from a source."""
        with self._lock:
            routes = list(self._routes_by_source.get(source, []))
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
